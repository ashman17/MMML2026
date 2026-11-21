import argparse
import io
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import build_inputs, make_train_samples, validate_letter_tokens
from losses import ce_loss, learned_mixin_loss, letter_logits


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VAL_EVERY = 100
VAL_SUBSET = 200
TARGET_MODULES = (r".*language_model.*\."
                  r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")


def _model_inputs(batch, device):
    return {key: batch[key].to(device) for key in
            ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


def _last_positions(attention_mask):
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask * positions.reshape(1, -1)).argmax(dim=1)


def _forward_scores(model, batch, letter_token_ids, mixin_head, loss_name, ent_weight):
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    labels = batch["labels"].to(device)
    bias = batch["bias"].to(device)
    n_options = batch["n_options"].to(device)
    inputs = _model_inputs(batch, device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    rows = torch.arange(labels.shape[0], device=device)
    positions = _last_positions(inputs["attention_mask"])
    logits_last = outputs.logits[rows, positions].float()
    hidden_last = outputs.hidden_states[-1][rows, positions]
    z = letter_logits(logits_last, letter_token_ids, n_options)
    if loss_name == "ce":
        return ce_loss(z, labels), z, torch.zeros(labels.shape[0], device=device)
    g = F.softplus(mixin_head(hidden_last).reshape(-1)).float()
    loss, combined = learned_mixin_loss(z, bias, labels, g, ent_weight)
    return loss, combined, g


def _validation_accuracy(model, samples, processor, letter_token_ids, batch_size, max_side):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(samples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda items: build_inputs(processor, items, max_side))
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device
            inputs = _model_inputs(batch, device)
            outputs = model(**inputs, use_cache=False)
            rows = torch.arange(batch["labels"].shape[0], device=device)
            positions = _last_positions(inputs["attention_mask"])
            z = letter_logits(outputs.logits[rows, positions].float(), letter_token_ids,
                              batch["n_options"].to(device))
            predictions = z.argmax(dim=-1)
            labels = batch["labels"].to(device)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    model.train()
    return correct / total


def _log_validation(path, model, val_samples, processor, letter_token_ids, batch_size,
                    max_side, step, train_loss, mean_g):
    accuracy = _validation_accuracy(model, val_samples, processor, letter_token_ids,
                                    batch_size, max_side)
    record = {"step": step, "train_loss": train_loss, "mean_g": mean_g,
              "val_accuracy": accuracy}
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record))


def train(model, processor, train_rows, val_rows, args, smoke=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor.tokenizer.padding_side = "left"
    letter_token_ids = validate_letter_tokens(processor)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    if any("visual" in name for name, parameter in model.named_parameters()
           if parameter.requires_grad):
        raise RuntimeError("LoRA must not target the vision tower")

    device = torch.device("cuda" if torch.cuda.is_available() and not smoke else "cpu")
    dtype = torch.float32 if smoke else torch.bfloat16
    model.to(device=device, dtype=dtype)
    hidden_size = model.config.text_config.hidden_size
    mixin_head = nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)
    train_samples = make_train_samples(train_rows, args.mirror_aug)
    val_samples = make_train_samples(val_rows, False)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_samples, batch_size=args.batch, shuffle=True, generator=generator,
        collate_fn=lambda items: build_inputs(processor, items, args.max_side))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += list(mixin_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")
    open(log_path, "w").close()

    model.train()
    optimizer.zero_grad()
    global_step = micro_step = 0
    loss_window = []
    g_window = []
    last_logged = 0
    stop = False
    for _ in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            loss, _, g = _forward_scores(
                model, batch, letter_token_ids, mixin_head, args.loss, args.ent_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss is not finite")
            (loss / args.grad_accum).backward()
            loss_window.append(float(loss.detach()))
            g_window.append(float(g.detach().mean()))
            micro_step += 1
            end_of_loader = batch_index + 1 == len(loader)
            if micro_step % args.grad_accum != 0 and not end_of_loader:
                continue
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % VAL_EVERY == 0:
                _log_validation(
                    log_path, model, val_samples[:VAL_SUBSET], processor, letter_token_ids, args.batch,
                    args.max_side, global_step, sum(loss_window) / len(loss_window),
                    sum(g_window) / len(g_window))
                loss_window.clear()
                g_window.clear()
                last_logged = global_step
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if global_step != last_logged:
        _log_validation(
            log_path, model, val_samples, processor, letter_token_ids, args.batch,
            args.max_side, global_step, sum(loss_window) / len(loss_window),
            sum(g_window) / len(g_window))
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    torch.save(mixin_head.state_dict(), os.path.join(adapter_dir, "mixin_head.pt"))
    if not smoke:
        merged_dir = os.path.join(args.out, "merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
    return adapter_dir


class _SmokeTokenizer:
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        return [{"A": 10, "B": 11, "C": 12, "D": 13}[text]]


class _SmokeProcessor:
    def __init__(self):
        self.tokenizer = _SmokeTokenizer()

    def apply_chat_template(self, conversations, **kwargs):
        import numpy as np
        import torch

        sequences = []
        pixel_values = []
        grids = []
        for conversation in conversations:
            sequence = [1]
            images = [part["image"] for part in conversation[0]["content"]
                      if part["type"] == "image"]
            for image in images:
                sequence += [121, 120, 120, 120, 120, 122]
                array = np.asarray(image.resize((4, 4)), dtype=np.float32) / 255.0
                for y in range(0, 4, 2):
                    for x in range(0, 4, 2):
                        patch = array[y:y + 2, x:x + 2].transpose(2, 0, 1).reshape(-1)
                        pixel_values.append(patch)
                grids.append([1, 2, 2])
            sequences.append(sequence + [2, 3, 4])
        length = max(map(len, sequences))
        input_ids = [[0] * (length - len(sequence)) + sequence for sequence in sequences]
        attention = [[0] * (length - len(sequence)) + [1] * len(sequence)
                     for sequence in sequences]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "pixel_values": torch.tensor(np.asarray(pixel_values), dtype=torch.float32),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long),
        }


def _smoke_rows():
    from PIL import Image

    rows = []
    for index in range(4):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (30 * index, 20, 200)).save(buffer, "PNG")
        rows.append({
            "id": index,
            "question": f"Which value is shown?\nOptions: A: {index}, B: other",
            "answer": "A" if index % 2 == 0 else "B",
            "images": [buffer.getvalue()],
            "question_type": "smoke",
            "mirror_answer": None,
            "noimage_logprobs": json.dumps({"A": math.log(0.55), "B": math.log(0.45)}),
        })
    return rows


def run_smoke(args):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "use_cache": False,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "in_channels": 3, "patch_size": 2, "spatial_merge_size": 1,
            "temporal_patch_size": 1, "window_size": 8, "out_hidden_size": 64,
            "fullatt_block_indexes": [0],
        },
        image_token_id=120, video_token_id=123, vision_start_token_id=121,
        vision_end_token_id=122)
    model = Qwen2_5_VLForConditionalGeneration(config)
    args.out = args.out or tempfile.mkdtemp(prefix="idea3_lora_smoke_")
    args.max_steps = 2
    args.epochs = 2
    args.grad_accum = 1
    args.batch = 1
    args.max_side = 4
    adapter_dir = train(model, _SmokeProcessor(), _smoke_rows(), _smoke_rows(), args, smoke=True)
    print(f"smoke passed: finite loss, adapter saved to {adapter_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--loss", choices=["ce", "lmh"], default="lmh")
    parser.add_argument("--mirror_aug", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--ent_weight", type=float, default=0.1)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        run_smoke(args)
        return
    if not args.train or not args.val or not args.out:
        raise SystemExit("--train, --val, and --out are required")

    import pandas as pd
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    train_rows = pd.read_parquet(args.train).to_dict("records")
    val_rows = pd.read_parquet(args.val).to_dict("records")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    train(model, processor, train_rows, val_rows, args)


if __name__ == "__main__":
    main()
