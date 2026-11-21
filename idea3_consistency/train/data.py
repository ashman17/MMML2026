import io
import json
import math
from dataclasses import dataclass

from PIL import Image, ImageOps

from mmsi_consistency import LETTERS, letter_score_prompt, option_letters


@dataclass
class TrainSample:
    row: dict
    flip: bool
    label: str


def make_train_samples(rows, mirror_aug=False):
    samples = []
    for row in rows:
        samples.append(TrainSample(row=row, flip=False, label=row["answer"]))
        mirror_answer = row.get("mirror_answer")
        if mirror_aug and mirror_answer is not None and not (
                isinstance(mirror_answer, float) and math.isnan(mirror_answer)):
            samples.append(TrainSample(row=row, flip=True, label=mirror_answer))
    return samples


def validate_letter_tokens(processor):
    token_ids = []
    for letter in LETTERS:
        encoded = processor.tokenizer.encode(letter, add_special_tokens=False)
        assert len(encoded) == 1, f"{letter} must encode to one token"
        token_ids.append(encoded[0])
    return token_ids


def _image(value, flip, max_side):
    raw = value["bytes"] if isinstance(value, dict) else value
    image = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return ImageOps.mirror(image) if flip else image


def _messages(sample, max_side):
    content = []
    for index, value in enumerate(sample.row["images"]):
        content.append({"type": "text", "text": f"Image {index + 1}:"})
        content.append({"type": "image", "image": _image(value, sample.flip, max_side)})
    letters = option_letters(sample.row["question"])
    content.append({"type": "text", "text": sample.row["question"]
                    + letter_score_prompt(letters)})
    return [{"role": "user", "content": content}]


def _array(values, floating=False):
    try:
        import torch
    except ImportError:
        return values
    torch_dtype = torch.float32 if floating else torch.long
    return torch.tensor(values, dtype=torch_dtype)


def build_inputs(processor, samples, max_side):
    conversations = [_messages(sample, max_side) for sample in samples]
    encoded = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, padding=True,
        return_dict=True, return_tensors="pt")
    labels = [LETTERS.index(sample.label) for sample in samples]
    n_options = [len(option_letters(sample.row["question"])) for sample in samples]
    bias = []
    for sample, count in zip(samples, n_options):
        scores = sample.row["noimage_logprobs"]
        if isinstance(scores, str):
            scores = json.loads(scores)
        bias.append([float(scores[letter]) if index < count else float("-inf")
                     for index, letter in enumerate(LETTERS)])
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "pixel_values": encoded["pixel_values"],
        "image_grid_thw": encoded["image_grid_thw"],
        "labels": _array(labels),
        "bias": _array(bias, floating=True),
        "n_options": _array(n_options),
    }
