"""Check the CMU LiteLLM gateway: list available models and send one tiny image request.
Usage:
  export LITELLM_API_KEY=...        # never paste the key into code or commit it
  python check_gateway.py                    # list models
  python check_gateway.py --model gpt-4o     # also test that this model accepts images
"""
import argparse, base64, io, os, sys
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base_url", default="https://ai-gateway.andrew.cmu.edu")
ap.add_argument("--model")
a = ap.parse_args()
key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not key:
    sys.exit("Set LITELLM_API_KEY first")
client = OpenAI(api_key=key, base_url=a.base_url)

print("Available models:")
for m in sorted(x.id for x in client.models.list().data):
    print("  ", m)

if a.model:
    from PIL import Image
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    im.paste((255, 0, 0), (0, 0, 32, 64))          # red on the LEFT half
    buf = io.BytesIO(); im.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.chat.completions.create(model=a.model, max_tokens=64, temperature=0, messages=[{
        "role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Is the red half on the left or the right? Answer with one word."}]}])
    print(f"\n{a.model} says: {r.choices[0].message.content!r}  (expected: left)")
