# MMSI-Bench human vs model evaluation

- 66 questions across the 11 categories; 6 per category (2 easy, medium, hard)
- 2 sets for review (set A: R1, R2; set B: R3, R4); each reviewer has 33 queestions.

## Setup

Initial setup:
```bash
python3 -m venv .venv
.venv/bin/pip install pyarrow==25.0.1
.venv/bin/python workflow.py download
.venv/bin/python workflow.py prepare --sample-only
.venv/bin/python workflow.py serve --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766/?reviewer=R1` (or `R2`, `R3`, `R4`) in a browser.
Answer the questions as a human reviewer, then compare with the model; explain its errors.
- Mike: R1
- Alex: R2
- Ashman: R3
- Vicky: R4


Relaunching the page:
```bash
.venv/bin/python workflow.py serve --host 127.0.0.1 --port 8766 --provider litellm --model gpt-5.6-terra
```

Use `previous` in case you mess up

NB: Don't commit `data/`


# Send reviews back

Reviewer run:
```bash
.venv/bin/python workflow.py export-review --reviewer R2 --out exports/R2.json
```

Then I'll run
```bash
.venv/bin/python workflow.py import-review --path exports/R2.json
```

## AI generation
NB: the reviewer interface (as well as saving reviewer answers in the databases) has been AI-generated. AI was used to help automate the querying of ChatGPT with LiteLLM.
