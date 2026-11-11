# writewell

An agent that talks like a person and writes like one, in Traditional Chinese (Taiwan) first.

- **Chat** goes to a small fine-tuned model (Gemma, trained on pre-LLM community text).
- **Writing** goes to a large model whose output is held to a measured style gate, then shown in its own panel.
- Routing is a plain rule today (`agent/chat.py`); `route/jev.py` holds the decision-model version.

## The style gate is measured, not hand-written

`gate/zh-Hant.json` is built by `gate/build_lexicon.py` from three corpora:

| corpus | size | role |
|---|---|---|
| pre-LLM Taiwanese text (Common Crawl captures dated 2018–2021) | 6,655 docs, 19 sources, 10 genres | what people write |
| machine-written articles | 1,097 docs across several subject areas | what a tuned writer over-uses |
| raw model output on the same titles as human posts | 143 docs | what a model over-uses by default |

A word or word sequence is gated only if it is over-used in every subject area of the machine corpus,
at more than twice the human rate in every one of the ten human genres. Subject words are removed by two
guards (uniform use across subject areas; nouns need confirmation from topic-matched raw output).
Below 8% of human documents it is banned outright, otherwise capped at 1.5x the human rate.
Sentence-length spread and question density have floors and ceilings taken from human articles.

No instruction sent to a model contains a gated word: `tests/test_prompts.py` fails the build otherwise.
The writer is told about form (how sentence length varies, how people connect events), never given a word list.

## Run

```bash
python -m venv .venv && .venv/bin/pip install jieba opencc-python-reimplemented numpy pytest
.venv/bin/python -m pytest -q tests
.venv/bin/python -m web.server            # http://localhost:8787
```

- `CHAT_BASE_URL`, `CHAT_MODEL`: any OpenAI-compatible endpoint for the chat model (e.g. Ollama at `http://localhost:11434/v1`).
- `ANTHROPIC_API_KEY`: used for writing; without it the local `claude` CLI is used.
- Memes: copy `memes/manifest.example.json` to `memes/manifest.json` and put the image files next to it. Images are not in the repo.

## Layout

```
collect/   Common Crawl collectors for the human corpus
calc/      n-gram distributions, two-model likelihood ratio, corpus comparison
gate/      lexicon builder, lexicon, checker
agent/     model access, controlled writer, chat persona and routing
route/     decision-model routing policy (not wired into the demo yet)
train/     chat-model fine-tuning
web/       demo server and page
```

Corpora are not included.
