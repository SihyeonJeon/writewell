# writewell

An agent that talks like a person and writes like one, in Traditional Chinese (Taiwan) first.

- **Chat** goes to a small fine-tuned model (Gemma, trained on pre-LLM community text).
- **Writing** goes to a large model whose output is held to a measured style gate, then shown in its own panel.
- Routing is a plain rule today (`agent/chat.py`); `route/jev.py` holds the decision-model version.

## The style gate is measured, not hand-written

`gate/zh-Hant.json` is built by `gate/build_lexicon.py` from three corpora:

| corpus | size | role |
|---|---|---|
| pre-LLM Taiwanese text (Common Crawl captures dated 2018 to 2021) | 6,655 docs, 19 sources, 10 genres | what people write |
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

Copy `.env.example` to `.env` and fill in what you have:

- `CHAT_BASE_URL`, `CHAT_MODEL`: any OpenAI-compatible endpoint serving the fine-tuned Gemma.
- `ANTHROPIC_API_KEY`: the frontier model used for writing; without it the local `claude` CLI is used.
- `OPENROUTER_API_KEY`: turns on the Jev decision model for routing (`route/jev.py`); without it a rule routes.
- `WW_REPLAY=demo/recording.json`: plays back a recording of real outputs at a watchable pace (`tools/record_demo.py` makes one).

## What is real in the demo

- **Chat replies** are outputs of our fine-tuned Gemma (gemma-4-12B, QLoRA, a mid-training checkpoint), drawn with
  `train/infer_demo.py`: 18 samples per turn, each checked by the gate. `demo/chat_candidates.json` holds every sample and
  the one used. Point `CHAT_BASE_URL` at a server hosting the model and the same turns run live.
- **The article** is written by a frontier model and passed through the gate; flagged sentences go back to the model for
  another turn. `demo/recording.json` holds the draft, each rewrite and the final text with their gate results.
- **Routing** in the demo is a rule. `route/jev.py` implements the decision-model version and switches on with its key.
- The fine-tuned model was trained on single exchanges, so it answers the current message alone (`agent/context.py`);
  conversation memory is kept by the agent and handed to the writer.

## Chat model weights

The fine-tuned adapter (LoRA for `google/gemma-4-12B-it`) is attached to the
[`gemma-adapter-v0.1` release](https://github.com/SihyeonJeon/writewell/releases/tag/gemma-adapter-v0.1)
as `writewell-gemma-4-12b-adapter.zip`. Load it on top of the base model (`train/infer_demo.py` shows how, `CKPT=<unzipped dir>`),
serve it behind any OpenAI-compatible server, and set `CHAT_BASE_URL` / `CHAT_MODEL` in `.env`: the agent then talks to it live.
The base model is under Google's Gemma terms of use.

This is a checkpoint from partway through training (step 2100, eval loss 3.22, still falling). We ran out of time before the
run finished, so the model is weaker than it will be: it answers one message at a time, and good replies come from sampling
several and keeping the ones that pass the gate.

### Before and after fine-tuning

The same 8 chat prompts (`PROBES` in `train/finetune.py`) were run on the base model and on the checkpoint.

| | base Gemma | after fine-tuning |
|---|---|---|
| form | every bit of small talk answered as a counseling report | the length and tone of a friend leaving a comment |
| script | one reply came back in Simplified Chinese | all 8 in Traditional Chinese |

## Reaction memes

`agent/skills/memes.py` answers with a meme when one fits and with none when nothing does. The library
(`memes/library/taiwan-v1/catalog.json`) comes from the social-monitoring collector (`collect/ig_trends.py`),
which records, per meme, the phrase, when it fits and when it doesn't.

Choosing stays cheap as the library grows: `python -m agent.skills.memes build` turns each card into one small
vector, and a turn costs one dot product per card. No image or card is read per turn. A card comes out only above
a minimum similarity; messages about real trouble (job loss, illness, grief) never get one.
`memes/pins.json` lists situations where a specific card must come out. Image files stay out of the repo.

## Layout

```
collect/   Common Crawl collectors for the human corpus
calc/      n-gram distributions, two-model likelihood ratio, corpus comparison
gate/      lexicon builder, lexicon, checker
agent/     model access, controlled writer, chat persona, routing, reaction skill
route/     decision-model routing policy (not wired into the demo yet)
train/     chat-model fine-tuning
web/       demo server and page
```

The corpora stay out of the repo.
