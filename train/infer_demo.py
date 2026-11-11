"""Collect the fine-tuned chat model's replies for the demo conversation, in context.

Each turn is built by the same code the agent runs (agent/context.py): the model sees the conversation so far,
including its own earlier replies, exactly as it would behind the server. K samples are drawn per turn; every
sample is checked by the style gate, and the first clean one is carried forward as the reply.

Colab (T4 is enough for 4-bit E4B):
    !pip -q install unsloth jieba opencc-python-reimplemented
    from google.colab import drive; drive.mount('/content/drive')
    !git clone https://github.com/SihyeonJeon/writewell /content/writewell
    %cd /content/writewell
    !python train/infer_demo.py

Env:  WORK (default /content/drive/MyDrive/writewell)   MODEL (default unsloth/gemma-4-E4B-it)   TAG
      CKPT   checkpoint or adapter dir; default = lowest eval loss under <run>/ckpt, else <run>/adapter
      K (6)  samples per turn   SEED (3407)   STYLE (finetuned | instructed)   PIN  json {user message: reply} to fix turns
Out:  <WORK>/demo/chat_answers.json      {user message: chosen reply}   -> CHAT_ANSWERS_FILE for the server
      <WORK>/demo/chat_candidates.json   every sample with its gate result, and the exact context it was drawn from
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

E = os.environ.get
MODEL = E("MODEL", "unsloth/gemma-4-E4B-it")
WORK = Path(E("WORK", "/content/drive/MyDrive/writewell"))
RUN = WORK / "runs" / (MODEL.split("/")[-1] + ("-" + E("TAG") if E("TAG") else ""))
K, STYLE, SEED = int(E("K", "6")), E("STYLE", "finetuned"), int(E("SEED", "3407"))


def best_checkpoint():
    """Lowest eval loss among the checkpoints still on disk; the finished adapter if there are none."""
    if E("CKPT"):
        return E("CKPT"), None
    best = (None, None)
    for d in sorted((RUN / "ckpt").glob("checkpoint-*")):
        state = d / "trainer_state.json"
        if not state.exists():
            continue
        step = int(d.name.split("-")[1])
        losses = [h["eval_loss"] for h in json.loads(state.read_text())["log_history"] if "eval_loss" in h and h.get("step") == step]
        if losses and (best[1] is None or losses[-1] < best[1]):
            best = (str(d), losses[-1])
    return best if best[0] else (str(RUN / "adapter"), None)


def typed(messages):
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


def tidy(text):
    text = re.sub(r"<[^>]{1,30}>", "", text).strip()
    return text.split("\n\n")[0].strip()[:160]


def main():
    from unsloth import FastModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template
    import torch

    from agent import chat, context
    from gate.check import check

    ckpt, loss = best_checkpoint()
    print("checkpoint:", ckpt, "| eval loss:", loss)
    model, tok = FastModel.from_pretrained(model_name=ckpt, dtype=None, max_seq_length=1024, load_in_4bit=True, full_finetuning=False)
    tok = get_chat_template(tok, chat_template="gemma-4")
    FastModel.for_inference(model)

    log = []

    def sample(system, window):
        msgs = ([{"role": "system", "content": system}] if system else []) + window
        try:
            ids = tok.apply_chat_template(typed(msgs), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        except Exception:                      # template without a system turn: fold it into the first message
            msgs = [{"role": "user", "content": system + "\n\n" + window[0]["content"]}] + window[1:]
            ids = tok.apply_chat_template(typed(msgs), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        ids = ids.to(model.device)
        outs = []
        for seed in range(K):
            torch.manual_seed(SEED + 1000 * len(log) + seed)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=120, do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
            outs.append(tidy(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))
        return msgs, outs

    pin = json.loads(Path(E("PIN")).read_text()) if E("PIN") else {}
    script = json.loads((ROOT / "demo" / "script.json").read_text())
    history, answers = [], {}
    for user in script:
        history.append({"role": "user", "content": user})
        if chat.WRITE_RE.search(user):                       # writing turns are handed off; the chat model is not asked
            history.append({"role": "assistant", "content": chat.ACK})
            continue
        system, window = context.chat(history, None, STYLE)
        msgs, outs = sample(system, window)
        cands = [{"text": t, "banned": [h["gram"] for h in check(t)["hits"] if h["kind"] == "ban"]} for t in dict.fromkeys(outs) if t]
        said = {m["content"] for m in history}                    # never an echo of the user or a repeat of an earlier reply
        cands = [c for c in cands if c["text"] not in said]
        ok = [c for c in cands if not c["banned"] and 2 <= len(c["text"]) <= 80]
        chosen = pin.get(user) or (ok[0]["text"] if ok else cands[0]["text"] if cands else "")
        answers[user] = chosen
        history.append({"role": "assistant", "content": chosen})
        log.append({"user": user, "context": msgs, "candidates": cands, "chosen": chosen})
        print("\n>", user)
        for c in cands:
            print("  ", "*" if c["text"] == chosen else " ", c["text"], ("  [gate: " + "、".join(c["banned"]) + "]") if c["banned"] else "")

    out = WORK / "demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chat_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1))
    (out / "chat_candidates.json").write_text(json.dumps({"checkpoint": ckpt, "eval_loss": loss, "style": STYLE, "turns": log}, ensure_ascii=False, indent=1))
    print("\nwrote", out / "chat_answers.json")


if __name__ == "__main__":
    main()
