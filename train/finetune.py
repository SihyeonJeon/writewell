"""QLoRA fine-tune of Gemma 4 on real Taiwanese chat rows. Runs on Colab (see colab.ipynb), resumable.

Recipe follows Unsloth's Gemma 4 text guide (FastModel, gemma-4 chat template, loss on model turns only).
Everything persistent lives in WORK (Google Drive), so a dropped Colab session resumes from the last checkpoint.

Env:  MODEL (default unsloth/gemma-4-E4B-it)  WORK (default /content/drive/MyDrive/writewell)
      MAX_ROWS (0 = all)  EPOCHS (1)  BATCH (8)  ACCUM (2)  LR (2e-4)  RANK (16)  SEQ (1024)
"""
import json
import os
import time
from pathlib import Path

from unsloth import FastModel  # noqa: I001 - unsloth must be imported before transformers
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

E = os.environ.get
MODEL = E("MODEL", "unsloth/gemma-4-E4B-it")
WORK = Path(E("WORK", "/content/drive/MyDrive/writewell"))
RUN = WORK / "runs" / MODEL.split("/")[-1]
RUN.mkdir(parents=True, exist_ok=True)
SEQ = int(E("SEQ", "1024"))

# The same prompts before and after training, so the change in voice is visible side by side.
PROBES = ["今天上班被主管念了一頓，心情超差", "欸你覺得珍奶要半糖還是微糖", "我室友又把我的布丁吃掉了",
          "期中考完全沒讀，明天要考了怎麼辦", "推薦一下台北適合一個人去的地方", "跟曖昧對象已讀不回三天了",
          "剛領薪水就想花光怎麼辦", "颱風假要幹嘛"]


def rows(name, limit=0):
    out = [json.loads(x) for x in open(WORK / "data" / f"{name}.jsonl")]
    return out[:limit] if limit else out


def typed(messages):
    """Gemma 4's processor is multimodal: message content must be a list of typed parts, not a string."""
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


def generate(model, tok, prompts):
    outs = []
    for p in prompts:
        ids = tok.apply_chat_template(typed([{"role": "user", "content": p}]), add_generation_prompt=True,
                                      tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=160, do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
        outs.append({"prompt": p, "reply": tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)})
    return outs


def main():
    model, tok = FastModel.from_pretrained(model_name=MODEL, dtype=None, max_seq_length=SEQ,
                                           load_in_4bit=True, full_finetuning=False)
    tok = get_chat_template(tok, chat_template="gemma-4")
    base_file = RUN / "samples_base.json"
    if not base_file.exists():
        base_file.write_text(json.dumps(generate(model, tok, PROBES), ensure_ascii=False, indent=1))

    r = int(E("RANK", "16"))
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     r=r, lora_alpha=r, lora_dropout=0, bias="none", random_state=3407)

    def fmt(batch):
        texts = [tok.apply_chat_template(typed(m), tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
                 for m in batch["messages"]]
        return {"text": texts}

    keep = lambda rs: Dataset.from_list([{"messages": x["messages"]} for x in rs]).map(fmt, batched=True)  # noqa: E731
    train, evals = keep(rows("chat", int(E("MAX_ROWS", "0")))), keep(rows("eval"))
    print("train rows", len(train), "eval rows", len(evals), "\n", train[0]["text"][:400])

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train, eval_dataset=evals,
        args=SFTConfig(
            dataset_text_field="text", output_dir=str(RUN / "ckpt"),
            per_device_train_batch_size=int(E("BATCH", "8")), gradient_accumulation_steps=int(E("ACCUM", "2")),
            per_device_eval_batch_size=8, num_train_epochs=float(E("EPOCHS", "1")),
            learning_rate=float(E("LR", "2e-4")), warmup_ratio=0.03, lr_scheduler_type="cosine",
            optim="adamw_8bit", weight_decay=0.001, seed=3407, report_to="none",
            logging_steps=20, eval_strategy="steps", eval_steps=300, save_steps=300, save_total_limit=2,
        ),
    )
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")

    t0 = time.time()
    before = trainer.evaluate()
    resume = any((RUN / "ckpt").glob("checkpoint-*"))
    stats = trainer.train(resume_from_checkpoint=resume)
    after = trainer.evaluate()

    model.save_pretrained(str(RUN / "adapter"))
    tok.save_pretrained(str(RUN / "adapter"))
    (RUN / "samples_tuned.json").write_text(json.dumps(generate(model, tok, PROBES), ensure_ascii=False, indent=1))
    report = {"model": MODEL, "train_rows": len(train), "eval_loss_before": before.get("eval_loss"),
              "eval_loss_after": after.get("eval_loss"), "train_loss": stats.training_loss,
              "minutes": round((time.time() - t0) / 60, 1), "gpu": torch.cuda.get_device_name(0),
              "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}
    (RUN / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if E("GGUF"):
        model.save_pretrained_gguf(str(RUN / "gguf"), tok, quantization_method="q4_k_m")


if __name__ == "__main__":
    main()
