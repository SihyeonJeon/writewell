#!/bin/bash
# Second stage: continue from the newest (lowest eval loss so far) checkpoint on the newest posts from all sources.
cd /content/drive/MyDrive/writewell || exit 1
export INIT=$(ls -d runs/gemma-4-12B-it/ckpt/checkpoint-* | sort -t- -k5 -n | while read d; do [ -f "$d/adapter_model.safetensors" ] && echo "$PWD/$d"; done | tail -1)
[ -n "$INIT" ] || { echo "no complete checkpoint found"; exit 1; }
echo "continuing from $INIT"
export MODEL=google/gemma-4-12B-it DATA=recent_train EVAL=recent_test TAG=recent EPOCHS=1 BATCH=16 ACCUM=1 LR=1e-4 EVAL_STEPS=250 MAX_ROWS=8000
python finetune_v2.py 2>&1 | tee runs/recent_run.log
