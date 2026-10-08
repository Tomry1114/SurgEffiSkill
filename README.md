# SurgEffiSkill

Surgical skill assessment from laparoscopic video (four competencies, 1-5 ordinal).

## Environment

```bash
# match torch to your CUDA (this is the tested build)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install \
    transformers==4.57.3 \
    peft==0.18.0 \
    accelerate==1.12.0 \
    numpy==1.26.4 \
    scipy==1.15.3 \
    qwen-vl-utils
# flash-attn is optional; without it attention falls back to sdpa
# GRPO (stage 2) additionally needs ms-swift:  pip install ms-swift
```

## Base model

Download `Qwen2.5-VL-7B-Instruct` into the repo root (or point `--base` / the `PROJECT`
path in the scripts at it):

```bash
# HuggingFace (prepend HF_ENDPOINT=https://hf-mirror.com if HF is blocked)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Qwen2.5-VL-7B-Instruct

# or ModelScope
pip install -U modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir Qwen2.5-VL-7B-Instruct
```

The videos, splits and frozen features are patient data and are not included; the paths in
`esr.py` and the pipeline scripts point at the authors' cluster and need adjusting.

## Training

All hyper-parameters default to the reported configuration, so training is one command:

```bash
python surgeffiskill/surgmind/esr/esr.py --tag esr
```

SOR is the default ordinal supervision; pass `--no-sor` to use the soft-label-CE + Huber
term instead.

## GRPO

Stage-2 RL from a LoRA checkpoint, run through ms-swift. The reward has two terms, defined in `surgeffiskill/surgmind/grpo/reward_plugin.py` and registered via
`--external_plugins`:

```bash
swift rlhf --rlhf_type grpo \
    --model Qwen2.5-VL-7B-Instruct \
    --adapters <lora_checkpoint> \
    --train_type lora --target_modules q_proj k_proj v_proj o_proj \
    --freeze_vit true --freeze_aligner true \
    --dataset splits82/train_grpo.jsonl \
    --external_plugins surgeffiskill/surgmind/grpo/reward_plugin.py \
    --reward_funcs surg_ordinal surg_format --reward_weights 1.0 0.5 \
    --num_generations 4 --learning_rate 1e-5 --num_train_epochs 1
```

`surg_ordinal` decodes the predicted levels and scores their ordinal distance to the
ground truth (uses `surgeffiskill/surgmind/surg_rubric.json`); `surg_format` scores answer format.

## Inference

Generation-readout evaluation of a trained LoRA checkpoint, per competency:

```bash
python surgeffiskill/surgmind/pipeline/infer_eval_perdim.py \
    --adapters <lora_checkpoint> --data splits82/val_pd.json \
    --tag esr --outdir results
```
