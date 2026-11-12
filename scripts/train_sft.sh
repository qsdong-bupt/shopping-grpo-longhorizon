#!/usr/bin/env bash
# V2.0 SFT 训练入口（veRL 0.9.1 fsdp_sft + ShoppingMultiTurnSFTDataset）
# 单卡 A800 80GB、BF16 全参数微调当前 Qwen3.5-4B 教材。
#
# train/val 按成功轨迹类别20:1划分，1280/64条。每卡micro-batch=1，动态batch全局8条，
# 单步最多16K token；梯度检查点降低长序列激活显存。默认只保存最终HF模型且不自动续训。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$ROOT/outputs/swanlog}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY must be set to upload SFT metrics to SwanLab" >&2
  exit 1
fi

RUN_NAME="qwen35_4b_curated1344_1epoch_swanlab_20261112"
DATA_DIR="$ROOT/outputs/sft_dataset/deepseek_curated1344_16k_jev070_20261111/split_20to1"
MODEL_DIR="$ROOT/models/Qwen3.5-4B"
CHECKPOINT_DIR="$ROOT/checkpoints/$RUN_NAME"

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m shopping_grpo.training.sft.trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.custom_cls.path=pkg://shopping_grpo.training.sft.multiturn_dataset \
  data.custom_cls.name=ShoppingMultiTurnSFTDataset \
  data.max_length=16384 \
  data.truncation=error \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=true \
  data.max_token_len_per_gpu=16384 \
  data.micro_batch_size_per_gpu=2 \
  data.train_batch_size=8 \
  data.num_workers=2 \
  model.path="$MODEL_DIR" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  model.use_remove_padding=true \
  +model.override_config.attn_implementation=sdpa \
  engine.strategy=fsdp \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.use_torch_compile=false \
  optim.optimizer=AdamW \
  optim.optimizer_impl=torch.optim \
  optim.lr=1e-5 \
  optim.weight_decay=0.01 \
  optim.lr_warmup_steps_ratio=0.03 \
  checkpoint.save_contents='[hf_model]' \
  checkpoint.load_contents='[hf_model]' \
  trainer.project_name=shopping-sft \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.total_epochs=1 \
  trainer.logger="['console','swanlab']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=-1 \
  trainer.max_ckpt_to_keep=1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.seed=20261111 \
  "$@"
