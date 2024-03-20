#!/bin/bash
# A script to download a model from huggingface if and only if it is not already fully downloaded.
# Set the env vars HF_CONTRIBUTOR, HF_MODEL_NAME, and HF_MODEL_REVISION before invoking 
# or you'll get Mistral-7B-Instruct quantized to 4bit by default
export DATA_DIR="${DATA_DIR:-$HOME/huggingface}"
export HF_CONTRIBUTOR="${HF_CONTRIBUTOR:-TheBloke}"
export HF_MODEL_NAME="${HF_MODEL_NAME:-Mistral-7B-Instruct-v0.1-GPTQ}"
export HF_MODEL_REVISION="${HF_MODEL_REVISION:-gptq-4bit-32g-actorder_True}"
export HF_REPO="${HF_CONTRIBUTOR}/${HF_MODEL_NAME}"
export MODEL_DIR="$DATA_DIR/$HF_MODEL_NAME"
export FINISHED_FLAG_FILE=$MODEL_DIR/download_completed.flag
if [ ! -e "$FINISHED_FLAG_FILE" ] ; then
  if ps auxwww | fgrep "huggingface-cli download" | grep -v grep; then
    echo "A huggingface download process is already running:"
    ps auxwww | fgrep "huggingface-cli download" | grep -v grep
    exit 1
  fi
  if [ -z "$HUGGINGFACE_TOKEN" ] ; then
    echo "HUGGINGFACE_TOKEN is not set"
    exit 1
  fi

  mkdir -p "$DATA_DIR" || { echo "failed to create DATA_DIR $DATA_DIR" ; exit 1 ; }
  mkdir -p "$MODEL_DIR" ||{ echo "failed to create MODEL_DIR $MODEL_DIR" ; exit 1 ; } 

  if [ -e "${MODEL_DIR}/model.safetensors.index.json" ] ; then
    RESUME_FLAG="--resume-download"
  else
    RESUME_FLAG="--force-download"
  fi

  cd "$DATA_DIR" || exit 1
  huggingface-cli login --token "$HUGGINGFACE_TOKEN"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  huggingface-cli download "${HF_REPO}" --revision "$HF_MODEL_REVISION" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False "${RESUME_FLAG}" || { echo "failed to download model" ; exit 1 ; }
  touch "$FINISHED_FLAG_FILE"
fi

