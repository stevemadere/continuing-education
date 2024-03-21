#!/usr/bin/env bash
export DATA_DIR="${DATA_DIR:-$HOME/huggingface}"
env | grep _ >> /etc/environment;

if [ -n "$SHOULD_DOWNLOAD_MODEL" ]
then
  /root/download_hf_model.bash || exit 1
fi

if [ -n "$SHOULD_START_JUPYTER" ]
then
  /root/start_jupyter.bash &
fi

if [ -n "$SHOULD_START_TRAINING" ]
then
  /root/domain_tune_llm_qlora.py
fi




