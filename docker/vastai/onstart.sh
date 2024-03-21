export DATA_DIR="${DATA_DIR:-$HOME/huggingface}"
env | grep _ >> /etc/environment;

cd /root

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

  if [ -n "$SHOULD_SYNC_CHECKPOINTS" ]
  then
    /root/sync_checkpoints.bash down || exit 1
  fi

  /root/domain_tune_llm_qlora.py

  if [ -n "$SHOULD_SYNC_CHECKPOINTS" ]
  then
    /root/sync_checkpoints.bash up || exit 1
  fi

fi

 
