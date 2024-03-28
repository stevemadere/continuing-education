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

  if [ -n "$SHOULD_CONTINUE_TRAINING" ]
  then
    while /root/domain_tune_llm_qlora.py
    do
      echo "Training exited gracefully.  May have completed a dataset segment.  Restarting."
    done
  else
    /root/domain_tune_llm_qlora.py
  fi

  if [ -n "SHOULD_DESTROY_INSTANCE" ]
  then
    echo "Training finished.  Destroying this instance"
    vastai destroy instance "$CONTAINER_ID"
  fi
fi

