# Use the pytorch/pytorch:latest image as the base image
FROM pytorch/pytorch:latest


RUN apt-get update && apt-get install -y --no-install-recommends git vim

# Install required Python packages and fail if any package cannot be installed
RUN pip install "git+https://github.com/stevemadere/transformers@resume-qlora-training"

RUN pip install jupyterlab \
    scipy \
    huggingface_hub \
    hf_transfer \
    accelerate \
    awscli \
    "bitsandbytes>=0.41.0" \
    "optimum>=1.12.0" \
    --no-cache-dir

RUN pip install "s3datasets>=2.1.1"
RUN pip install "continuing-education>=0.0.2a1"

# Just in case you also want to run inference on gptq models
RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ 


# Copy scripts into the /root directory of the image
COPY onstart.sh /root/
COPY start_jupyter.bash /root/
COPY download_hf_model.bash /root/
COPY domain_tune_llm_qlora.py /root/
COPY sync_checkpoints.bash /root/

# Force a rebild of the final image layer in case any of the scripts have changed
RUN /bin/sh -c 'cksum /root/*.py /root/*.bash /root/*.sh'

# Set any additional configurations or environment variables if needed
# For example:
# ENV MY_ENV_VARIABLE=my_value


# Optionally, you can specify the default command to run when the container starts
# CMD ["/bin/bash"]
