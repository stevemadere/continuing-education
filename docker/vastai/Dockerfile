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
    "bitsandbytes>=0.41.0" \
    "optimum>=1.12.0" \
    --no-cache-dir

RUN pip install "s3datasets>=2.1.0"
RUN pip install "continuing-education>=0.0.1a2"

# Just in case you also want to run inference on gptq models
RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ 

RUN python --version


# Copy scripts into the /root directory of the image
COPY onstart.sh /root/
COPY start_jupyter.bash /root/
COPY download_hf_model.bash /root/
COPY domain_tune_llm_qlora.py /root/

# Set any additional configurations or environment variables if needed
# For example:
# ENV MY_ENV_VARIABLE=my_value


# Optionally, you can specify the default command to run when the container starts
# CMD ["/bin/bash"]
