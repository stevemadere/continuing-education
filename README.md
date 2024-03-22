# continuing_education

The `continuing_education` module provides for easily training large language models (LLMs) with Quantized Low Rank Adapters (QLoRA), focusing on network and storage efficiency and the ability to continue training from checkpoints. This is particularly useful in environments prone to interruptions or where network bandwidth and storage space is limited. e.g. vast.ai or salad.com instances.

## Features

- **Efficient Training with QLoRA**: Enhance your Hugging Face Transformer models with Low Rank Adapters, optimizing for both performance and memory efficiency.
- **Checkpointing for Continuation**: Seamlessly resume training from the last checkpoint, minimizing data reprocessing and model initialization times.
- **AWS Integration**: Leverage AWS S3 buckets for dataset storage, ensuring scalable and accessible data management.
- **Flexible Training Schemes**: Configure training sessions according to your specific needs, including setting steps per round, maximum steps, and segment-based training.
- **Automatic Tokenization and Dataset Preparation**: Utilize integrated tokenization and dataset management for a streamlined setup process.

## Installation

```bash
pip install continuing-education
```

## Usage

### Setting Up Environment Variables

Before using `continuing_education`, ensure the following environment variables are set:

- `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: For AWS authentication.

### Example Usage

```python
from continuing_education import QLoRAContinuingTrainer, logger
import os

# Ensure AWS credentials and bucket names are set
if 'AWS_PROFILE' not in os.environ and not ('AWS_SECRET_ACCESS_KEY' in os.environ and 'AWS_ACCESS_KEY_ID' in os.environ):
    raise EnvironmentError("AWS credentials required.")

dataset_bucket_name = os.environ['DATASET_BUCKET']
output_dir = os.environ['OUTPUT_DIR']

base_model_id = 'Mistralai/Mistral-7Bv0.1'  # Default model

# Initialize trainer with environment configurations
continuing_trainer = QLoRAContinuingTrainer(
    base_model_id=base_model_id,
    dataset_bucket_name=dataset_bucket_name,
    output_dir=output_dir,
    steps_per_round=10000,  # Example configuration
    max_steps=250000  # Example configuration
)

# Start training
continuing_trainer.train()
```

Note that it is up to you to download the resulting checkpoints.
An example script to do this with rsync to a vast.ai instance is included in the examples directory in the source repo:
[vast_sync.bash](https://github.com/stevemadere/continuing-education/blob/main/examples/vast_sync.bash)

## API Reference

### QLoRAContinuingTrainer

`QLoRAContinuingTrainer` is the core class for training models with QLoRA. It extends `BaseContinuingTrainer`, providing mechanisms to load models, prepare datasets, and manage training sessions effectively.

## Initialization Parameters

The `QLoRAContinuingTrainer` class initializes with the following parameters:

- `base_model_id (str)`: Identifier for the base model to which LoRA will be attached. This is the ID of a Hugging Face model that will serve as the starting point for training.

- `dataset_bucket_name (str)`: Name of the S3 bucket containing the training text documents.

- `output_dir (str, optional)`: Directory where checkpoints will be saved. Defaults to "/root/outputs". It is crucial for managing training interruptions and resumptions.

- `dataset_id (Union[str, None], optional)`: Key to an S3 object containing a JSON array of keys for training text objects. This is used if your dataset is a single, JSON document listing all of the keys of all of the training text documents.

- `dataset_series (Union[str, None], optional)`: Pattern for S3 keys of JSON objects specifying keys of objects for training. Must include "{segment_number}" if specified. This is useful for large datasets where the key list is split across multiple files. Defaults to None. Exactly one of `dataset_id` or `dataset_series` must be specified.

- `test_dataset_id (Union[str, None], optional)`: Key to an S3 object with a JSON array of keys for evaluation text objects.

- `steps_per_round (Union[int, None], optional)`: Maximum number of training steps per call to `train()`. This parameter can be used to limit the training duration per execution, while debugging/experimenting for quick turnaround.

- `max_seq_length (int, optional)`: Maximum token count per training example. This parameter is critical for memory management and ensuring the model can handle the inputs without exceeding GPU memory limits. Defaults to 2048.

- `max_steps (Union[int, None])`: Explicit maximum training steps. This parameter sets a hard limit on the number of training steps, providing a way to precisely control the training duration. It's necessary to specify this if not using `estimated_max_steps()`. Defaults to None.

- `save_steps (int, optional)`: Interval of training steps after which a checkpoint is automatically saved. Regular checkpoints are crucial for resuming training without losing progress. Defaults to 1000.


#### Methods

- `train()`: Starts the training process, utilizing checkpoints and dataset management to efficiently continue training sessions.

## Turnkey solution for training a QLoRA on vast.ai

1. Place a large number of text documents in an S3 bucket
2. Create a DATASET_SERIES in your bucket (detailed instructions in the [next section](#creating-a-dataset_series))
2. Login to your vast.ai account
3. Create a new vast template [here](https://cloud.vast.ai/templates/edit/)
    1. Choose the Image Path **stevemadere/vast_train_llm:qlora**
    2. Set your Docker Options to this (editing as necessary) 
    ```bash
    -e NOTEBOOK_DIR=/root/notebooks
    -e HUGGINGFACE_TOKEN=REPLACE_WITH_YOUR_HUGGINGFACE_API_TOKEN
    -e HF_MODEL_NAME=Mistral-7B-v0.1
    -e HF_CONTRIBUTOR=mistralai
    -e HF_MODEL_REVISION=main
    -e DATASET_BUCKET=THE_NAME_OF_YOUR_S3_BUCKET_CONTAINING_THE_TEXT_DOCUMENTS
    -e DATASET_SERIES=THE_PATTERN_FOR_YOUR_DATASET_SERIES
    -e CHECKPOINTS_BUCKET=THE_NAME_OF_YOUR_S3_BUCKET_FOR_SAVING_MODELS
    -e CHECKPOINTS_PREFIX=THE_PATH_IN_YOUR_BUCKET_FOR_THIS_SERIES_OF_CHECKPOINTS
    -e DATA_DIR=/root/huggingface
    -e AWS_ACCESS_KEY_ID=YOUR_IAM_CREDENTIALS
    -e AWS_SECRET_ACCESS_KEY=YOUR_IAM_CREDENTIALS
    -e OUTPUT_DIR=/root/outputs
    -e STEPS_PER_ROUND=10000
    -e SHOULD_DOWNLOAD_MODEL=YES
    -e SHOULD_START_TRAINING=YES
    -e SHOULD_SYNC_CHECKPOINTS=YES
    ```
    3. select **Run interactive shell server, SSH** and **Use direct SSH connection**
    4. Empty the "On-start Script" field as there is already an _onstart.sh_ script in the docker image
    5. Fill in a template name and description 
    6. press [SELECT AND SAVE]
4. Go to the Search tab and find a host with a RTX-4090 and high bandwidth (>500Mbps is best).  (Watch out for excessive bandwidth rates.  Some unscrupulous hosts try to pull a fast one with bandwidth rates exceeding $20/TB whereas most charge less than $3/TB)
5. RENT
6. Check to see that everything is working by switching to the Instances tab and pressing the [ >_ CONNECT ] button which will simply give you an ssh command to copy and paste to your local shell.

- It may take a while for the base model to finish downloading to your instance.  If you suspect that the base model download failed, you can examine the system logs from the instance card.
- Once the base model finishes downloading, a pair of log files should be created by the training process in /root/continuing_trainer.info.log and continuing_trainer.debug.log.  You can examine either of those to see what kind of progress the trainer is making.


### Creating a DATASET_SERIES
While the ContinuingTrainer can flexibly handle multiple methods of specifying the set of documents in your bucket to be used as a training corpus, the example script used in the pre-built docker image only allows for the DATASET_SERIES pattern method.

This is how you create a DATASET_SERIES:

1. Decide on a name pattern such as **my_datasets/pretraining_test_{segment_number}.json.gz**
2. Decide how many segments you want to have in your series.  One is fine.
3. Make a list of all of the S3 keys of all of the text documents in your dataset bucket that you want to use for your training dataset.
4. Split that list into as many segments as you chose in step 2.
5. Create compressed JSON files, each containing a list of document keys from one of those segments of the document ids list and each JSON file with a name matching the pattern you chose in step 1.
    e.g.:  **my_datasets/pretraining_test_1.json.gz**
6. Upload those files to your S3 bucket with keys matching the filenames (_aws s3 sync_ works well for this)


If you encounter problems, feel free to reach out to me on [linked-in]( https://linkedin.com/in/smadere)


## Contributing

Contributions to `continuing_education` are welcome! Feel free to open an issue if you encounter problems or want to suggest features.

