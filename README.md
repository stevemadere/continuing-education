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

Sure, let's provide a more detailed overview of the initialization parameters for the `QLoRAContinuingTrainer` class, directly leveraging the comprehensive information from the class docstrings:

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

## Contributing

Contributions to `continuing_education` are welcome! Feel free to open an issue if you encounter problems or want to suggest features.

