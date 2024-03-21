#!/usr/bin/env python
# coding: utf-8


import re
from typing import Union, cast
import torch
from .checkpoint_registry import CheckpointRegistry, CheckpointInfo
from .checkpoint_manager import CheckpointManager
from s3datasets import S3TextDataset
from datasets import Dataset, IterableDataset
from tokengenerators import TextDS2TokensGenerator

from peft.mapping import  get_peft_model
from peft.peft_model import PeftModel
from peft import LoraConfig # type: ignore
from peft.utils.other import prepare_model_for_kbit_training
from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

def set_up_logging():
    import logging
    import sys
    logger = logging.getLogger('standard')
    logger.setLevel(logging.DEBUG)

    info_fh = logging.FileHandler('continuing_trainer.info.log')
    info_fh.setLevel(logging.INFO)
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_fh.setFormatter(info_formatter)

    debug_fh=logging.FileHandler('continuing_trainer.debug.log')
    debug_fh.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s in %(funcName)s at %(filename)s:%(lineno)s')
    debug_fh.setFormatter(debug_formatter)

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.ERROR)


    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)


    logger.addHandler(info_fh)
    logger.addHandler(debug_fh)
    logger.addHandler(err_handler)
    logger.addHandler(out_handler)
    return logger

logger = set_up_logging()




def torchify_collator(batch):
    # Initialize empty lists to hold each attribute
    labels = []
    attention_mask = []

    # Iterate through each item in the batch, grouping by column
    batched_columns = {}
    for item in batch:
        for column_name,v in item.items():
           if not column_name in batched_columns:
               batched_columns[column_name] = []
           batched_columns[column_name].append(v)

        labels.append(item['labels'])
        attention_mask.append(item['attention_mask'])

    # Convert lists to tensors
    collated = {}
    for column_name,column_data in batched_columns.items():
        if isinstance(column_data[0], list):
            collated[column_name] = torch.tensor(column_data,dtype=torch.long)
        else:
            collated[column_name] = column_data
    return collated


class BaseContinuingTrainer(ABC):
    """
    An abstract base class for all ContinuingTrainer implementations.

    Manages the training of a LoRA on a huggingface Transformer Language Model from text documents in an S3 bucket.
    While the Huggingface Trainer does this just fine on its own in a single uninterrupted session, it's
    not practical for use in an environment where training may be interrupted half way through an enormous
    corpus of training documents.

    This class enables the easy continuation of training after an interruption without wasting huge amounts
    of time, network bandwidth, and disk storage re-downloading and then ignoring all of the text already
    seen in the training process. (which is what transformers.Trainer does)

    This class leverages s3datasets to manage all of the downloading and tokenizing/chunking of the
    documents in the corpus.

    In practice, this base class is not used directly but instead, a subclass such as QLoRAContinuingTrainer
    is instantiated.

    More subclasses can be implemented to enable different quantization and PEFT techniques.  e.g. LoftQ

    Each subclass of BaseContinuingTrainer must override the following abstract methods:
        - load_checkpoint_model(self,checkpont_path, base_model_id) to load a previously saved checkpoint
        - load_starting_model(self, base_model_id) to construct a starting model from the base_model_id.


    """
    base_model_id: str # the id of the base model which will have a LoRA attached to it
    dataset_bucket_name: str # The name of an S3 bucket containing text documents on which to train
    output_dir: str # The filesystem directory where checkpoints should be saved
    dataset_id: Union[str,None]  # A key to an S3 object containing a JSON array of keys of the text objects on which to train
    dataset_series: Union[str,None] # A pattern for S3 keys of json objects specifying keys of the specific objects to use in training
    test_dataset_id: Union[str,None] # A key to an S3 object containing a JSON array of keys of the text objects to use an an evaluation dataset
    steps_per_round: Union[int,None] # Maximum number of training steps to execute in a single call to train().  This is mostly useful for testing the process of interrupting and then continuing training.  Once that's been tested, just make this a very large number.
    max_seq_length: int # maximum number of tokens in a single training example (GPU memory constrains this )
    save_steps: int # Automatically make a checkpoint when this many training steps are completed
    explicit_max_steps: Union[int,None] # If the caller specified max_steps during construction, the value is stored here

    checkpoint_registry: CheckpointRegistry
    tokenizer: PreTrainedTokenizerBase
    base_model: PreTrainedModel
    model: PeftModel
    starting_step: Union[int,None]
    starting_checkpoint_info: Union[CheckpointInfo,None]
    text_dataset: Dataset
    train_tokens_generator: TextDS2TokensGenerator
    train_inputs: IterableDataset
    test_inputs: Union[Dataset, IterableDataset,None]
    checkpoint_manager: CheckpointManager

    def __init__(self,
                 base_model_id: str,
                 dataset_bucket_name: str,
                 output_dir:str = "/root/outputs",
                 dataset_id: Union[str,None] = None,
                 dataset_series: Union[str,None] = None,
                 test_dataset_id: Union[str,None] = None,
                 steps_per_round: Union[int,None] = None,
                 max_seq_length: int = 2048,
                 max_steps: Union[int,None] = None,
                 save_steps: int = 1000

                 ):
        if (dataset_id and dataset_series) or (dataset_id == None and dataset_series == None):
            raise ValueError("exactly one of dataset_id or dataset_series must be specified")
        if dataset_series and not '{segment_number}' in dataset_series:
            raise ValueError('dataset_series if specified must contain the substring "{segment_number}"')
        self.base_model_id = base_model_id
        self.dataset_bucket_name = dataset_bucket_name
        self.output_dir= output_dir
        self.dataset_id = dataset_id
        self.test_dataset_id = test_dataset_id
        self.dataset_series = dataset_series
        self.steps_per_round = steps_per_round
        self.max_seq_length = max_seq_length
        self.explicit_max_steps = max_steps
        self.save_steps = save_steps
        self.checkpoint_registry = CheckpointRegistry(output_dir=output_dir)
        self.starting_checkpoint_info = None

        self.prepare()

    @abstractmethod
    def load_checkpoint_model(self, checkpoint_path:str, base_model_id: str) -> PeftModel:
        """
            This method must be overridden by subclasses.
            Its purpose is to load the model to be trained from a previously saved checkpoint saved
            in the directory specified by the checkpoint_path parameter.

            e.g.  Load a QLoRA model from a checkpoint using the checkpoint_path and base_model_id
        """
        if checkpoint_path or base_model_id: # prevent PyRight warnings about unused params
            pass

    @abstractmethod
    def load_starting_model(self, base_model_id:str) -> PeftModel:
        """
            This method must be overridden by subclasses.
            Its purpose is to load/prepare the model to be trained from a base model.

            e.g.  Load and quantize the base model and attach a no-op LoRA to it for QLoRA training
        """
        if base_model_id: # prevent PyRight warnings about unused params
            pass

    def prepare(self):
        self._cuda_check()
        self._get_continuation_state()
        self._load_model()
        self._prepare_datasets()
        self._prepare_trainer()

    def train(self):
        # pre-requisites
        assert self.trainer
        assert not self.starting_step == None

        logger.debug(f'generator cursor before training is segment {self.dataset_segment_number()},  {self.train_tokens_generator.get_cursor().to_dict()}')
        try:
            if self.starting_checkpoint_info:
                logger.info(f'Resuming training from checkpoint {self.starting_checkpoint_info.path()}')
                self.trainer.train(resume_from_checkpoint=self.starting_checkpoint_info.path(),
                              without_checkpoint_model=True) # We already loaded the model we want, don't screw it up
            else:
                logger.info(f'Starting training from the beginning.')
                self.trainer.train()
        except ValueError as e:
            # check if the message matches the regex '^Batch does not contain any data'
            if re.match('^Batch does not contain any data', str(e)):
                logger.info(f'ran out of data and the dataset_generator cursor is {self.train_tokens_generator.get_cursor().to_dict()}')
                # calling this because the stupid exception thrown above prevented it from being invoked
                # it needs to be called to ensure the last state of the model is saved
                self.checkpoint_manager.on_train_end(self.trainer.args, self.trainer.state, self.trainer.control)

        logger.debug(f'generator cursor after training is segment {self.dataset_segment_number()},  {self.train_tokens_generator.get_cursor().to_dict()}')
        logger.info(f'Trained for {self.checkpoint_manager.steps_seen} steps')
        logger.debug(f'post training trainer state {self.trainer.state.__repr__()}')

    def dataset_segment_number(self):
        if 'checkpoint_manager' in self.__dict__ and self.checkpoint_manager:
            return self.checkpoint_manager.current_dataset_segment_number
        elif self.starting_checkpoint_info and self.starting_checkpoint_info.segment_number:
            return self.starting_checkpoint_info.segment_number
        else:
            return 1

    def _get_continuation_state(self) -> int:
        self.starting_step = self.checkpoint_registry.latest_step()
        logger.debug(f"starting_step is initially {self.starting_step}")
        if self.starting_step:
           self.starting_checkpoint_info = self.checkpoint_registry.get_checkpoint_for_step(self.starting_step)
           if not self.starting_checkpoint_info:
               raise RuntimeError(f"starting_step is defined as {self.starting_step} but could not retrieve checkpoint_info for it")
           # check if self.previous_checkpoint_path exists and raise an error if it does not
           if not self.starting_checkpoint_info.exists():
               raise RuntimeError(f"starting checkpoint path is defined as {self.starting_checkpoint_info.path()} but it does not exist")
           logger.info(f"resuming training from checkpoint {self.starting_checkpoint_info.__dict__.__repr__()}")
        else:
           self.previous_checkpoint_path = None
           self.starting_step = 0
        return self.starting_step

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.starting_checkpoint_info:
            checkpoint_path=self.starting_checkpoint_info.path()
            logger.info(f"Loading previous checkpoint model at {checkpoint_path}")
            self.model = self.load_checkpoint_model(checkpoint_path, self.base_model_id)
        else:
            logger.info(f"Starting with untrained LoRA on {self.base_model_id}")
            self.model = self.load_starting_model(self.base_model_id)
        assert self.model
        logger.info(f"LoRA model stats: {self.trainable_parameters_description(self.model)}")
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        return self.model

    def _prepare_datasets(self) -> tuple[IterableDataset, Union[Dataset,None]]:
        assert self.tokenizer

        train_dataset_id = self._current_dataset_id()
        logger.debug(f"Preparing train dataset from {train_dataset_id}")
        self.text_dataset = S3TextDataset(self.dataset_bucket_name, dataset_id=train_dataset_id).to_full_dataset()
        logger.info(f'text_dataset contains {len(self.text_dataset)} documents')
        # TODO:  maybe make these constructor params?
        min_stride=64
        max_waste=64
        self.train_tokens_generator = TextDS2TokensGenerator(self.text_dataset, self.tokenizer, chunk_len = self.max_seq_length, min_stride = min_stride, max_waste = max_waste)
        features = self.train_tokens_generator.features()
        if self.test_dataset_id:
            test_text_dataset = S3TextDataset(self.dataset_bucket_name, dataset_id=self.test_dataset_id).to_full_dataset()
            test_tokens_generator = TextDS2TokensGenerator(test_text_dataset, self.tokenizer, chunk_len = self.max_seq_length, min_stride = min_stride, max_waste = max_waste)
            test_inputs_ds: Dataset = cast(Dataset, Dataset.from_generator(test_tokens_generator))
            self.test_inputs = test_inputs_ds
        else:
            self.test_inputs = None
        self.train_inputs = IterableDataset.from_generator(self.train_tokens_generator,features=features)
        return (self.train_inputs, self.test_inputs)

    def estimated_max_steps(self):
        raise RuntimeError("Have not yet implemented estimated_max_steps(), so specify max_steps explicitly at construction.")

    def _prepare_trainer(self) -> Trainer:
        # pre-requisites
        assert self.model
        assert self.train_inputs
        assert self.train_tokens_generator

        data_collator = torchify_collator
        num_train_epochs = 1
        gradient_accumulation_steps = 4
        #max_steps = estimated_number_of_training_examples // gradient_accumulation_steps
        max_steps = self.explicit_max_steps if self.explicit_max_steps else self.estimated_max_steps()

        warmup_steps = 8 if self.dataset_segment_number() in [1,None] else 0
        starting_learning_rate = 2e-4

        training_args = TrainingArguments(
                ignore_data_skip = True, # when resuming, do not try to skip past previously trained examples.  The train_tokens_generator.set_cursor() call takes care of that.
                remove_unused_columns = True,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                learning_rate=starting_learning_rate,
                fp16=True,
                logging_steps=1,
                output_dir=self.output_dir,
                optim="paged_adamw_8bit",
                num_train_epochs = num_train_epochs,
                save_steps=self.save_steps,      # Checkpoint regularly every this many steps
                max_steps=max_steps,
                evaluation_strategy="no",  # Do not bother with evaluation
                include_num_input_tokens_seen = True,
                #eval_steps = eval_steps,  # Uncomment and set this if you choose evaluation_strategy="steps"
            )


        self.trainer = Trainer(
                        model=self.model,
                        train_dataset=self.train_inputs, # type: ignore (IterableDataset is apparently not envisioned here)
                        eval_dataset=self.test_inputs, # type: ignore (IterableDataset is apparently not envisioned here)
                        args=training_args,
                        data_collator=data_collator
        )
        # We want to be able to restart the trainer at will so instead of putting max_steps
        # in the trainer's arguments, we put it in a callback that breaks early on an infinite steps
        # overall training run

        # it will add itself as a callback to the trainer in its constructor
        self.checkpoint_manager = CheckpointManager(self.trainer,
                                                  self.checkpoint_registry,
                                                  self.train_tokens_generator,
                                                  starting_checkpoint_info=self.starting_checkpoint_info,
                                                  stop_after_steps = self.steps_per_round,
                                                  logger=logger)
        return self.trainer


    def _cuda_check(self) -> None:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            logger.debug(f'__CUDNN VERSION: {torch.backends.cudnn.version()}')
            logger.debug(f'__Number CUDA Devices: {torch.cuda.device_count()}')
            logger.debug(f'__CUDA Device Name: {torch.cuda.get_device_name(0)}')
            logger.debug(f'__CUDA Device Total Memory [GB]: {torch.cuda.get_device_properties(0).total_memory/1e9}')

    def _current_dataset_id(self) -> str:
        if self.dataset_id:
            return self.dataset_id
        else:
            assert self.dataset_series
            dataset_id = self.dataset_series.replace("{segment_number}",str(self.dataset_segment_number()))
            return dataset_id

    @staticmethod
    def trainable_parameters_description(model) -> str:
        """
        Reports the number of trainable parameters in a model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                # print(_)
                trainable_params += param.numel()
        return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"




class QLoRAContinuingTrainer(BaseContinuingTrainer):

    def _load_base_model(self, base_model_id: str) -> PreTrainedModel:
        logger.debug(f"loading base model {self.base_model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        base_model.gradient_checkpointing_enable()         # significantly the reduce memory footprint of the activations cache during training
        base_model = prepare_model_for_kbit_training(base_model)
        return base_model

    def announce_model_config(self, action:str, model: PeftModel) -> None:
        active_adapter = model.active_adapter
        # sometimes active_adaapter is a string, sometimes a function, ???
        if not isinstance(active_adapter, str):
            active_adapter = active_adapter()
        logger.info(f"{action} Q-LoRA model. active adapter: {active_adapter} ")

    def load_checkpoint_model(self, checkpoint_path: str, base_model_id: str) -> PeftModel:
        if not self.__dict__.get('base_model',None):
            self.base_model = self._load_base_model(base_model_id)
        assert self.base_model
        model = PeftModel.from_pretrained(model=self.base_model,
                                          model_id=checkpoint_path,
                                          adapter_name="default",
                                          is_trainable=True)
        self.announce_model_config('Loaded', model)
        return model

    def build_adapter_config(self) -> LoraConfig:
        mistral_7b_target_modules =   [
                        'q_proj',
                        'k_proj',
                        'v_proj',
                        'o_proj',
                        'gate_proj',
                        'up_proj',
                        'down_proj',
                        'lm_head', # unsloth does not use this in their benchmark, why?
        ]

        adapter_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=mistral_7b_target_modules,
            lora_dropout=0,  # Initially 0.05, changed to 0 because of unsloth
            bias="none",
            task_type="CAUSAL_LM",
        )
        return adapter_config

    def load_starting_model(self, base_model_id: str) -> PeftModel:
        if not self.__dict__.get('base_model',None):
            self.base_model = self._load_base_model(base_model_id)
        assert self.base_model
        model = cast(PeftModel,get_peft_model(self.base_model,self.build_adapter_config()))
        self.announce_model_config('Constructed', model)
        return model

