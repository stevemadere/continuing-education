#!/usr/bin/env python

import os
import warnings
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Union
from tokengenerators import TextDS2TokensGenerator, DSGeneratorCursor
from checkpoint_registry import CheckpointRegistry, CheckpointInfo
import logging

default_logger = logging.getLogger(__name__)

DATASET_CURSOR_FILENAME='dataset_cursor.json'


class CheckpointManager(TrainerCallback):
      """
      Adds a callback to a huggingface transformers.Trainer to watch for checkpoints being
      made and record the cursor of the generator behind the training dataset so that
      training resumption is feasible on huge iterable datasets.

      Also ensures that the generator cursor is properly initialized when training is resumed.

      usage:
          checkpoint_callback = CheckpointManager(trainer,checkpoint_registry, dataset_generator, starting_checkpoint_info, stop_after_steps = 10000, save_on_stop = True, my_logger)
          trainer.train()
      """
      trainer: Trainer
      dataset_generator: Union[TextDS2TokensGenerator,None]
      stop_after_steps: Union[int, None]
      save_on_stop: bool
      steps_seen: int
      checkpoint_registry: Union[CheckpointRegistry,None]
      starting_checkpoint_info: Union[CheckpointInfo,None]
      current_dataset_segment_number: int
      logger: logging.Logger
      trainer_state_at_save: Union[dict,None]

      def __init__(self,
                   trainer: Trainer,
                   checkpoint_registry : Union[CheckpointRegistry,None] = None,
                   dataset_generator: Union[TextDS2TokensGenerator,None] = None,
                   starting_checkpoint_info: Union[CheckpointInfo,None] = None,
                   stop_after_steps: Union[int,None] = None,
                   save_on_stop: bool = True,
                   logger: logging.Logger = default_logger
                   ):
          super()
          self.trainer = trainer
          self.dataset_generator = dataset_generator
          self.checkpoint_registry = checkpoint_registry
          self.starting_checkpoint_info = starting_checkpoint_info
          self.stop_after_steps = stop_after_steps
          self.save_on_stop = save_on_stop
          self.logger = logger
          self.steps_seen = 0
          self._last_save_was_at_step = None
          self.trainer_state_at_save = None
          self.current_dataset_segment_number = self.starting_checkpoint_info.segment_number if (self.starting_checkpoint_info and self.starting_checkpoint_info.segment_number) else 1
          self.trainer.add_callback(self)

      def on_train_begin(self,
                         args: TrainingArguments,
                         state: TrainerState, control: TrainerControl, **kwargs ):
          if args or state or control or kwargs: # silence unused var warnings
              pass

          checkpoint = self.starting_checkpoint_info.path() if self.starting_checkpoint_info else None
          self.logger.debug(f"CheckpointManager on_train_begin with checkpoint {checkpoint}")
          if checkpoint and self.dataset_generator:
              checkpoint_cursor  = CheckpointManager.get_checkpoint_dataset_cursor(checkpoint)
              if checkpoint_cursor:
                  self.logger.debug(f"restoring cursor for checkpoint {checkpoint}")
                  self.dataset_generator.set_cursor(checkpoint_cursor)
              else:
                  self.logger.warn(f"checkpoint {checkpoint} has no dataset cursor")
          if self.dataset_generator:
              self.logger.info(f"initial_cursor is segment {self.current_dataset_segment_number}, {self.dataset_generator.get_cursor().to_dict()}")
          else:
              self.logger.debug(f"CheckpointManager is unaware of a dataset_generator")

      def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs ):
          if args or state or kwargs: # suppress unused param warnings
              pass
          self.logger.debug(f"about to do local step {self.steps_seen+1} and dataset cursor is segment {self.current_dataset_segment_number}, {self.dataset_generator.get_cursor().to_dict().__repr__() if self.dataset_generator else 'not defined'}")
          # Why the heck do I have to do this?  It's insane.  This should be the default behavior of Trainer anyway.
          # Instead, if it runs out of training data, it raises an error
          if self.dataset_generator and self.dataset_generator.exhausted:
              self.logger.debug("dataset_generator is exhausted. Stopping training.")
              control.should_training_stop = True

          if control.should_training_stop and self.save_on_stop:
              self.logger.debug("Trying to trigger a save_on_stop")
              control.should_save = True

      def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs ):
          if args or state or kwargs: # suppress unused param warnings
              pass
          self.steps_seen += 1
          self.logger.debug(f"just did local step {self.steps_seen} and dataset cursor is segment {self.current_dataset_segment_number}, {self.dataset_generator.get_cursor().to_dict().__repr__() if self.dataset_generator else 'not defined'}")
          if self.stop_after_steps and self.steps_seen >= self.stop_after_steps:
              self.logger.debug(f"Terminating training loop after {self.stop_after_steps} steps")
              control.should_training_stop = True

          # Why the heck do I have to do this?  It's insane.  This should be the default behavior of Trainer anyway.
          # Instead, if it runs out of training data, it raises an error
          if self.dataset_generator and self.dataset_generator.exhausted:
              control.should_training_stop = True

          if control.should_training_stop and self.save_on_stop:
              self.logger.debug("Trying to trigger a save_on_stop")
              control.should_save = True

      def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs ):
          if args and state and control and kwargs:
              pass
          self.logger.debug(f"detected a checkpoint save at step {state.global_step}")
          checkpoint_dir = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
          checkpoint_path = os.path.join(args.output_dir,checkpoint_dir)
          self._last_save_was_at_step = state.global_step
          # if something exists at checkpoint_path...
          if os.path.exists(checkpoint_path):
              global_step = state.global_step
              # make a deep copy of state for debugging purposes
              self.trainer_state_at_save = state.__dict__.copy()
              if self.checkpoint_registry:
                  if self.dataset_generator and self.dataset_generator.exhausted:
                      self.current_dataset_segment_number += 1
                      self.logger.debug(f"Current dataset exhausted, advancing segment number to {self.current_dataset_segment_number}")
                      self.dataset_generator.set_cursor(DSGeneratorCursor(0,0))
                  self.checkpoint_registry.add_checkpoint(global_step=global_step,
                                                          segment_number = self.current_dataset_segment_number,
                                                          checkpoint = checkpoint_dir)
              self._save_dataset_cursor(checkpoint_path)
          else:
              raise RuntimeError(f"expected checkpoint_path {checkpoint_path} does not exist")

      def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs ):
          if self.save_on_stop and ((not self._last_save_was_at_step) or self._last_save_was_at_step < state.global_step):
              self.trainer._save_checkpoint(self.trainer.model,trial=None,metrics=None)
              self.on_save(args,state,control,kwargs=kwargs)

      @staticmethod
      def checkpoint_dataset_cursor_filepath(checkpoint_path: str):
          return os.path.join(checkpoint_path,DATASET_CURSOR_FILENAME)

      @staticmethod
      def get_checkpoint_dataset_cursor(checkpoint_path: str) -> Union[DSGeneratorCursor,None]:
          cursor_file_path = CheckpointManager.checkpoint_dataset_cursor_filepath(checkpoint_path)
          if os.path.exists(cursor_file_path):
              return DSGeneratorCursor.from_file_path(cursor_file_path)
          else:
              return None
 
      def _save_dataset_cursor(self,checkpoint_path: str) -> bool:
          if self.dataset_generator:
              try:
                  cursor = self.dataset_generator.get_cursor()
              except AttributeError:
                  warnings.warn("tried to save dataset cursor in checkpoint but generator does not support cursors")
                  return False
              cursor_file_path = self.__class__.checkpoint_dataset_cursor_filepath(checkpoint_path)
              cursor.save_to_file_path(cursor_file_path)
              return True
          else:
              return False


