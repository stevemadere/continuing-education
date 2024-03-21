#!/usr/bin/env python

from continuing_education import QLoRAContinuingTrainer, logger

import os

if ('AWS_PROFILE' not in os.environ) and not ('AWS_SECRET_ACCESS_KEY' in os.environ and 'AWS_ACCESS_KEY_ID' in os.environ):
  raise EnvironmentError("AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY required in the environment.")

dataset_bucket_name=os.environ.get('DATASET_BUCKET')
assert dataset_bucket_name

output_dir = os.environ.get('OUTPUT_DIR')
assert output_dir

if 'HF_MODEL_NAME' in os.environ and 'DATA_DIR' in os.environ:
    base_model_id = f"{os.environ['DATA_DIR']}/{os.environ['HF_MODEL_NAME']}"
elif 'HF_MODEL_NAME' in os.environ and 'HF_CONTRIBUTOR' in os.environ:
    base_model_id = f"{os.environ['HF_CONTRIBUTOR']}/{os.environ['HF_MODEL_NAME']}"
else:
    base_model_id = 'Mistralai/Mistral-7Bv0.1'

logger.info(f'base model is {base_model_id}')

# If you don't provide this environment var, the training process will try to train on absolutely
# every S3 object in your bucket as a text document (in lexicographical order of their S3 keys).
# If you do provide it, there must be a set of objects in your bucket whose keys match the provided
# value but with numbers from 1 to N substituted for the substring {segment_number} in your provided value.
# e.g.  if you provide "/datasets/my_big_dataset_{segment_number}.json.gz and you have at least an object whose
# key is "/datasets/my_big_dataset_1.json.gz" in your bucket
# The content of the corresponding objects must be a valid json array of keys of text objects in that
# same bucket.  e.g. "[ '/text_data/document1.txt', '/text_data/another_document.txt', '/text_data/random_object_name_but_definitely_for_a_text_document', '/surprise_location/more_random_key_shennanegins']
dataset_series= os.environ.get('DATASET_SERIES')


# steps_per_round limits the number of steps taken in continuing_trainer.train() below.
# Make this number small at first while you shake out your training process.
# This allows you to train for a bit, see that all is well, make changes if necessary, continue, etc.
# Once you feel it's solid, go ahead and crank it up to a very large number
steps_per_round = int(os.environ.get('STEPS_PER_ROUND') or 10000)

# For my dataset with about 100k documents, each yielding about 16 chunks of tokens, this is a reasonable estimate
# of the total number of steps that it takes to get through it all.
# You can determine approx_steps_per_segment empirically for your own s3dataset using 
# TextDS2TokensGenerator.estimate_available_chunks()
approx_steps_per_segment = 25000
num_segments = 10
max_steps = int(os.environ.get('MAX_STEPS') or approx_steps_per_segment*num_segments)

continuing_trainer = QLoRAContinuingTrainer(
                      base_model_id = base_model_id,
                      dataset_bucket_name = dataset_bucket_name,
                      output_dir = output_dir,
                      dataset_series = dataset_series,
                      steps_per_round = steps_per_round,
                      max_steps = max_steps
                      )

continuing_trainer.train()

