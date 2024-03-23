# conftest.py
import os
import re
import pytest
import boto3
import dotenv


def pytest_configure():
    dotenv.load_dotenv()

s3_uri_env_var = 'TEST_CHECKPOINTS_S3_URI'

my_dirname:str = os.path.dirname(__file__)

checkpoints_dir_content:dict[str,str] = {
    'checkpoint-1/c1f1': 'content of checkpoint-1/c1f1',
    'checkpoint-1/c1f2': 'content of checkpoint-1/c1f2',
    'checkpoint-1/c1f3': 'content of checkpoint-1/c1f3',
    'checkpoint-1/c1d1/c1d1f1': 'content of checkpoint-1/c1d1/c1d1f1',
    'checkpoint-1/c1d1/c1d1f2': 'content of checkpoint-1/c1d1/c1d1f2',
    'checkpoint-1/c1d1/c1d2f1': 'content of checkpoint-1/c1d1/c1d2f1',
    'checkpoint-1/c1d1/c1d2f2': 'content of checkpoint-1/c1d1/c1d2f2',
    'checkpoint-2/c2f1': 'content of checkpoint-2/c2f1',
    'checkpoint-2/c2f2': 'content of checkpoint-2/c2f2',
    'checkpoint-2/c2f3': 'content of checkpoint-2/c2f3',
    'checkpoint-2/c2d1/c1d1f1': 'content of checkpoint-2/c2d1/c1d1f1',
    'checkpoint-2/c2d1/c1d1f2': 'content of checkpoint-2/c2d1/c1d1f2',
    'checkpoint-2/c2d1/c1d2f1': 'content of checkpoint-2/c2d1/c1d2f1',
    'checkpoint-2/c2d1/c1d2f2': 'content of checkpoint-2/c2d1/c1d2f2',
    'checkpoint_registry.json': '{"1": {"checkpoint_name": "checkpoint-1", "global_step": 1, "segment_number": 1}, "2": {"checkpoint_name": "checkpoint-2", "global_step": 10000, "segment_number": 2} }'
}

@pytest.fixture(scope="module")
def populated_s3_checkpoints():
        s3_uri = os.environ.get(s3_uri_env_var,None)
        if not s3_uri:
            reason =f'There is no environment variable {s3_uri_env_var} set in .env file.'
            print(f'{reason}  Skipping dependent tests.')
            yield { 'uri' : None, 'reason' : reason  }
        else:
            match = re.match(r'^s3://([^/]+)/(.+)', s3_uri)
            if match:
                bucket_name, prefix = match.groups()
            else:
                raise ValueError(f'The environment variable {s3_uri_env_var} is not a valid S3 URI but "{s3_uri}"')
            s3_resource = boto3.resource('s3')
            bucket = s3_resource.Bucket(bucket_name)
            for key, value in checkpoints_dir_content.items():
                s3_key = f'{prefix}/{key}'
                bucket.put_object( Key=s3_key, Body=value)
            yield {'uri': s3_uri, 'contents': checkpoints_dir_content }

