# conftest.py
import os
import pytest
import json
from moto import mock_s3
import datasets
import boto3
import dotenv


def pytest_configure():
    dotenv.load_dotenv()

my_dirname:str = os.path.dirname(__file__)

bucket_content:dict[str,str] = {
    'checkpoints/test/checkpoint-1/c1f1': 'content of checkpoint-1/c1f1',
    'checkpoints/test/checkpoint-1/c1f2': 'content of checkpoint-1/c1f2',
    'checkpoints/test/checkpoint-1/c1f3': 'content of checkpoint-1/c1f3',
    'checkpoints/test/checkpoint-1/c1d1/c1d1f1': 'content of checkpoint-1/c1d1/c1d1f1',
    'checkpoints/test/checkpoint-1/c1d1/c1d1f2': 'content of checkpoint-1/c1d1/c1d1f2',
    'checkpoints/test/checkpoint-1/c1d1/c1d2f1': 'content of checkpoint-1/c1d1/c1d2f1',
    'checkpoints/test/checkpoint-1/c1d1/c1d2f2': 'content of checkpoint-1/c1d1/c1d2f2',
    'checkpoints/test/checkpoint-2/c2f1': 'content of checkpoint-2/c2f1',
    'checkpoints/test/checkpoint-2/c2f2': 'content of checkpoint-2/c2f2',
    'checkpoints/test/checkpoint-2/c2f3': 'content of checkpoint-2/c2f3',
    'checkpoints/test/checkpoint-2/c2d1/c1d1f1': 'content of checkpoint-2/c2d1/c1d1f1',
    'checkpoints/test/checkpoint-2/c2d1/c1d1f2': 'content of checkpoint-2/c2d1/c1d1f2',
    'checkpoints/test/checkpoint-2/c2d1/c1d2f1': 'content of checkpoint-2/c2d1/c1d2f1',
    'checkpoints/test/checkpoint-2/c2d1/c1d2f2': 'content of checkpoint-2/c2d1/c1d2f2',
    'checkpoints/test/checkpoint_registry.json': '{"1": {"checkpoint_name": "checkpoint-1", "global_step": 1, "segment_number": 1}, "2": {"checkpoint_name": "checkpoint-2", "global_step": 10000, "segment_number": 2} }'
}

@pytest.fixture(scope="module")
def mock_s3_bucket():
    with mock_s3():
        s3 = boto3.client('s3')
        bucket_name = 'my-mock-bucket'
        s3.create_bucket(Bucket=bucket_name)
        for key, value in bucket_content.items():
            s3.put_object(Bucket=bucket_name, Key=key, Body=value)
        yield {'bucket_name': bucket_name, 'contents': bucket_content, 'prefix': 'checkpoints/test' }


