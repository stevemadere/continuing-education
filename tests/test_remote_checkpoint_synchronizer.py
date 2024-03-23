import pytest
import os
import sys

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)

from continuing_education import RemoteCheckpointSynchronizer
from continuing_education.checkpoint_registry import S3RemoteCheckpointSynchronizer

bucket_name='test_bucket'
prefix='/checkpoints/test'
s3_uri = f's3://{bucket_name}/{prefix}'
local_output_dir = 'test_outputs'

unsupported_uri = f'mongo://whatever/monog/uris/have/in/the/path'

def test_handler_registry_populated() -> None:
    assert len(RemoteCheckpointSynchronizer._handler_registry) > 0
    assert S3RemoteCheckpointSynchronizer in RemoteCheckpointSynchronizer._handler_registry

def test_can_handle_uri() -> None:
    assert S3RemoteCheckpointSynchronizer.can_handle_uri(s3_uri)

def test_from_uri_with_s3_uri() -> None:
    synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir)
    assert synchronizer.__class__ == S3RemoteCheckpointSynchronizer


def test_from_uri_with_unsupported_uri() -> None:
    with pytest.raises(RuntimeError) as err_info:
        synchronizer = RemoteCheckpointSynchronizer.from_uri(unsupported_uri, local_output_dir)
        assert not synchronizer
    assert 'No RemoteCheckpointSynchronizer exists that can handle the uri' in str(err_info.value)




