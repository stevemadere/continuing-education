import pytest
import tempfile
import os
import re
import sys
from logging import warning
import inspect

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)

from continuing_education import RemoteCheckpointSynchronizer
from continuing_education.checkpoint_registry import S3RemoteCheckpointSynchronizer

bucket_name='test_bucket'
prefix='/checkpoints/test'
s3_uri = f's3://{bucket_name}/{prefix}'
local_output_dir = 'test_outputs'

unsupported_uri = f'mongo://whatever/monog/uris/have/in/the/path'

class PersistentTemporaryDirectory:
    """
        A replacement for TemporaryDirectory that skips cleaning up to aid in debugging.
        This is the same behavior one gets with the delete=True parameter in Python 3.12+
        but it works in prior versions of Python as well
    """
    _dir:str

    def __init__(self):
        # Create the temporary directory upon instantiation
        self._dir = tempfile.mkdtemp()
        warning(f'creating temporary directory {self._dir}')

    @property
    def name(self):
        # Provide access to the name (path) of the temporary directory
        return self._dir

    # Implement the context manager protocol to use with `with` statement
    def __enter__(self):
        # Return self to allow access to the class instance and its properties/methods
        return self._dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb or True: # silence PyRight unused param warnings
            # No cleanup is done here, making the directory persistent after exiting the context
            pass

def current_test_name() -> str:
    cframe =  inspect.currentframe()
    if cframe and cframe.f_back and cframe.f_back.f_code:
        return cframe.f_back.f_code.co_name
    else:
        return 'no freaking idea what function I am in right now'

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


def test_construct_with_real_uri(populated_s3_checkpoints):
    s3_uri = populated_s3_checkpoints['uri']
    if s3_uri:
        contents = populated_s3_checkpoints['contents']
        assert contents
        with tempfile.TemporaryDirectory() as temp_dir:
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True


def test_download_one_sync_with_file(populated_s3_checkpoints):
    s3_uri = populated_s3_checkpoints['uri']
    if s3_uri:
        contents = populated_s3_checkpoints['contents']
        rel_path = 'checkpoint_registry.json' if 'checkpoint_registry.json' in contents else contents.keys()[0]
        remote_file_content = contents[rel_path]
        with tempfile.TemporaryDirectory() as temp_dir:
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
            synchronizer.download_one_sync(rel_path)
            full_path = os.path.join(temp_dir,rel_path)
            assert os.path.exists(full_path), f'{full_path} does not exist'
            local_file_content = open(full_path,'r').read()
            assert local_file_content == remote_file_content
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True

#def confirm_directory_download_correct(temp_dir: str, rel_path: str, contents dict[str,str]) -> bool:

def test_download_one_sync_with_directory(populated_s3_checkpoints):
    s3_uri = populated_s3_checkpoints['uri']
    if s3_uri:
        contents = populated_s3_checkpoints['contents']

        first_matching_checkpoint_dash_number = None
        pattern = r'^(checkpoint-\d+)/.+$'
        for k in contents.keys():
            match = re.match(pattern,k)
            if match:
                first_matching_checkpoint_dash_number = match.group(1)
                break
        assert first_matching_checkpoint_dash_number, f'no checkpoint directory found in {s3_uri}'
        rel_path = first_matching_checkpoint_dash_number
        # print(f'trying download_one_sync on rel_path "{rel_path}"')
        with tempfile.TemporaryDirectory() as temp_dir:
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
            synchronizer.download_one_sync(rel_path)
            full_path = os.path.join(temp_dir,rel_path)
            assert os.path.exists(full_path), f'{full_path} does not exist'
            assert os.path.isdir(full_path), f'{full_path} is not a directory'
            keys_under_rel_path = list(filter(lambda x: x.startswith(os.path.join(rel_path,'')), contents.keys()))
            # print(f'Confirming content of objects under {rel_path}: {keys_under_rel_path}')
            for key in keys_under_rel_path:
                corresponding_local_path = os.path.join(temp_dir,key)
                # print(f'comparing {key} with {corresponding_local_path}')
                assert os.path.exists(corresponding_local_path), f'{corresponding_local_path} does not exist'
                local_file_content = open(corresponding_local_path,'r').read()
                assert local_file_content == contents[key]
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True



