import pytest
import tempfile
import os
import re
import sys
from logging import warning

from .utils.persistent_temporary_directory import PersistentTemporaryDirectory
if PersistentTemporaryDirectory: # Supress pyright unused symbol warnings
    pass

from .utils.misc import current_test_name, copy_dict_to_fs


path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)

import continuing_education
from continuing_education import  RemoteCheckpointSynchronizer
from continuing_education.checkpoint_registry import S3RemoteCheckpointSynchronizer
continuing_education.checkpoint_registry.verbose = False # Make this True for debugging

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


def test_download_one_sync_with_file_that_exists(populated_s3_checkpoints):
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

def test_download_one_sync_with_file_nonexistent(populated_s3_checkpoints):
    s3_uri = populated_s3_checkpoints['uri']
    if s3_uri:
        contents = populated_s3_checkpoints['contents']
        rel_path = 'nonexistent_file.bogus'
        print(f'checking for nonexistent file "{rel_path}"')
        assert not rel_path in contents
        #with PersistentTemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f'the tempdir is {temp_dir}/')
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
            synchronizer.download_one_sync(rel_path)
            full_path = os.path.join(temp_dir,rel_path)
            assert not os.path.exists(full_path), f'{full_path} should not exist because it was no on the remote'
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True

def confirm_directory_download_correct(temp_dir: str, rel_path: str, contents: dict[str,str]) -> bool:
    if rel_path == '':
        keys_under_rel_path = list(contents.keys())
    else:
        keys_under_rel_path = list(filter(lambda x: x.startswith(os.path.join(rel_path,'')), contents.keys()))

    # print(f'Confirming content of objects under {rel_path}: {keys_under_rel_path}')
    for key in keys_under_rel_path:
        corresponding_local_path = os.path.join(temp_dir,key)
        # print(f'comparing {key} with {corresponding_local_path}')
        assert os.path.exists(corresponding_local_path), f'{corresponding_local_path} does not exist'
        local_file_content = open(corresponding_local_path,'r').read()
        assert local_file_content == contents[key]
    return True

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
#        with PersistentTemporaryDirectory() as temp_dir:
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
            synchronizer.download_one_sync(rel_path)
            full_path = os.path.join(temp_dir,rel_path)
            assert os.path.exists(full_path), f'{full_path} does not exist'
            assert os.path.isdir(full_path), f'{full_path} is not a directory'
            assert confirm_directory_download_correct(temp_dir, rel_path, contents), f'The content of local dir "{full_path}" does not match {s3_uri}/{rel_path}'
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True


def test_download_all_sync(populated_s3_checkpoints):
    s3_uri = populated_s3_checkpoints['uri']
    if s3_uri:
        contents = populated_s3_checkpoints['contents']

        # print(f'trying download_one_sync on rel_path "{rel_path}"')
        with tempfile.TemporaryDirectory() as temp_dir:
            synchronizer = RemoteCheckpointSynchronizer.from_uri(s3_uri, local_output_dir = temp_dir)
            assert synchronizer
            synchronizer.download_all_sync()
            assert confirm_directory_download_correct(temp_dir,'',contents), f'The content of local dir "{temp_dir}" does not match {s3_uri}'
    else:
        reason = populated_s3_checkpoints['reason']
        warning(f'Skipping {current_test_name()}() because {reason}')
        assert True


from .utils.temp_s3_object import TempS3Object
from .utils.s3_dict import S3Dict


checkpoints_content:dict[str,str] = {
    'checkpoint-1000/c1f1': 'local content of checkpoint-1000/c1f1',
    'checkpoint-1000/c1f2': 'local content of checkpoint-1000/c1f2',
    'checkpoint-1000/c1f3': 'local content of checkpoint-1000/c1f3',
    'checkpoint-1000/c1d1/c1d1f1': 'local content of checkpoint-1000/c1d1/c1d1f1',
    'checkpoint-1000/c1d1/c1d1f2': 'local content of checkpoint-1000/c1d1/c1d1f2',
    'checkpoint-1000/c1d1/c1d2f1': 'local content of checkpoint-1000/c1d1/c1d2f1',
    'checkpoint-1000/c1d1/c1d2f2': 'local content of checkpoint-1000/c1d1/c1d2f2',
    'checkpoint-2000/c2f1': 'local content of checkpoint-2000/c2f1',
    'checkpoint-2000/c2f2': 'content of checkpoint-2000/c2f2',
    'checkpoint-2000/c2f3': 'content of checkpoint-2000/c2f3',
    'checkpoint-2000/c2d1/c1d1f1': 'content of checkpoint-2000/c2d1/c1d1f1',
    'checkpoint-2000/c2d1/c1d1f2': 'content of checkpoint-2000/c2d1/c1d1f2',
    'checkpoint-2000/c2d1/c1d2f1': 'content of checkpoint-2000/c2d1/c1d2f1',
    'checkpoint-2000/c2d1/c1d2f2': 'content of checkpoint-2000/c2d1/c1d2f2',
    'checkpoint_registry.json': '{"1000": {"checkpoint_name": "checkpoint-1000", "global_step": 1000, "segment_number": 1}, "2000": {"checkpoint_name": "checkpoint-2000", "global_step": 2000, "segment_number": 2} }'
}

def test_upload_all_aync(writable_s3_uri):
    if writable_s3_uri:
        bucket, prefix = TempS3Object.parse_s3_uri(writable_s3_uri)
        if prefix: # suppress pyright warnings about unused var
            pass
        with tempfile.TemporaryDirectory() as temp_local_dir:
            copy_dict_to_fs(checkpoints_content, temp_local_dir)
            with TempS3Object(writable_s3_uri).TemporaryDirectory(delete=True) as temp_s3_dir:
                dest_s3_uri = f's3://{bucket}/{temp_s3_dir}'
                synchronizer = RemoteCheckpointSynchronizer.from_uri(dest_s3_uri, local_output_dir = temp_local_dir)
                task = synchronizer.upload_all_async()
                task.wait_for_it()
                assert not task.error
                s3_dict = S3Dict(dest_s3_uri)
                for rel_path, file_content in checkpoints_content.items():
                    assert s3_dict[rel_path] == file_content

    else:
        reason = 'No writable S3 URI defined'
        warning(f'Skipping {current_test_name()}() because {reason}')

