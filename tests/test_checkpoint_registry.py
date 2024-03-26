import tempfile
import os
import sys

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)
from continuing_education import CheckpointRegistry, RemoteCheckpointSynchronizer

from .utils.misc import  copy_dict_to_s3
from .utils.directory_dict import DirectoryDict
from .utils.temp_s3_object import TempS3Object
from .utils.persistent_temporary_directory import PersistentTemporaryDirectory
if PersistentTemporaryDirectory: # Supress pyright unused symbol warnings
    pass

remote_checkpoints_content:dict[str,str] = {
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

def test_construction_with_remote_synchronizer(writable_s3_uri):
    bucket, prefix = TempS3Object.parse_s3_uri(writable_s3_uri)
    if prefix: # suppress pyright warnings about unused var
        pass
    with TempS3Object(writable_s3_uri).TemporaryDirectory(delete=True) as temp_s3_dir:
        temp_s3_uri = f's3://{bucket}/{temp_s3_dir}'
        copy_dict_to_s3(remote_checkpoints_content, temp_s3_uri)
        with tempfile.TemporaryDirectory() as temp_local_dir:
        # with PersistentTemporaryDirectory() as temp_local_dir:
            syncronizer_s3_uri = f's3://{bucket}/{temp_s3_dir}'

            synchronizer = RemoteCheckpointSynchronizer.from_uri(syncronizer_s3_uri, local_output_dir = temp_local_dir)
            registry = CheckpointRegistry(output_dir = temp_local_dir, remote_synchronizer = synchronizer)
            assert registry._registry[1000].checkpoint_name == 'checkpoint-1000'
            assert registry._registry[2000].checkpoint_name == 'checkpoint-2000'
            assert registry.latest_step() == 2000
            checkpoint2000 = registry.get_checkpoint_for_step(2000)
            assert checkpoint2000
            checkpoint_name = checkpoint2000.checkpoint_name 
            checkpoint_full_path = os.path.join(temp_local_dir, checkpoint_name) 
            assert os.path.exists(checkpoint_full_path)
            assert os.path.isdir(checkpoint_full_path)
            dir_dict = DirectoryDict(temp_local_dir)
            for rel_path in remote_checkpoints_content.keys():
                full_path = os.path.join(temp_local_dir,rel_path)
                # if it begins with checkpoint_name
                if rel_path.startswith(checkpoint_name+'/') or rel_path == 'checkpoint_registry.json':
                    assert os.path.exists(full_path)
                    expected_content = remote_checkpoints_content[rel_path]
                    assert dir_dict[rel_path] == expected_content
                else:
                    assert not os.path.exists(full_path)

def test_construction_without_remote_synchronizer():
    with tempfile.TemporaryDirectory() as temp_local_dir:
        registry = CheckpointRegistry(output_dir = temp_local_dir)
        assert registry
        assert registry.latest_step() == None
