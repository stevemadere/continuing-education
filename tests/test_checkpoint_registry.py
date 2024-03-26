import tempfile
import os
import sys

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)
import continuing_education
from continuing_education import CheckpointRegistry, RemoteCheckpointSynchronizer
continuing_education.checkpoint_registry.verbose = False # Make this True for debugging

from .utils.misc import  copy_dict_to_s3, copy_dict_to_fs
from .utils.directory_dict import DirectoryDict
from .utils.s3_dict import S3Dict
from .utils.temp_s3_object import TempS3Object
from .utils.persistent_temporary_directory import PersistentTemporaryDirectory
if PersistentTemporaryDirectory: # Supress pyright unused symbol warnings
    pass

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
    'checkpoint_registry.json': '{"1000": {"checkpoint_name": "checkpoint-1000", "global_step": 1000, "segment_number": 1}, "2000": {"checkpoint_name": "checkpoint-2000", "global_step": 2000, "segment_number": 2}}'
}

def test_construction_with_remote_synchronizer(writable_s3_uri):
    bucket, prefix = TempS3Object.parse_s3_uri(writable_s3_uri)
    if prefix: # suppress pyright warnings about unused var
        pass
    with TempS3Object(writable_s3_uri).TemporaryDirectory(delete=True) as temp_s3_dir:
        temp_s3_uri = f's3://{bucket}/{temp_s3_dir}'
        copy_dict_to_s3(checkpoints_content, temp_s3_uri)
        with tempfile.TemporaryDirectory() as temp_local_dir:
        # with PersistentTemporaryDirectory() as temp_local_dir:
            synchronizer_s3_uri = f's3://{bucket}/{temp_s3_dir}'

            synchronizer = RemoteCheckpointSynchronizer.from_uri(synchronizer_s3_uri, local_output_dir = temp_local_dir)
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
            for rel_path in checkpoints_content.keys():
                full_path = os.path.join(temp_local_dir,rel_path)
                # if it begins with checkpoint_name
                if rel_path.startswith(checkpoint_name+'/') or rel_path == 'checkpoint_registry.json':
                    assert os.path.exists(full_path)
                    expected_content = checkpoints_content[rel_path]
                    assert dir_dict[rel_path] == expected_content
                else:
                    assert not os.path.exists(full_path)

def test_construction_without_remote_synchronizer_or_data():
    with tempfile.TemporaryDirectory() as temp_local_dir:
        registry = CheckpointRegistry(output_dir = temp_local_dir)
        assert registry
        assert registry.latest_step() == None

def test_construction_without_remote_synchronizer_but_local_data():
    with tempfile.TemporaryDirectory() as temp_local_dir:
        copy_dict_to_fs(checkpoints_content, temp_local_dir)
        registry = CheckpointRegistry(output_dir = temp_local_dir)
        assert registry
        latest_step = registry.latest_step()
        assert latest_step == 2000
        checkpoint_2000 = registry.get_checkpoint_for_step(latest_step)
        assert checkpoint_2000
        dd = DirectoryDict(checkpoint_2000.path())
        for checkpoint_rel_path in dd.keys():
            output_dir_rel_path = f'{checkpoint_2000.checkpoint_name}/{checkpoint_rel_path}'
            assert dd[checkpoint_rel_path] == checkpoints_content[output_dir_rel_path]

def test_uploads_of_new_checkpoints(writable_s3_uri):
    synchronizer_s3_uri = writable_s3_uri + '.extra_temp'
    with tempfile.TemporaryDirectory() as temp_local_dir:
        copy_dict_to_fs(checkpoints_content, temp_local_dir)
        synchronizer = RemoteCheckpointSynchronizer.from_uri(synchronizer_s3_uri, local_output_dir = temp_local_dir)
        registry = CheckpointRegistry(output_dir = temp_local_dir, remote_synchronizer = synchronizer)
        # initiate an async upload
        registry.save()
        assert registry.upload_in_progress
        registry.finish_up()
        assert not registry.upload_in_progress
        bucket, prefix = TempS3Object.parse_s3_uri(synchronizer_s3_uri)
        s3d = S3Dict(bucket)
        for s3_path in s3d.keys():
            if s3_path.startswith(prefix+'/'):
                checkpoint_rel_path = s3_path[len(prefix)+1:]
                assert s3d[s3_path] == checkpoints_content[checkpoint_rel_path]


