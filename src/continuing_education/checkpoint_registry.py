import os
import sys 
if sys: # suppress pyright warnings about unused sys when all diagnostic prints are commented out
    pass

import re
import json
from typing import Callable, Dict, Union, Any, cast, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import subprocess
import queue
from queue import Queue
import boto3
from botocore.exceptions import ClientError
from datetime import timezone

REGISTRY_FILENAME='checkpoint_registry.json'

verbose = False

def debug_print(message:str, file=sys.stdout):
    if verbose:
        print(message, file=file)

@dataclass
class CheckpointInfo:
    checkpoint_name: str
    global_step: int
    segment_number: Union[int,None]
    output_dir: str

    def mini_dict(self) -> Dict[str,Any]:
        return {k: v for k,v in self.__dict__.items() if k != 'output_dir'}

    def path(self) -> str:
        return os.path.join(self.output_dir,self.checkpoint_name)

    def exists(self) -> bool:
        return os.path.exists(self.path())

    @classmethod
    def from_mini_dict(cls, output_dir, mini_dict: Dict[str,Any]) -> 'CheckpointInfo':
        instance = cls(output_dir = output_dir, **mini_dict)
        return instance


import time

class RemoteCheckpointSynchronizer(ABC):
    local_output_dir: str
    remote_uri: str
    draining_the_queue: bool

    _handler_registry: list[Type['RemoteCheckpointSynchronizer']] = []
    _worker_thread: threading.Thread

    def __init__(self, local_output_dir: str, remote_uri: str = ""):

        assert self.__class__.can_handle_uri(remote_uri)

        self.local_output_dir = local_output_dir
        self.remote_uri = remote_uri

        self.draining_the_queue = False
        self._task_queue = RemoteCheckpointSynchronizer.TaskQueue()
        self._worker_thread = threading.Thread(target=self._work_the_queue)
        self._worker_thread.daemon = True  # Ensure thread exits when main program exits
        self._worker_thread.start()


    def _work_the_queue(self):
        all_done = False
        while not all_done:
            task: Union[RemoteCheckpointSynchronizer.Task,None] = None
            try:
                task = self._task_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                if self.draining_the_queue:
                    all_done = True
            if task:
                try:
                    task.execute()
                    task.done = True
                finally:
                    self._task_queue.task_done()  # Mark the task as done

    def finish_up(self):
        self.draining_the_queue = True
        self._worker_thread.join()

    @classmethod
    def register_thyself(cls: Type['RemoteCheckpointSynchronizer']) -> None:
        cls._handler_registry.append(cls)

    @classmethod
    @abstractmethod
    def can_handle_uri(cls, uri: str) -> bool:
        """ Subclasses must override this to return whether or not they can handle the uri in question """
        return False

    @classmethod
    def handler_for_uri(cls,uri) -> Type['RemoteCheckpointSynchronizer']:
        for handler in cls._handler_registry:
            if handler.can_handle_uri(uri):
                return handler
        raise RuntimeError(f'No RemoteCheckpointSynchronizer exists that can handle the uri "{uri}"')

    @classmethod
    def from_uri(cls, uri:str, local_output_dir:str) -> 'RemoteCheckpointSynchronizer':
        klass:Type['RemoteCheckpointSynchronizer'] = cls.handler_for_uri(uri)
        assert klass
        return  klass(local_output_dir = local_output_dir, remote_uri=uri)

    @abstractmethod
    def _upload_all_sync(self):
        """ Subclasses must override this to implement a synchronous upload of everything in the local_output_dir """
        pass

    @abstractmethod
    def download_all_sync(self) -> bool:
        """ 
            Subclasses must overrride this.
            Downloads all files and subdirectories of the checkpoints directory from the remote to local.
            Returns true if the the local directory exists after the operation.
        """
        pass

    @abstractmethod
    def download_one_sync(self,relative_path: str) -> Union[str,None]:
        """ downloads one file or directory under the local_dir and returns either the full path to it or None if it failed to download """
        pass

    def upload_all_async(self) -> 'RemoteCheckpointSynchronizer.Task':
        # Add the sync operation to the queue
        task = RemoteCheckpointSynchronizer.Task(self._upload_all_sync,{})
        self._task_queue.put(task)
        return task

    class Task:
        func: Callable
        kwargs: dict
        done: bool
        result: Any
        error: Union[BaseException, None]

        def __init__(self, func:Callable, kwargs:dict) -> None:
            self.func = func
            self.kwargs = kwargs
            self.done=False
            self.error = None
            self.result = None

        def execute(self) -> None:
            try:
                self.result = self.func(**self.kwargs)
            except Exception as e:
                self.error = e
            finally:
                self.done=True

        def wait_for_it(self, poll_interval = 0.1) -> None:
            while not self.done:
                time.sleep(poll_interval)

    class TaskQueue(Queue):
        def put(self, item: 'RemoteCheckpointSynchronizer.Task', *args, **kwargs) -> None:
            return super().put(item, *args, **kwargs)

        def get(self, *args, **kwargs) -> 'RemoteCheckpointSynchronizer.Task':
            return cast('RemoteCheckpointSynchronizer.Task',super().get(*args,**kwargs))




class S3RemoteCheckpointSynchronizer(RemoteCheckpointSynchronizer):

    def _upload_all_sync(self):
        cmd = f'aws s3 sync "{self.local_output_dir}" "{self.remote_uri}"'
        if not verbose:
            cmd += ' > /dev/null 2>&1'
        subprocess.run(cmd, shell=True, check=True)

    def download_one_sync(self, relative_path: str) -> Union[str,None]:
        full_local_path = os.path.join(self.local_output_dir,relative_path)
        full_uri = f"{self.remote_uri}/{relative_path}"
        # first check if it is an ordinary file and if so, follow sync logic on it.
        # aws s3 sync command only works on directories so I had to implement this:
        if not S3RemoteCheckpointSynchronizer.sync_s3_uri_to_local_file(full_uri,full_local_path):
            # use s3 sync in case the source is a directory
            cmd = f'aws s3 sync "{full_uri}" "{full_local_path}"'
            if not verbose:
                cmd += ' > /dev/null 2>&1'
            debug_print(f'about to run command "{cmd}"', file=sys.stderr)
            subprocess.run(cmd, shell=True, check=True)
        return full_local_path if (os.path.exists(full_local_path)) else None

    def download_all_sync(self) -> bool:
        cmd = f'aws s3 sync "{self.remote_uri}" "{self.local_output_dir}"'
        if not verbose:
            cmd += ' > /dev/null 2>&1'
        subprocess.run(cmd, shell=True, check=True)
        return os.path.exists(self.local_output_dir)

    @classmethod
    def can_handle_uri(cls, uri: str) -> bool:
        # does it match s3: ?
        return bool(re.match("^s3://",uri))

    @staticmethod
    def parse_s3_uri(s3_uri:str) -> tuple[str,str]:
        uri_pattern = r'^s3://([^/]+)/(.+)$'
        match = re.match(uri_pattern, s3_uri)
        if match:
            # Extract bucket name and key from the matched groups
            bucket,key = match.groups()
            return (bucket,key)
        else:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    @staticmethod
    def sync_s3_uri_to_local_file(s3_uri: str, local_file_path: str) -> bool:
        # Parse the S3 URI
        bucket_name, key = S3RemoteCheckpointSynchronizer.parse_s3_uri(s3_uri)

        # Create an S3 client
        s3_client = boto3.client('s3')

        try:
            # Get object metadata
            metadata = s3_client.head_object(Bucket=bucket_name, Key=key)
            assert metadata

            # Check if local file exists
            if os.path.exists(local_file_path):
                # Get local file stats
                local_file_stat = os.stat(local_file_path)
                local_file_size = local_file_stat.st_size
                local_file_mtime = local_file_stat.st_mtime

                # Convert S3 object last modified to timestamp
                s3_last_modified = metadata['LastModified'].replace(tzinfo=timezone.utc).timestamp()

                # Compare file sizes and modification timestamps
                if local_file_size == metadata['ContentLength'] and local_file_mtime >= s3_last_modified:
                    debug_print("Local file is up to date. No download needed.")
                    return True
                else:
                    debug_print("Remote is newer or sizes differ")
                    pass
            else:
                debug_print("Local file does not exist.")
                pass
            # Download the file if size or timestamp differ, or if local file does not exist
            debug_print(f"Attemting to download {s3_uri} to {local_file_path}")
            s3_client.download_file(bucket_name, key, local_file_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                debug_print(f"{s3_uri} does not exist.", file=sys.stderr)
                return False
            else:
                raise

S3RemoteCheckpointSynchronizer.register_thyself()


class CheckpointRegistry():
    output_dir: str
    remote_synchronizer: Union[RemoteCheckpointSynchronizer,None]
    _registry: Dict[int,CheckpointInfo]  # it is necessary to have string keys because of json serialization
    upload_in_progress: Union[RemoteCheckpointSynchronizer.Task,None]

    def __init__(self, output_dir: str, remote_synchronizer: Union[RemoteCheckpointSynchronizer,None] = None):
        self.output_dir = output_dir
        self.remote_synchronizer = remote_synchronizer
        if self.remote_synchronizer:
            assert self.output_dir == self.remote_synchronizer.local_output_dir
        self.upload_in_progress = None
        # ensure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._load()

    def registry_path(self):
        return f"{self.output_dir}/{REGISTRY_FILENAME}"

    def save(self):
        # save _registry safely as json
        realfilename = self.registry_path()
        tempfilename = f"{self.registry_path()}.tmp"
        with open(tempfilename, 'w') as f:
            dumpable = {}
            for steps,checkpoint_info in self._registry.items():
                dumpable[str(steps)] = checkpoint_info.mini_dict()
            json.dump(dumpable, f)
        os.rename(tempfilename, realfilename)
        if self.remote_synchronizer:
            if self.upload_in_progress:
                self.upload_in_progress.wait_for_it()
                self.upload_in_progress = None
            self.upload_in_progress = self.remote_synchronizer.upload_all_async()

    def _load(self) -> None:
        registry_path = self.registry_path()

        if self.remote_synchronizer and not os.path.exists(registry_path):
            self.remote_synchronizer.download_one_sync(REGISTRY_FILENAME)

        loaded_registry:Dict[int,CheckpointInfo] = {}
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                loadable = json.load(f)
                for steps_str,mini_dict in loadable.items():
                    loaded_registry[int(steps_str)] = CheckpointInfo.from_mini_dict(output_dir=self.output_dir, mini_dict = mini_dict)
        self._registry = loaded_registry

    def add_checkpoint(self, global_step: int, checkpoint: str, segment_number: Union[int,None] = None):
        checkpoint_info= CheckpointInfo(output_dir=self.output_dir,
                                        checkpoint_name= checkpoint,
                                        global_step=global_step ,
                                        segment_number= segment_number)
        self._registry[global_step] = checkpoint_info
        self.save()

    def get_checkpoint_for_step(self, global_step: int) -> Union[CheckpointInfo,None]:
        supposed_checkpoint_info: Union[CheckpointInfo,None] = self._registry.get(global_step,None)
        if self.remote_synchronizer and supposed_checkpoint_info and not supposed_checkpoint_info.exists():
            checkpoint_name = supposed_checkpoint_info.checkpoint_name
            debug_print(f'about to download_one_sync({checkpoint_name})')

            self.remote_synchronizer.download_one_sync(checkpoint_name)
        if supposed_checkpoint_info and supposed_checkpoint_info.exists():
            return supposed_checkpoint_info
        else:
            return None

    def latest_step(self) -> Union[int,None]:
        if self._registry.keys():
            return max(k for k in self._registry.keys())
        else:
            return None

    def finish_up(self):
        if self.remote_synchronizer:
            self.remote_synchronizer.finish_up()
            self.upload_in_progress = None

