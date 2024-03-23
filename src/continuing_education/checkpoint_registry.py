import os
import re
import json
from typing import Callable, Dict, Union, Any, cast, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import subprocess
from queue import Queue

REGISTRY_FILENAME='checkpoint_registry.json'

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


class RemoteCheckpointSynchronizer(ABC):
    local_output_dir: str
    remote_uri: str

    class Task:
        func: Callable
        kwargs: dict

        def __init__(self, func:Callable, kwargs:dict) -> None:
            self.func = func
            self.kwargs = kwargs

        def execute(self) -> Any:
            return self.func(**self.kwargs)

    class TaskQueue(Queue):
        def put(self, item: 'RemoteCheckpointSynchronizer.Task', *args, **kwargs) -> None:
            return super().put(item, *args, **kwargs)
        def get(self, *args, **kwargs) -> 'RemoteCheckpointSynchronizer.Task':
            return cast('RemoteCheckpointSynchronizer.Task',super().get(*args,**kwargs))


    def __init__(self, local_output_dir: str, remote_uri: str = ""):

        assert self.__class__.can_handle_uri(remote_uri)

        self.local_output_dir = local_output_dir
        self.remote_uri = remote_uri

        self._task_queue = RemoteCheckpointSynchronizer.TaskQueue()
        self._thread = threading.Thread(target=self._worker)
        self._thread.daemon = True  # Ensure thread exits when main program exits
        self._thread.start()

    def _worker(self):
        while True:
            task:RemoteCheckpointSynchronizer.Task = self._task_queue.get()  # Waits for a task to be available
            try:
                task.execute()
            finally:
                self._task_queue.task_done()  # Mark the task as done

    _handler_registry: list[Type['RemoteCheckpointSynchronizer']] = []


    @classmethod
    def register_thyself(cls: Type['RemoteCheckpointSynchronizer']) -> None:
        print(f'registering {cls}')
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

    def upload_all_async(self):
        # Add the sync operation to the queue
        self._task_queue.put(RemoteCheckpointSynchronizer.Task(self._upload_all_sync,{}))



class S3RemoteCheckpointSynchronizer(RemoteCheckpointSynchronizer):
                         
     def _upload_all_sync(self):
         cmd = f'aws s3 sync "{self.local_output_dir}" "{self.remote_uri}"'
         subprocess.run(cmd, shell=True, check=True)

     def download_one_sync(self, relative_path: str) -> Union[str,None]:
         full_local_path = os.path.join(self.local_output_dir,relative_path)
         # use s3 sync first in case the source is a directory
         cmd = f'aws s3 sync "{self.remote_uri}/{relative_path}" "{full_local_path}"'
         subprocess.run(cmd, shell=True, check=True)
         # if the source was an ordinary file, the s3 sync would be a no-op so try to just s3 cp it
         if (not os.path.exists(full_local_path)):
             cmd = f'aws s3 cp "{self.remote_uri}/{relative_path}" "{full_local_path}"'
             subprocess.run(cmd, shell=True, check=True)
         return full_local_path if (os.path.exists(full_local_path)) else None

     def download_all_sync(self) -> bool:
         cmd = f'aws s3 sync "{self.remote_uri}" "{self.local_output_dir}"'
         subprocess.run(cmd, shell=True, check=True)
         return os.path.exists(self.local_output_dir)

     @classmethod
     def can_handle_uri(cls, uri: str) -> bool:
         # does it match s3: ?
         return bool(re.match("^s3://",uri))

S3RemoteCheckpointSynchronizer.register_thyself()


class CheckpointRegistry():
    output_dir: str
    remote_synchronizer: Union[RemoteCheckpointSynchronizer,None]
    _registry: Dict[int,CheckpointInfo]  # it is necessary to have string keys because of json serialization

    def __init__(self, output_dir: str, remote_synchronizer: Union[RemoteCheckpointSynchronizer,None]):
        self.output_dir = output_dir
        self.remote_synchronizer = remote_synchronizer
        if self.remote_synchronizer:
            assert self.output_dir == self.remote_synchronizer.local_output_dir
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
            self.remote_synchronizer.upload_all_async()

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
            self.remote_synchronizer.download_one_sync(supposed_checkpoint_info.path())
        if supposed_checkpoint_info and supposed_checkpoint_info.exists():
            return supposed_checkpoint_info
        else:
            return None

    def latest_step(self) -> Union[int,None]:
        if self._registry.keys():
            return max(k for k in self._registry.keys())
        else:
            return None


