import os
import json
from typing import Dict, Union, Any
from dataclasses import dataclass

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

class CheckpointRegistry():
    output_dir: str
    _registry: Dict[int,CheckpointInfo]  # it is necessary to have string keys because of json serialization

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
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

    def _load(self) -> None:
        rpath = self.registry_path()
        loaded_registry:Dict[int,CheckpointInfo] = {}
        if os.path.exists(rpath):
            with open(rpath, 'r') as f:
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
        if supposed_checkpoint_info and supposed_checkpoint_info.exists():
            return supposed_checkpoint_info
        else:
            return None

    def latest_step(self) -> Union[int,None]:
        if self._registry.keys():
            return max(k for k in self._registry.keys())
        else:
            return None


