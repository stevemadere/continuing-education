
import inspect
from .directory_dict import DirectoryDict

def current_test_name() -> str:
    cframe =  inspect.currentframe()
    if cframe and cframe.f_back and cframe.f_back.f_code:
        return cframe.f_back.f_code.co_name
    else:
        return 'no freaking idea what function I am in right now'


def copy_dict_to_fs(source_dict: dict[str,str], fs_dir_path: str):
    dd = DirectoryDict(fs_dir_path)
    for rel_path, content in source_dict.items():
        dd[rel_path] =  content

