
import inspect
from .directory_dict import DirectoryDict
from .s3_dict import S3Dict
from .temp_s3_object import TempS3Object

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

def copy_dict_to_s3(source_dict: dict[str,str], s3_uri: str):
    s3d = S3Dict(s3_uri)
    for rel_path, content in source_dict.items():
        s3d[rel_path] =  content

