
import os
from typing import Union


class DirectoryDict:
    """
    A class to wrap a filesystem directory tree in dict-like semantics.

    # Example usage:
    dir_dict = DirectoryDict('/path/to/directory')
    print(dir_dict.keys())  # List all file names
    print(dir_dict.items())  # List all file names and their contents

    # Access contents of a file as if it were an item in a dictionary
    relative_path = 'subdir/myfile.txt'
    print(dir_dict[relative_path])
    """

    def __init__(self, path:str):
        self.path = path

    def __getitem__(self, key:str ) -> str:
        file_path = os.path.join(self.path, key)
        if os.path.isdir(file_path):
            raise TypeError(f'{file_path} is a directory')
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise KeyError(key)

    def __setitem__(self, key:str, value:str):
        file_path = os.path.join(self.path, key)
        file_dir = os.path.dirname(file_path)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        with open(file_path, 'w') as file:
            file.write(value)

    def __delitem__(self, key:str):
        file_path = os.path.join(self.path, key)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise KeyError(key)

    def get(self,key:str,default:Union[str,None] = None) -> Union[str,None]:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]

    def items(self) -> list[tuple[str,str]]:
        return [(f, self[f]) for f in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

