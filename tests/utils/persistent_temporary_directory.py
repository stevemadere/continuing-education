import tempfile
from logging import warning

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

