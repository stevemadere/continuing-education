import boto3
import uuid
from typing import Union, Any

class TempS3Object:
    """
    Provides functionality similar to the tempfile module but on an S3 bucket rather than a filesystem

    Rather than explicitly checking for name collisions, it just trusts that uuids will be unique and
    thus never collide.

    # Example usage:
    uri = 's3://your-bucket-name/optional-prefix/'

    # Using the TemporaryDirectory context manager
    with TempS3Object(uri).TemporaryDirectory(delete=True) as temp_dir:
        print(f"Temporary directory (prefix) created at: {temp_dir}")
        # Do work with temp_dir here
        # assert temp_dir in uri # Your specific logic for validation

    # Using the NamedTemporaryFile context manager
    with TempS3Object(uri).NamedTemporaryFile('sample.txt', delete=True) as temp_file:
        print(f"Temporary file created at: {temp_file}")
        # Do work with temp_file here
    """

    s3_client: Any # Arrrgh!  Dynamic return type from boto3.client().  I can't even get a list of options.
    bucket_name: str
    prefix: str

    def __init__(self, s3_uri: str) -> None:
        self.s3_client = boto3.client('s3')
        self.bucket_name, self.prefix = self.parse_s3_uri(s3_uri)
        #ensure the prefix ends in exaclty one slash, not zero, not more than one
        self.prefix = self.prefix.rstrip('/') + '/'

    @staticmethod
    def parse_s3_uri(s3_uri:str) -> tuple[str,str]:
        assert s3_uri.startswith("s3://"), "S3 URI must start with 's3://'"
        parts:list[str] = s3_uri[5:].split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        return bucket_name, prefix

    class TemporaryObjectPrefix:
        parent: 'TempS3Object'
        delete: bool
        prefix: str

        def __init__(self, parent: 'TempS3Object', delete: bool =True) -> None:
            self.parent = parent
            self.delete = delete
            self.prefix = ""

        def __enter__(self) -> str:
            unique_id = str(uuid.uuid4())
            self.prefix = f"{self.parent.prefix}temp-dir-{unique_id}"
            # No need to create the prefix in S3, as S3 is key-value storage and doesn't require empty directories
            return self.prefix

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type or exc_val or exc_tb: # Supress pyright unused param warnings
                pass
            if self.delete:
                paginator = self.parent.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.parent.bucket_name, Prefix=self.prefix+'/'):
                    keys_to_delete:list[dict[str,str]] = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
                    if keys_to_delete:
                        self.parent.s3_client.delete_objects(
                            Bucket=self.parent.bucket_name,
                            Delete={'Objects': keys_to_delete}
                        )

    class NamedTemporaryObject:
        def __init__(self, parent, delete=True):
            self.parent = parent
            self.delete = delete
            self.key = None

        def __enter__(self):
            self.key = f"{self.parent.prefix}tempfile-{str(uuid.uuid4())}"
            # Initially, don't put any content. Assume caller will upload as needed.
            return self.key

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type or exc_val or exc_tb: # Supress pyright unused param warnings
                pass
            if self.delete:
                self.parent.s3_client.delete_object(Bucket=self.parent.bucket_name, Key=self.key)

    def TemporaryDirectory(self, delete=True):
        return self.TemporaryObjectPrefix(self, delete)

    def NamedTemporaryFile(self, delete=True):
        return self.NamedTemporaryObject(self, delete)
