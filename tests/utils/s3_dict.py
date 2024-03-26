
import boto3
from botocore.client import BaseClient as BotocoreBaseClient
from typing import Union, cast

class S3Dict:
    """
    A class to wrap an s3 bucket in dict-like semantics.

    # Example usage:
    s3_dict = S3Dict(bucket_name)
    print(s3_dict.keys())  # List all s3 keys
    print(s3_dict.items())  # List all s3 keys and object values

    # Access an s3 object as if it were an item in a dictionary
    key = 'path/to/my_object'
    print(s3_dict[key])
    """

    bucket_name: str
    client: BotocoreBaseClient

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = cast(BotocoreBaseClient,boto3.client('s3'))

    def __getitem__(self, key: str) -> str:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read().decode('utf-8')
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: str):
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=value)

    def __delitem__(self, key: str):
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def keys(self) -> list[str]:
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name)
        keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    keys.append(obj['Key'])
        return keys

    def items(self) -> list[tuple[str, str]]:
        keys = self.keys()
        return [(key, self[key]) for key in keys]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def get(self, key: str, default:Union[str,None]=None) -> Union[str,None]:
        try:
            return self[key]
        except KeyError:
            return default

