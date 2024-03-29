
import boto3
from botocore.client import BaseClient as BotocoreBaseClient
from typing import Union, cast

class S3Dict:
    """
    A class to wrap an s3 bucket in dict-like semantics.

    # Example usage:
    s3_uri = f's3://{bucket_name}/{prefix}'
    s3_dict = S3Dict(s3_uri)
    print(s3_dict.keys())  # List all s3 keys
    print(s3_dict.items())  # List all s3 keys and object values

    # Access an s3 object as if it were an item in a dictionary
    key = 'relative/path/to/my_object'
    print(s3_dict[key])
    """

    bucket_name: str
    prefix: str
    client: BotocoreBaseClient

    def __init__(self, s3_uri: str):
        self.bucket_name, self.prefix = S3Dict.parse_s3_uri(s3_uri)
        # ensure exactly one trailing slash on prefix
        self.prefix = self.prefix.rstrip('/') + '/'
        self.client = cast(BotocoreBaseClient,boto3.client('s3'))

    def __getitem__(self, key: str) -> str:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=self._s3_key(key))
            return response['Body'].read().decode('utf-8')
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: str):
        self.client.put_object(Bucket=self.bucket_name, Key=self._s3_key(key), Body=value)

    def __delitem__(self, key: str):
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=self._s3_key(key))
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def _s3_key(self, key:str) -> str:
        return self.prefix + key.lstrip('/')

    def keys(self) -> list[str]:
        # find all keys with prefix self.prefix
        # and return them as a list
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
        dict_keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    dict_key = obj['Key'][len(self.prefix):]
                    dict_keys.append(dict_key)
        return dict_keys

    def items(self) -> list[tuple[str, str]]:
        keys = self.keys()
        return [(key, self[key]) for key in keys]

    # FIXME:  This is ridiculously inefficient
    def __iter__(self):
        return iter(self.keys())

    # FIXME:  This is ridiculously inefficient
    def __len__(self):
        return len(self.keys())

    def get(self, key: str, default:Union[str,None]=None) -> Union[str,None]:
        try:
            return self[key]
        except KeyError:
            return default

    @staticmethod
    def parse_s3_uri(s3_uri:str) -> tuple[str,str]:
        assert s3_uri.startswith("s3://"), "S3 URI must start with 's3://'"
        parts:list[str] = s3_uri[5:].split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        return bucket_name, prefix

