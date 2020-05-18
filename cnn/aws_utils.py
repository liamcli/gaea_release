import os
import boto3


def upload_to_s3(source, bucket, key):
    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(source, bucket, key)


def upload_directory(path, bucket):
    for root, dirs, files in os.walk(path):
        for f in files:
            full_path = os.path.join(root, f)
            upload_to_s3(full_path, bucket, full_path)


def download_from_s3(key, bucket, target):
    # Make sure directory exists before downloading to it.
    target_dir = os.path.dirname(target)
    if len(target_dir) and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    s3 = boto3.resource("s3")
    try:
        s3.meta.client.download_file(bucket, key, target)
    except Exception as e:
        print(e)
