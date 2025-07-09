import boto3
import os

def upload_to_s3(file_path, bucket_name, object_name=None, aws_access_key_id=None, aws_secret_access_key=None, region=None):
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )
        s3 = session.client('s3')
        s3.upload_file(file_path, bucket_name, object_name)
        return True, f"Uploaded {object_name} to AWS S3 bucket {bucket_name}"
    except Exception as e:
        return False, str(e)
