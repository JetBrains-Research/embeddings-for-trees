import boto3
from os.path import getsize, basename
from tqdm.auto import tqdm

# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html


class ProgressPercentage(object):

    def __init__(self, filename, size=None):
        self._filename = filename
        if size is None:
            self._size = float(getsize(filename))
        else:
            self._size = size
        self._progress_bar = tqdm(total=self._size, unit='iB', unit_scale=True)

    def __call__(self, bytes_amount):
        self._progress_bar.update(bytes_amount)


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(
        file_name, bucket, object_name,
        Callback=ProgressPercentage(file_name)
    )


def download_file(file_name, bucket, object_name):

    s3_client = boto3.client('s3')
    s3_resources = boto3.resource('s3')
    for s3_object in s3_resources.Bucket(bucket).objects.all():
        if s3_object.key == object_name:
            size = s3_object.size
            s3_client.download_file(
                bucket, object_name, file_name,
                Callback=ProgressPercentage(file_name, size)
            )
            return True
    print(f"Can't find {object_name} file")
    return False
