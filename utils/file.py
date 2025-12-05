from google.cloud import storage

def list_files_in_folder(bucket: str, folder: str):
    """Lista todos los archivos dentro de una carpeta en GCS."""
    client = storage.Client()
    bucket_obj = client.bucket(bucket)

    return [
        blob.name
        for blob in bucket_obj.list_blobs(prefix=folder)
        if not blob.name.endswith("/")
    ]