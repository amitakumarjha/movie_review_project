import tarfile

# Replace 'path/to/aclImdb_v1.tar.gz' with the actual path to your tar file
tar_file_path = 'aclImdb_v1.tar.gz'

# Open the tar file in read mode
with tarfile.open(tar_file_path, 'r:gz') as tar:
    # Extract all contents of the tar file to the current directory
    tar.extractall()