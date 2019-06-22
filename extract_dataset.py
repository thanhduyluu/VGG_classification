import tarfile
import os
import tarfile
import zipfile

root = os.getcwd()

path = root + '/dataset_image/'

file = os.listdir(path)

def extract_file(fname, to_directory):
    if (fname.endswith("tar.xz")):
        tar = tarfile.open(fname, "r:xz")
        tar.extractall(to_directory)	
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(to_directory)
        tar.close()

print(path)
for i in file:
    ex = path + i
    extract_file(ex,path)
    os.remove(ex)

