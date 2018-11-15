import sys, os
import shutil

def create_and_return_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def move_file(src, dest):
    create_and_return_directory(dest)
    shutil.move(src, dest)

def copy_file(src, dest):
    create_and_return_directory(dest)
    shutil.copy(src, dest)
