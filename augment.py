import sys, os
from optparse import OptionParser
from argparse import ArgumentParser
from data_prep.augment_dataset import augment_directory, augment_image

def main():
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path")
    parser.add_option("-d", "--directory", dest="directory")
    options, args = parser.parse_args()
    path = ""
    directory = ""
    if options.path:
        print("Path accepted!")
        path = options.path
        augment_image(path)
    elif options.directory:
        print("Directory accepted")
        directory = options.directory
        augment_directory(directory)
    else:
        print("No path or directory given!")
        return


if __name__ == '__main__':
    main()
