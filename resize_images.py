import os, sys
from PIL import Image

def resize_image_save(image_path, base_length=128):
    if("gif" in image_path): return
    try:
        im = Image.open(image_path)
    except Exception as e:
        print(e)
        print("Failed to open image")
        return
    if im.size[0] < base_length / 2 or im.size[1] < base_length / 2:
        print("Image too small!")
        return
    resize_path = "resized_faces"
    im_name = os.path.basename(image_path)
    im = im.resize((base_length, base_length))
    im.save(os.path.join(resize_path, im_name))
    print("Saved Image")

def iterate_images_over_directory(image_path):
    images = os.listdir(image_path)
    for image in images:
        abs_path = os.path.join(image_path, image)
        resize_image_save(abs_path)

def main(argv):
    image_path = argv[-1]
    iterate_images_over_directory(image_path)

if __name__ == '__main__':
    main(sys.argv[1:])
