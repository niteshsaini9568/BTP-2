import os
import shutil
import argparse
from skimage.io import imread

def move_grayscale_images(FLAGS):
    if not os.path.isdir(FLAGS.dir_output):
        os.makedirs(FLAGS.dir_output)
    list_images = sorted(os.listdir(FLAGS.dir_input))
    num_images = len(list_images)

    print(f"num images to be processed : {num_images}")

    for idx in range(num_images):
        img = imread(os.path.join(FLAGS.dir_input, list_images[idx]))
        if len(img.shape) == 2:
            print(list_images[idx])
            shutil.move(os.path.join(FLAGS.dir_input, list_images[idx]), os.path.join(FLAGS.dir_output, list_images[idx]))

def main():
    dir_input = "sample_dir_1"
    dir_output = "sample_dir_2"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_input", default=dir_input,
        type=str, help="full path of the directory with train images")
    parser.add_argument("--dir_output", default=dir_output,
        type=str, help="full path of the directory to move grayscale images to")

    FLAGS, unparsed = parser.parse_known_args()
    move_grayscale_images(FLAGS)

main()
