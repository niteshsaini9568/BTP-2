import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn

from image_utils import *
from model import ImageToImageConditionalGAN

def activate_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
    return

def generate_gan_results(FLAGS):
    if not os.path.isdir(FLAGS.dir_generated_results):
        os.makedirs(FLAGS.dir_generated_results)

    try:
        device = torch.device("cuda")
    except:
        print("CUDA device not found, so exiting....")
        sys.exit(0)
    cond_gan_model = ImageToImageConditionalGAN(device)
    cond_gan_model.eval()
    cond_gan_model.load_state_dict(torch.load(FLAGS.model_checkpoint))
    cond_gan_model.to(device)
    cond_gan_model.net_gen.apply(activate_dropout)

    list_test_files = sorted(os.listdir(FLAGS.dir_dataset_test))
    num_test_files = len(list_test_files)
    print(f"Num test images to be processed : {num_test_files}")

    for idx in range(num_test_files):
        img = read_image(os.path.join(FLAGS.dir_dataset_test, list_test_files[idx]))

        if len(img.shape) == 3:
            # if input image is RGB
            img_rgb_resized = resize_image(img, (FLAGS.image_size, FLAGS.image_size))
            # resized rgb image is in [0, 255]
            img_lab = convert_rgb2lab(img_rgb_resized)
            img_l = img_lab[:, :, 0]
            # L channel is in [0, 100]
        else:
            # if input image is grayscale
            img_gray_resized = resize_image(img, (FLAGS.image_size, FLAGS.image_size))
            # resized grayscale is in [0, 1]
            img_l = rescale_grayscale_image_l_channel(img_gray_resized)
            # L channel is in [0, 100]

        # apply pre-processing on L channel image
        img_l_preprocessed = apply_image_l_pre_processing(img_l)
        # repeat L channel 3 times because ResNet needs a 3 channel input
        img_l_preprocessed = np.repeat(np.expand_dims(img_l_preprocessed, axis=-1), 3, axis=-1)
        img_l_preprocessed = np.expand_dims(img_l_preprocessed, axis=0)
        # NCHW format
        img_l_preprocessed = np.transpose(img_l_preprocessed, (0, 3, 1, 2))

        img_l_tensor = torch.tensor(img_l_preprocessed).float()
        img_l_tensor = img_l_tensor.to(device, dtype=torch.float)

        # Use Generator network to generate ab channels with pre-processed and repeated L channel as the input
        gen_img_ab_tensor = cond_gan_model.net_gen(img_l_tensor)
        gen_img_ab = gen_img_ab_tensor.detach().cpu().numpy()
        gen_img_ab = np.squeeze(gen_img_ab)
        gen_img_ab = np.transpose(gen_img_ab, [1, 2, 0])
        gen_img_ab_postprocessed = apply_image_ab_post_processing(gen_img_ab)

        # concat L and Generator network generated ab channels
        gen_img_lab = np.concatenate((np.expand_dims(img_l, axis=-1), gen_img_ab_postprocessed), axis=-1)
        # convert Lab to RGB
        gen_img_rgb = convert_lab2rgb(gen_img_lab)
        gen_img_rgb = gen_img_rgb * 255
        gen_img_rgb = gen_img_rgb.astype(np.uint8)
        save_image_rgb(os.path.join(FLAGS.dir_generated_results, list_test_files[idx]), gen_img_rgb)
        print(f"{(idx+1)} / {num_test_files} done")
    print(f"Generating results completed successfully, generated results in : {FLAGS.dir_generated_results}")
    return

def main():
    image_size = 320
    dir_dataset_test = "/home/abhishek/sample_dataset/test/"
    dir_generated_results = "generated_cgan_results/"
    model_checkpoint = "sample.pt"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_size", default=image_size,
        type=int, help="image size used to generate results with the model")
    parser.add_argument("--model_checkpoint", default=model_checkpoint,
        type=str, help="full path of model file to be used for generating results")
    parser.add_argument("--dir_dataset_test", default=dir_dataset_test,
        type=str, help="full directory path to test dataset")
    parser.add_argument("--dir_generated_results", default=dir_generated_results,
        type=str, help="full directory path to save generated results")

    FLAGS, unparsed = parser.parse_known_args()
    generate_gan_results(FLAGS)

if __name__ == "__main__":
    main()
