import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn

from logger_utils import CSVWriter
from dataset import get_dataset_loader
from model import ImageToImageConditionalGAN

def train_gan(FLAGS):
    if not os.path.isdir(FLAGS.dir_model):
        os.makedirs(FLAGS.dir_model)

    csv_writer = CSVWriter(
        file_name=os.path.join(FLAGS.dir_model, FLAGS.file_logger_train),
        column_names=["epoch", "loss_gen_gan", "loss_gen_l1", "loss_dis_real", "loss_dis_fake"]
    )

    train_dataset_loader = get_dataset_loader(
        FLAGS.dir_dataset_train, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size
    )

    try:
        device = torch.device("cuda")
    except:
        print("CUDA device not found, so exiting....")
        sys.exit(0)
    cond_gan_model = ImageToImageConditionalGAN(device)
    cond_gan_model.to(device)
    cond_gan_model.train()

    print("Training started")
    for epoch in range(1, FLAGS.num_epochs + 1):
        epoch_start_time = time.time()
        for data in train_dataset_loader:
            cond_gan_model.setup_input(data)
            cond_gan_model.optimize_params()
        epoch_end_time = time.time()
        losses = cond_gan_model.get_current_losses()
        print(f"epoch : {epoch}, time : {(epoch_end_time - epoch_start_time):.3f} sec.")
        print(f"loss gen gan : {losses['loss_gen_gan']:.6f}, loss gen l1 : {losses['loss_gen_l1']:.6f}")
        print(f"loss dis real : {losses['loss_dis_real']:.6f}, loss dis fake : {losses['loss_dis_fake']:.6f}\n")
        csv_writer.write_row(
            [
                epoch,
                losses["loss_gen_gan"],
                losses["loss_gen_l1"],
                losses["loss_dis_real"],
                losses["loss_dis_fake"],
            ]
        )
        torch.save(cond_gan_model.state_dict(), os.path.join(FLAGS.dir_model, f"{FLAGS.file_model}_{epoch}.pt"))
    print("Training completed")
    return

def main():
    batch_size = 16
    num_epochs = 100
    image_size = 320
    file_model = "colorizer_cgan"
    file_logger_train = "train_metrics.csv"
    dir_dataset_train = "/home/abhishek/Desktop/machine_learning/coco_2017/all_train/"
    dir_model = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="batch size to use for training")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="num epochs to train the model")
    parser.add_argument("--image_size", default=image_size,
        type=int, help="image size used to train the model")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="file name of the model to be used for saving")
    parser.add_argument("--file_logger_train", default=file_logger_train,
        type=str, help="file name of the logger csv file with train losses")
    parser.add_argument("--dir_dataset_train", default=dir_dataset_train,
        type=str, help="full directory path to train dataset")
    parser.add_argument("--dir_model", default=dir_model,
        type=str, help="full directory path to save model files")

    FLAGS, unparsed = parser.parse_known_args()
    train_gan(FLAGS)
    return

if __name__ == "__main__":
    main()
