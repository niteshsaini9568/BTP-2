import torch
import argparse
from torchsummary import summary

from model import ImageToImageConditionalGAN

def get_model_summary(FLAGS):
    try:
        device = torch.device("cuda")
    except:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    cond_gan_model = ImageToImageConditionalGAN(device)
    cond_gan_model.eval()
    #cond_gan_model.load_state_dict(torch.load(FLAGS.model_checkpoint))
    cond_gan_model.to(device)

    print("Generator model summary")
    summary(cond_gan_model.net_gen, (3, FLAGS.image_size, FLAGS.image_size), FLAGS.batch_size)

    print("\n\nDiscriminator model summary")
    summary(cond_gan_model.net_dis, (3, FLAGS.image_size, FLAGS.image_size), FLAGS.batch_size)
    return

def main():
    batch_size = 1
    image_size = 320

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="batch size to use for training")
    parser.add_argument("--image_size", default=image_size,
        type=int, help="image size used to train the model")

    FLAGS, unparsed = parser.parse_known_args()
    get_model_summary(FLAGS)
    return

if __name__ == "__main__":
    main()
