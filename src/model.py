import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from loss import GANLoss

class ResNetEncoder(nn.Module):
    """
    Defines ResNet-34 Encoder
    """
    def __init__(self, pretrained=True):
        """
        ---------
        Arguments
        ---------
        pretrained : bool (default=True)
            boolean to control whether to use a pretrained resnet model or not
        """
        super().__init__()
        self.resnet34 = resnet34(pretrained=pretrained)

    def forward(self, x):
        self.block1 = self.resnet34.conv1(x)
        self.block1 = self.resnet34.bn1(self.block1)
        self.block1 = self.resnet34.relu(self.block1)   # [64, H/2, W/2]

        self.block2 = self.resnet34.maxpool(self.block1)
        self.block2 = self.resnet34.layer1(self.block2)  # [64, H/4, W/4]
        self.block3 = self.resnet34.layer2(self.block2)  # [128, H/8, W/8]
        self.block4 = self.resnet34.layer3(self.block3)  # [256, H/16, W/16]
        self.block5 = self.resnet34.layer4(self.block4)  # [512, H/32, W/32]
        return self.block5

class UNetDecoder(nn.Module):
    """
    Defines UNet Decoder
    """
    def __init__(self, encoder_net, out_channels=2):
        """
        ---------
        Arguments
        ---------
        encoder_net : PyTorch model object of the encoder
            PyTorch model object of the encoder
        out_channels : int (default=2)
            number of output channels of UNet Decoder
        """
        super().__init__()
        self.encoder_net = encoder_net
        self.up_block1 = self.up_conv_block(512, 256, use_dropout=True)
        self.conv_reduction_1 = nn.Conv2d(512, 256, kernel_size=1)

        self.up_block2 = self.up_conv_block(256, 128, use_dropout=True)
        self.conv_reduction_2 = nn.Conv2d(256, 128, kernel_size=1)

        self.up_block3 = self.up_conv_block(128, 64)
        self.conv_reduction_3 = nn.Conv2d(128, 64, kernel_size=1)

        self.up_block4 = self.up_conv_block(64, 64)
        self.conv_reduction_4 = nn.Conv2d(128, 64, kernel_size=1)

        self.up_block5 = self.final_up_conv_block(conv_tr_in_channels=64, conv_tr_out_channels=32, out_channels=out_channels)

    def forward(self, x):
        self.up_1 = self.up_block1(x)   # [256, H/16, W/16]
        self.up_1 = torch.cat([self.encoder_net.block4, self.up_1], dim=1)     # [512, H/16, W/16]
        self.up_1 = self.conv_reduction_1(self.up_1)    # [256, H/16, W/16]

        self.up_2 = self.up_block2(self.up_1)   # [128, H/8, W/8]
        self.up_2 = torch.cat([self.encoder_net.block3, self.up_2], dim=1)     # [256, H/8, H/8]
        self.up_2 = self.conv_reduction_2(self.up_2)    # [128, H/8, W/8]

        self.up_3 = self.up_block3(self.up_2)   # [64, H/4, W/4]
        self.up_3 = torch.cat([self.encoder_net.block2, self.up_3], dim=1)     # [128, H/4, W/4]
        self.up_3 = self.conv_reduction_3(self.up_3)    # [64, H/4, W/4]

        self.up_4 = self.up_block4(self.up_3)   # [64, H/2, W/2]
        self.up_4 = torch.cat([self.encoder_net.block1, self.up_4], dim=1)     # [128, H/2, W/2]
        self.up_4 = self.conv_reduction_4(self.up_4)    # [64, H/2, W/2]

        self.out_features = self.up_block5(self.up_4)    # [2, H, W]
        return self.out_features

    def final_up_conv_block(self, conv_tr_in_channels, conv_tr_out_channels, out_channels, conv_tr_kernel_size=4):
        """
        ---------
        Arguments
        ---------
        conv_tr_in_channels : int
            number of input channels for conv transpose
        conv_tr_out_channels : int
            number of output channels for conv transpose
        out_channels : int
            number of output channels in the final layer
        conv_tr_kernel_size : int (default=4)
            kernel size for convolution transpose layer

        -------
        Returns
        -------
        A sequential block depending on the input arguments
        """
        final_block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(conv_tr_in_channels, conv_tr_out_channels, kernel_size=conv_tr_kernel_size, stride=2, padding=1, bias=False),
            nn.Conv2d(conv_tr_out_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )
        return final_block

    def up_conv_block(self, in_channels, out_channels, conv_tr_kernel_size=4, use_dropout=False):
        """
        ---------
        Arguments
        ---------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        use_dropout : bool (default=False)
            boolean to control whether to use dropout or not [induces randomness - used instead of random noise vector as input in Generator]
        conv_tr_kernel_size : int (default=4)
            kernel size for convolution transpose layer

        -------
        Returns
        -------
        A sequential block depending on the input arguments
        """
        if use_dropout:
            block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=conv_tr_kernel_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
            )
        else:
            block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=conv_tr_kernel_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        return block

class ResUNet(nn.Module):
    """
    Defines Residual UNet model
    """
    def __init__(self):
        super().__init__()
        self.encoder_net = ResNetEncoder()
        self.decoder_net = UNetDecoder(self.encoder_net)

    def forward(self, x):
        self.encoder_features = self.encoder_net(x)
        self.decoder_features = self.decoder_net(self.encoder_features)
        return self.decoder_features

class Generator(nn.Module):
    """
    Defines a Generator in a GAN
    """
    def __init__(self):
        super().__init__()
        self.res_u_net = ResUNet()

    def forward(self, x):
        return self.res_u_net(x)

class PatchDiscriminatorGAN(nn.Module):
    """
    Defines a Patch discriminator for GAN
    """
    def __init__(self, in_channels, num_filters=64, num_blocks=3):
        """
        ---------
        Arguments
        ---------
        in_channels : int
            number of input channels for Discriminator
        num_filters : int (default=64)
            number of filters in the first layer of Discriminator
        num_blocks : int (default=3)
            number of blocks to be used in the Discriminator
        """
        super().__init__()
        model_blocks = [self.get_conv_block(in_channels, num_filters, is_batch_norm=False)]
        for i in range(num_blocks):
            if i != num_blocks-1:
                model_blocks += [self.get_conv_block(num_filters*(2**i), num_filters*(2**(i+1)))]
            else:
                model_blocks += [self.get_conv_block(num_filters*(2**i), num_filters*(2**(i+1)), stride=1)]
        model_blocks += [self.get_conv_block(num_filters*(2**num_blocks), 1, stride=1, is_batch_norm=False, is_activation=False)]
        self.model = nn.Sequential(*model_blocks)

    def get_conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, is_batch_norm=True, is_activation=True):
        """
        ---------
        Arguments
        ---------
        in_channels : int
            input number of channels
        out_channels : int
            output number of channels
        kernel_size : int
            convolution kernel size
        stride : int
            stride to be used for convolution
        padding : int
            padding to be used for convolution
        is_batch_norm : bool
            boolean to control whether to add a batchnorm layer to the block
        is_activation : bool
            boolean to control whether to add an activation function to the block

        -------
        Returns
        -------
        a sequential block depending on the input arguments
        """
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=not(is_batch_norm))
        ]
        if is_batch_norm:
            block += [nn.BatchNorm2d(out_channels)]
        if is_activation:
            block += [nn.ELU()]
        return nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)

class ImageToImageConditionalGAN(nn.Module):
    """
    Defines Image (domain A) to Image (domain B) Conditional Adversarial Network
    """
    def __init__(self, device, lr_gen=2e-4, lr_dis=2e-4, beta1=0.5, beta2=0.999, lambda_=100.0):
        super().__init__()
        self.device = device
        self.loss_names = ["gen_gan", "gen_l1", "dis_real", "dis_fake"]
        self.lambda_ = lambda_
        self.net_gen = Generator()
        self.net_dis = PatchDiscriminatorGAN(in_channels=3)

        self.criterion_GAN = GANLoss().to(self.device)
        self.criterion_l1 = nn.L1Loss()

        self.optimizer_gen = torch.optim.Adam(self.net_gen.parameters(), lr=lr_gen, betas=(beta1, beta2))
        self.optimizer_dis = torch.optim.Adam(self.net_dis.parameters(), lr=lr_dis, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """
        ---------
        Arguments
        ---------
        model : model object
            PyTorch model object
        requires_grad : bool (default=True)
            boolean to control whether the model requires gradients or not
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

    def setup_input(self, data):
        """
        ---------
        Arguments
        ---------
        data : dict
            dictionary object containing image data of domains 1 and 2
        """
        self.real_domain_1 = data["domain_1"].to(self.device)
        self.real_domain_2 = data["domain_2"].to(self.device)

        if self.device == torch.device("cuda"):
            self.real_domain_1_1_ch = self.real_domain_1[:, 0, :, :]
            self.real_domain_1_1_ch = self.real_domain_1_1_ch[:, None, :, :]
        else:
            self.real_domain_1_1_ch = self.real_domain_1[:, :, :, 0]
            self.real_domain_1_1_ch = self.real_domain_1_1_ch[:, :, :, None]

    def forward(self):
        # compute fake image in domain_2: Generator(domain_1)
        self.fake_domain_2 = self.net_gen(self.real_domain_1)

    def backward_gen(self):
        """
        Calculate GAN and L1 loss for generator
        """
        # first, Generator(domain_1) should try to fool the Discriminator
        fake_domain_12 = torch.cat((self.real_domain_1_1_ch, self.fake_domain_2), dim=1)
        pred_fake = self.net_dis(fake_domain_12)
        self.loss_gen_gan = self.criterion_GAN(pred_fake, True)

        # second, Generator(domain_1) = domain_2,
        # i.e. output predicted by Generator should be close the domain_2
        self.loss_gen_l1 = self.criterion_l1(self.fake_domain_2, self.real_domain_2) * self.lambda_

        # compute the combined loss
        self.loss_gen = self.loss_gen_gan + self.loss_gen_l1
        self.loss_gen.backward()

    def backward_dis(self):
        """
        Calculate GAN loss for discriminator
        """
        # Fake
        fake_domain_12 = torch.cat((self.real_domain_1_1_ch, self.fake_domain_2), dim=1)
        # stop backprop to generator by detaching fake_domain_12
        pred_fake = self.net_dis(fake_domain_12.detach())
        # Discriminator should identify the fake image
        self.loss_dis_fake = self.criterion_GAN(pred_fake, False)

        # Real
        real_domain_12 = torch.cat((self.real_domain_1_1_ch, self.real_domain_2), dim=1)
        pred_real = self.net_dis(real_domain_12)
        # Discriminator should identify the real image
        self.loss_dis_real = self.criterion_GAN(pred_real, True)

        # compute the combined loss
        self.loss_dis = (self.loss_dis_fake + self.loss_dis_real) * 0.5
        self.loss_dis.backward()

    def optimize_params(self):
        # compute fake image in domain_2: Generator(domain_1)
        self.forward()

        """
        --------------------
        Update Discriminator
        --------------------
        # enable backprop for Discriminator
        # set Discriminator's gradients to zero
        # compute gradients for Discriminator
        # update Discriminator's weights
        """
        self.set_requires_grad(self.net_dis, True)
        self.optimizer_dis.zero_grad()
        self.backward_dis()
        self.optimizer_dis.step()

        """
        ----------------
        Update Generator
        ----------------
        # Discriminator requires no gradients when optimizing Generator
        # set Generator's gradients to zero
        # calculate gradients for Generator
        # update Generator's weights
        """
        self.set_requires_grad(self.net_dis, False)
        self.optimizer_gen.zero_grad()
        self.backward_gen()
        self.optimizer_gen.step()

    def get_current_losses(self):
        all_losses = dict()
        for loss_name in self.loss_names:
            all_losses["loss_" + loss_name] = float(getattr(self, "loss_" + loss_name))
        return all_losses
