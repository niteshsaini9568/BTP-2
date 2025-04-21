import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    """
    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, loss_mode="vanilla", real_label=1.0, fake_label=0.0):
        """
        ---------
        Arguments
        ---------
        loss_mode : str
            GAN loss mode (default="vanilla")
        real_label : bool
            label for real image
        fake_label : bool
            label for fake image
        """
        super().__init__()
        self.loss_mode = loss_mode
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        self.loss = None
        if self.loss_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GANLoss with {self.loss_mode} mode - not implemented yet")

    def get_target_tensor(self, prediction, target_is_real):
        """
        ---------
        Arguments
        ---------
        prediction : tensor
            prediction from a discriminator
        target_is_real : bool
            whether the groundtruth label is for a real image or a fake image

        -------
        Returns
        -------
        tensor : A label tensor filled with groundtruth label with the same size as that of input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        ---------
        Arguments
        ---------
        prediction : tensor
            prediction from a discriminator
        target_is_real : bool
            whether the groundtruth label is for a real image or a fake image

        -------
        Returns
        -------
        loss : the computed loss
        """
        if self.loss_mode == "vanilla":
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        else:
            loss = 0
        return loss
