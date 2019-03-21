import torch
import torch.nn as nn
from torch.nn import init
import numpy as np



##############################################################################
# Classes
##############################################################################
class MaskLoss(nn.Module):
    """Define GAN confidence map losses

    The MaskLoss class finds the cosine similarity between the generated image
    and the ground truth and uses that for the loss computation
    """

    def __init__(self, loss_type='cosine'):
        """ Initialize the MaskLoss class.

        Parameters:
            None

        """
        super(MaskLoss, self).__init__()
        self.loss_type = loss_type
        self.criterion_L1_loss = nn.L1Loss()
        if loss_type == 'cosine':
          self.cosine_similarity = nn.CosineSimilarity(dim=1)
        elif loss_type == 'angerr':
          print('Mask will be trained using angular error')
        else:
          assert ('Incorrect mask loss type. Possible options are Cosine similarity (cosine) and Angular error (angerr)')

    def rgb2lin(self, input):
        """Linearize gamma-corected RGB image
        For more details refer: https://en.wikipedia.org/wiki/SRGB

        Parameters:
          input (Tensor) - - typcially the input to the generator or the generated output

        Returns:
          Linearized gamma-corrected image
        """
        input = input / 255
        mask = (input <= 0.04045).float()
        output = mask * (input / 12.92) + \
                  (1 - mask) * torch.pow((input + 0.055) / 1.055, 2.4)
        output = torch.round(output * 255)

        return output

    def normalize(self, input):
        """Compute the L2 norm of an image

        Parameters:
          input (Tensor) - - typically an image

        Returns:
            The L2 normalized image
        """
        input = input + 1e-8
        squared = torch.pow(input, 2)
        norm = torch.sum(squared, dim=1).sqrt()
        out = input / norm

        return out

    def get_illumination(self, input, output):
        """Compute the normalized illumination for the given images

        Parameters:
          input (Tensor) - - typically the input to the generator
          output (Tensor) - - typically the output from the generator or the ground truth

        Returns:
            The illumination color
        """
        input = self.rgb2lin(input) + 1e-8
        output = self.rgb2lin(output) + 1e-8
        light = output / input
        illumination = self.normalize(light)

        return illumination

    def get_inverse_illumination(self, input, output):
        """Compute the normalized inverse illumination for the given images

        Parameters:
          input (Tensor) - - typically the input to the generator
          output (Tensor) - - typically the output from the generator or the ground truth

        Returns:
            The inverse of illumination color
        """
        input = self.rgb2lin(input) + 1e-8
        output = self.rgb2lin(output) + 1e-8
        light = input / output
        illumination = self.normalize(light)

        return illumination

    def __call__(self, real_A, real_B, fake_B, fake_B_mask):
        """Calculate loss given Discriminator's output and grount truth along with the generated
        confidence map or heat map or mask

        Parameters:
            real_A (tensor) - - tpyically the input image to the generator
            real_B (tensor) - - tpyically the ground truth
            fake_B (tensor) - - tpyically the prediction output from the generator
            fake_B_mask (tensor) - - tpyically the prediction mask from the generator

        Returns:
            the calculated loss.
        """
        if self.loss_type == 'cosine':
          similarity = self.cosine_similarity(real_B, fake_B).unsqueeze_(0)
          loss = torch.mean(1 - similarity)
        elif self.loss_type == 'angerr':
          with torch.no_grad():
            light_target = self.get_illumination(real_A, real_B)
            light_pred = self.get_inverse_illumination(real_A, fake_B)
            e = torch.sum(light_pred * light_target, dim=1)
            ang_err = torch.acos(torch.tanh(e))
            confidence = (ang_err / np.pi).unsqueeze(0)
            
          local_loss = self.criterion_L1_loss(fake_B_mask, confidence)
          loss = torch.mean(local_loss)
        else:
          assert("Incorrect loss type for mask")

        return loss


##############################################################################
# End of Classes
##############################################################################


