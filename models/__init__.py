from .base import *
from .vanilla_vae import *
from .Gan import GAN
from .Cgan import CGAN
from .CCgan import CCGAN

from .MNIST import *

# Aliases
#VAE = VanillaVAE
GaussianVAE = VanillaVAE
#CVAE = ConditionalVAE
#GumbelVAE = CategoricalVAE

GENERATOR = {
             'GAN':GAN,
             'LitMNIST':LitMNIST,
             "CNN_MNIST":CNN_MNIST,
             'CGAN':CGAN,
             'CCGAN':CCGAN}
