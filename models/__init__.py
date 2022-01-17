from .base import *
from .vanilla_vae import *
from .Gan import GAN
from .Cgan import CGAN

from .MNIST import LitMNIST

# Aliases
#VAE = VanillaVAE
GaussianVAE = VanillaVAE
#CVAE = ConditionalVAE
#GumbelVAE = CategoricalVAE

GENERATOR = {
             'GAN':GAN,
             'LitMNIST':LitMNIST,
             'CGAN':CGAN,}
