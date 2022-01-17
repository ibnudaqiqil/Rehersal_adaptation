from .base import *
from .vanilla_vae import *
from .Gan import GAN

from .MNIST import LitMNIST
from .Cgan import CGAN  
# Aliases
#VAE = VanillaVAE
GaussianVAE = VanillaVAE
#CVAE = ConditionalVAE
#GumbelVAE = CategoricalVAE

GENERATOR = {
             'GAN':GAN,
             'LitMNIST':LitMNIST,
             'CGAN':CGAN,}
