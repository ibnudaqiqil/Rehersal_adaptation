import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl


class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self, n_classes=10, image_w=28, image_h=28,latent_dim=100):
    super().__init__()
    self.n_classes = n_classes
    self.image_w = image_w
    self.image_h = image_h

    self.embedding = nn.Embedding(n_classes, n_classes)
    #self.embedding = nn.Embedding(10, 10)
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_dim+n_classes, out_features=256),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=image_w*image_h),
                                nn.Tanh())

  def forward(self, z, y):
    # pass the labels into a embedding layer
    labels_embedding = self.embedding(y)
    # concat the embedded labels and the noise tensor
    # x is a tensor of size (batch_size, 110)
    x = torch.cat([z, labels_embedding], dim=-1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, n_classes=10,  image_w=28, image_h=28, latent_dim=100):
    super().__init__()
    self.n_classes = n_classes
    self.embedding = nn.Embedding(self.n_classes, self.n_classes)
    self.layer1 = nn.Sequential(nn.Linear(in_features=image_w*image_h+n_classes, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())

  def forward(self, x, y):
    # pass the labels into a embedding layer
    labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    x = x.view(-1, 28*28)
  
    x = torch.cat([x, labels_embedding], dim=-1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x


class CGAN(pl.LightningModule):

  def __init__(self, n_classes=10, image_w=28, image_h=28, latent_dim=100):
    super().__init__()
    self.n_classes = n_classes
    self.image_w = image_w
    self.image_h = image_h
    self.latent_dimx = latent_dim
    self.generator = Generator(n_classes=self.n_classes, image_w=self.image_w, image_h=self.image_h,latent_dim=self.latent_dimx)
    self.discriminator = Discriminator(n_classes=self.n_classes, latent_dim=self.latent_dimx)
    self.img_dim = (1, image_w, image_h)

  def forward(self, z, y):
    """
    Generates an image using the generator
    given input noise z and labels y
    """
    return self.generator(z, y)

  def generator_step(self, x):
    """
    Training step for generator
    1. Sample random noise and labels
    2. Pass noise and labels to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss
    """

    # Sample random noise and labels
    z = torch.randn(x.shape[0], self.latent_dimx, device=self.device)
    y = torch.randint(0, self.n_classes, size=(x.shape[0],), device=self.device)

    # Generate images
    generated_imgs = self(z, y)

    # Classify generated image using the discriminator
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = nn.BCELoss()(d_output, torch.ones(x.shape[0], device=self.device))
    self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
    return g_loss

  def discriminator_step(self, x, y):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """

    # Real images
    d_output = torch.squeeze(self.discriminator(x, y))
    loss_real = nn.BCELoss()(d_output, torch.ones(x.shape[0], device=self.device))

    # Fake images
    z = torch.randn(x.shape[0], self.latent_dimx, device=self.device)
    y = torch.randint(0, self.n_classes, size=(x.shape[0],), device=self.device)

    generated_imgs = self(z, y)
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))
    loss_fake = nn.BCELoss()(d_output, torch.zeros(x.shape[0], device=self.device))
    self.log("d_loss", loss_real + loss_fake, on_epoch=True, prog_bar=True)
    return loss_real + loss_fake

  def training_step(self, batch, batch_idx, optimizer_idx):
    X, y, _ = batch

    # train generator
    if optimizer_idx == 0:
      loss = self.generator_step(X)

    # train discriminator
    if optimizer_idx == 1:
      loss = self.discriminator_step(X, y)

    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []

  def on_epoch_end(self):
         # generate images
        with torch.no_grad():
            z = torch.randn(10, self.latent_dimx, device=self.device)
            y = torch.randint(0, self.n_classes, size=(10,), device=self.device)
            images = self.generator(z,y)
           


        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=8,
            #padding=self.padding,
            #normalize=self.normalize,
            #range=self.norm_range,
            #scale_each=self.scale_each,
            #pad_value=self.pad_value,
        )
        str_title = f"{self.__class__.__name__}_images"
        self.logger.experiment.add_image(
            str_title, grid, self.current_epoch)
