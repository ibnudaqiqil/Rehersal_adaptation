import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl

#Discriminator


class Discriminator(nn.Module):
    def __init__(self, label_size):
        super(Discriminator, self).__init__()
        #label = [B]
        self.embedding = nn.Embedding(label_size, 4*28*28)
        #label after embed = [B, 4*28*28]
        self.label_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.image_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #input shape: [B, 1, 28, 28]
        self.pred = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=1, padding=0),
            nn.Sigmoid()

        )

    def forward(self, x, labels):
        labels = self.embedding(labels)
        labels = labels.view(labels.size(0), 1, 56, 56)  # [B, 1, 56, 56]
        labels = self.label_conv(labels)

        #x = [B, 1, 24, 24]
        x = self.image_conv(x)

        x = torch.cat([x, labels], 1)
        x = self.pred(x)
        return x

#generator


class Generator(nn.Module):
    def __init__(self, label_size=10, embedding_dim=10, latent_size=100):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(label_size, embedding_dim)
        #input image shape(B,C,H,W) = [B, latent_size + embedding_dim, 1, 1]
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_size + embedding_dim,
                               50, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(50),
            #result: [B, 50, 2, 2]

            nn.ConvTranspose2d(50, 30, kernel_size=(4, 4),
                               stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(30),
            #result: [B, 30, 4, 4]

            nn.ConvTranspose2d(30, 15, kernel_size=(4, 4),
                               stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(15),
            #result: [B, 15, 7, 7]

            nn.ConvTranspose2d(15, 8, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            #result: [B, 8, 14, 14]

            nn.ConvTranspose2d(8, 1, kernel_size=(4, 4), stride=2, padding=1),
            nn.Tanh()
            #result: [B, 1, 28, 28]
        )

    def forward(self, x, labels):
        #x = [B, latent_size]
        labels = self.embedding(labels)
        #labels after embedding = [B, embedding_dim]
        x = torch.cat([x, labels], 1)
        # = [B, latent_size + embedding_dim, 1, 1]
        x = x.view(x.size(0), -1, 1, 1)
        return self.conv(x)

'''
class Generator(nn.Module):
 

  def __init__(self, num_classes, img_shape=[28,28]):
    super().__init__()
    self.img_shape = img_shape
    self.latent_dim = 100
    self.init_size = img_shape[1] // 4
    self.embedding = nn.Embedding(num_classes, num_classes)
    self.layer1 = nn.Sequential(nn.Linear(in_features=self.latent_dim + num_classes, out_features=128 * self.init_size ** 2),
                                nn.LeakyReLU())
    self.conv_blocks = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
        nn.Tanh(),
    )

  def forward(self, z, y):
    # pass the labels into a embedding layer
    labels_embedding = self.embedding(y)
    # concat the embedded labels and the noise tensor
    # x is a tensor of size (batch_size, 110)
    x = torch.cat([z, labels_embedding], dim=-1)
    x = self.layer1(x)
    x = x.view(x.shape[0], 128, self.init_size, self.init_size)
    img = self.conv_blocks(x)
    return img


class Discriminator(nn.Module):


  def __init__(self, num_classes=10, img_shape=[28, 28]):
    super().__init__()
    self.embedding = nn.Embedding(num_classes, num_classes)
    self.layer1 = nn.Sequential(nn.Linear(in_features=28*28+num_classes, out_features=1024),
                                nn.LeakyReLU())

    def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

    self.model = nn.Sequential(
        *discriminator_block(img_shape[0], 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
    )

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
    x = torch.cat([x, labels_embedding], dim=-1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x
'''

class CCGAN(pl.LightningModule):

  def __init__(self, num_classes=10):
    super().__init__()
    self.num_classes = num_classes
    self.generator = Generator(label_size=self.num_classes)
    self.discriminator = Discriminator(label_size=self.num_classes)

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
    z = torch.randn(x.shape[0], 100, device=self.device)
    y = torch.randint(0, self.num_classes, size=(
        x.shape[0],), device=self.device)

    # Generate images
    generated_imgs = self(z, y)

    # Classify generated image using the discriminator
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = nn.BCELoss()(d_output,
                          torch.ones(x.shape[0], device=self.device))

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
    loss_real = nn.BCELoss()(d_output,
                             torch.ones(x.shape[0], device=self.device))

    # Fake images
    z = torch.randn(x.shape[0], 100, device=self.device)
    y = torch.randint(0, self.num_classes, size=(
        x.shape[0],), device=self.device)

    generated_imgs = self(z, y)
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))
    loss_fake = nn.BCELoss()(d_output,
                             torch.zeros(x.shape[0], device=self.device))

    return loss_real + loss_fake

  def training_step(self, batch, batch_idx, optimizer_idx):
    X, y,_ = batch

    # train generator
    #print(optimizer_idx)
    if optimizer_idx == 0:
      loss = self.generator_step(X)

    # train discriminator
    if optimizer_idx == 1:
      loss = self.discriminator_step(X, y)

    return loss

  def on_epoch_end(self):
    
    # log sampled images
    z = torch.randn(10, 100, device=self.device)
    y = torch.randint(0, self.num_classes, size=( 10,), device=self.device)

    generated_imgs = self(z, y)
    generated_imgs = generated_imgs.view(10, 1, 28, 28)
    #self.logger.experiment.add_image('generated_images', generated_imgs, 0)
   # print(d_output.shape)
    grid = torchvision.utils.make_grid(generated_imgs)
    self.logger.experiment.add_image(
        f'generated_images-{self.current_epoch}', grid, self.current_epoch)


  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []



