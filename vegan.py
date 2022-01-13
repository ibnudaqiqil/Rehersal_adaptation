from vegans.GAN import VanillaGAN
import vegans.utils as utils
import vegans.utils.loading as loading

# Data preparation (Load your own data or example MNIST)
loader = loading.MNISTLoader("./store/dataset2/")
X_train, _, X_test, _ = loader.load()
x_dim = X_train.shape[1:]  # [height, width, nr_channels]
z_dim = 64

# Define your own architectures here. You can use a Sequential model or an object
# inheriting from torch.nn.Module. Here, a default model for mnist is loaded.
generator = loader.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=None)
discriminator = loader.load_adversary(x_dim=x_dim, y_dim=None)

gan = VanillaGAN(
    generator=generator, adversary=discriminator,
    z_dim=z_dim, x_dim=x_dim, folder=None
)
gan.summary()  # optional, shows architecture

# Training
gan.fit(X_train, enable_tensorboard=False, epochs=100)

# Vizualise results
images, losses = gan.get_training_results()
utils.plot_images(images)
utils.plot_losses(losses)

# Sample new images, you can also pass a specific noise vector
samples = gan.generate(n=36)
utils.plot_images(samples)
