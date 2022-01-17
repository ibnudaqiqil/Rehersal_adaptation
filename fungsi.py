from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import split_train_val, concat
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def create_MNIST_scenario(mnist_path="./store/dataset", class_inc=2, intial_class=2):
    trfm = [transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5]),
            #transforms.Lambda(lambda x: x.view(-1, 784)),
            # transforms.Lambda(lambda x: torch.squeeze(x))
            ]

    dataset = MNIST(mnist_path, download=True, train=True)
    test_dataset = MNIST(mnist_path, download=True, train=False)

    scenario = ClassIncremental(
        dataset,
        increment=class_inc,
        initial_increment=intial_class,
        class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        transformations=trfm,
    )
    scenario_test = ClassIncremental(
        test_dataset,
        increment=class_inc,
        initial_increment=intial_class,
        class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        transformations=trfm
    )

    return scenario, scenario_test
