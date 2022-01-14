from models.MNIST import LitMNIST
from models.Cgan import CGAN
from models.CCgan import CCGAN
from rich import print
from rich.console import Console
from models.Wgan import WGAN
from pytorch_lightning.loggers import TensorBoardLogger
from colorama import Fore, Back, Style
import logging
from continuum import rehearsal
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

from torch.utils.data import DataLoader

from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import split_train_val, concat

import numpy as np

console = Console()
console.log("Loading MNIST dataset...")
trfm = [transforms.ToTensor(),
         #transforms.Normalize(mean=[0.5], std=[0.5]),
         #transforms.Lambda(lambda x: x.view(-1, 784)),
        # transforms.Lambda(lambda x: torch.squeeze(x))
         ]
                            
dataset = MNIST("./store/dataset", download=True, train=True)
                                             

test_dataset = MNIST("./store/dataset", download=True, train=False)
pl.utilities.distributed.log.setLevel(logging.ERROR)

console.log("Splitting dataset and create scenario...")
scenario = ClassIncremental(
    dataset,
    increment=2,
    initial_increment=2,
    class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    transformations=trfm,
 
)
scenario_test = ClassIncremental(
    test_dataset,
    increment=2,
    initial_increment=2,
    class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    transformations=trfm
)
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


logger = TensorBoardLogger("runs", name="MNIST_GAN")

console.log("Creating model...")
for task_id, train_taskset in enumerate(scenario):
    
    console.log(f"[red]Task {task_id}")
    jumlah_kelas = (task_id + 1)*2
    #prepare datatrain set
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_taskset, batch_size=BATCH_SIZE, shuffle=False)
    #prepare psudodataset
    if task_id > 0:
        console.log("Augmenting psudo data")
        for ps_id in range(0, task_id):
            jumlah_sample= 5000
            label_sample = ps_id
            Z = torch.randn(jumlah_sample, 100)
            mem_y = torch.full([jumlah_sample], label_sample)
            mem_x = pseudo_generator.generator(Z, mem_y)
            mem_x = mem_x.detach()
            mem_x = mem_x.view(-1,  28, 28)
        #print(mem_x.shape)
    #    mem_x, mem_y, mem_t = memory.get()
            train_taskset.add_samples(
                mem_x.numpy(), mem_y.detach().numpy(), None)
    
    # Train the model
    classifier = LitMNIST(num_classes=jumlah_kelas)
    trainer_classifier = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10,
        logger=logger,
        #verbose=True,
        #weights_summary=None,
        #progress_bar_refresh_rate=0,
        enable_progress_bar=True
    )
    trainer_classifier.fit(classifier, train_loader, val_loader)


    if task_id > 0:
        t=[]
        for test_id in range(0, task_id+1):
            #print(Fore.GREEN + "Test test_id", test_id," Class=", scenario_test[test_id].get_classes())
            #print(Style.RESET_ALL)
            test_loader = DataLoader(
                scenario_test[test_id], batch_size=32, shuffle=False)
            hasil = trainer_classifier.test(classifier, test_loader,verbose=False)
            t.append((test_id, hasil[0]['Test_acc']))
        #test_taskset = concat(t)
        console.log(f"accuracy : {t}")
    else:
        test_taskset = scenario_test[task_id]
        test_loader = DataLoader(test_taskset, batch_size=32, shuffle=False)
        hasil = trainer_classifier.test(classifier, test_loader,verbose=False)
        #print(hasil[0]['Test_acc'])
        console.log(f"accuracy : {hasil[0]['Test_acc']}")

    #generated_imgs = np.transpose(generated_imgs, (0, 2, 3, 1))
    #generated_imgs = generated_imgs.reshape(100, 28, 28)

    # Data preparation (Load your own data or example MNIST)
    console.log("Training Generator")
    pseudo_generator = CCGAN(num_classes=jumlah_kelas)

    trainer = pl.Trainer(max_epochs=100, gpus=AVAIL_GPUS,
                         progress_bar_refresh_rate=50, logger=logger,)
    trainer.fit(pseudo_generator, train_loader)

        #print(hasil)
    
