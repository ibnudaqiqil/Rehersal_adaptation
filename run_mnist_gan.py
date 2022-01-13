
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
from models.MNIST import LitMNIST
from models.Gan import Generator,GAN


trfm = transforms.Compose([transforms.ToTensor(),
                           transforms.ToPILImage(),
                           transforms.Pad(2),
                           transforms.ToTensor(), ])
dataset = MNIST("./store/dataset", download=True, train=True)
                                             

test_dataset = MNIST("./store/dataset", download=True, train=False)
pl.utilities.distributed.log.setLevel(logging.ERROR)

scenario = ClassIncremental(
    dataset,
    increment=2,
    initial_increment=2,
    class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 
)


scenario_test = ClassIncremental(
    test_dataset,
    increment=2,
    initial_increment=2,
    class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

memory = rehearsal.RehearsalMemory(
    memory_size=2000,
    herding_method="random"
)
logger = TensorBoardLogger("runs", name="MNIST_GAN")
z_dim = [1, 4, 4]
x_dim = [1, 28, 28]
for task_id, train_taskset in enumerate(scenario):
    
    print(Fore.RED + f'TRAINING TASK: {task_id}  Class{train_taskset.get_classes()}')
    print(Style.RESET_ALL)
    #prepare datatrain set
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_taskset, batch_size=BATCH_SIZE, shuffle=False)
    #prepare psudodataset
    #if task_id > 0:
    #    mem_x, mem_y, mem_t = memory.get()
    #    train_taskset.add_samples(mem_x, mem_y, mem_t)
    
    # Train the model
    classifier = LitMNIST()
    trainer_classifier = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=1,
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
            print(Fore.GREEN + "Test test_id", test_id," Class=", scenario_test[test_id].get_classes())
            print(Style.RESET_ALL)
            test_loader = DataLoader(
                scenario_test[test_id], batch_size=32, shuffle=False)
            hasil = trainer_classifier.test(classifier, test_loader,verbose=False)
            t.append((test_id, hasil[0]['Test_acc']))
        #test_taskset = concat(t)
        print(t)
    else:
        test_taskset = scenario_test[task_id]
        test_loader = DataLoader(test_taskset, batch_size=32, shuffle=False)
        hasil = trainer_classifier.test(classifier, test_loader,verbose=False)
        print(hasil[0]['Test_acc'])

    model = GAN((1, 28, 28))
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=50, logger=logger,
                      progress_bar_refresh_rate=20)
    trainer.fit(model, train_loader)

        #print(hasil)
    
