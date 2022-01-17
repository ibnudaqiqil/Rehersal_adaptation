
from rich import print
from rich.console import Console


import logging

import os
import warnings
import argparse


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


from torch.utils.data import DataLoader



import numpy as np
from fungsi import *
from models import *

console = Console()


pl.utilities.distributed.log.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logger = TensorBoardLogger("runs", name="MNIST_GAN")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

console.log("Loading MNIST dataset...")
scenario, scenario_test = create_MNIST_scenario(mnist_path="./store/dataset")

#parse arguments
parser = argparse.ArgumentParser()

parser.add_argument( "--gan_epochs",  type=int,  default=100, help="number of epochs to train generator",)
parser.add_argument( "--epochs",  type=int,  default=1, help="number of epochs to train generator",)
parser.add_argument( "--batch_size",   type=int,  default=BATCH_SIZE, help="size of the batches",)
parser.add_argument( "--model",type=str, default="GAN", help="model Generator")

args = parser.parse_args()

console.log("Creating model...")
for task_id, train_taskset in enumerate(scenario):
    
   
    jumlah_kelas = (task_id + 1)*2
    console.log(f"[red]Task {task_id} jumlah kelas = {jumlah_kelas}")
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

            train_taskset.add_samples(
                mem_x.numpy(), mem_y.detach().numpy(), None)
    
    # Train the model
    classifier = LitMNIST(num_classes=jumlah_kelas)
    trainer_classifier = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=args.epochs,
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
    pseudo_generator = GENERATOR[args.model](jumlah_kelas)
    callbacks = [TensorboardGenerativeModelImageSampler()]
    trainer = pl.Trainer(max_epochs=args.gan_epochs, gpus=AVAIL_GPUS,
                         progress_bar_refresh_rate=50, logger=logger, 
                         
                         )
    trainer.fit(pseudo_generator, train_loader)

        #print(hasil)
    
