import os
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split

from load_data import *
from model_mae_timm import *
from utils import *
import io
import json

from torch.optim.lr_scheduler import LRScheduler

from google.cloud import storage
from PIL import Image

setup_seed()

# These are some default parameters.
# For the encoder/decoder setting refer to model_mae_timm.py, I initialize the pretraining model using MAE_ViT method.
# OBS!!!: Don't forget to look at the utils.py and change the name of the bucket to your own bucket name.


# DATA
BATCH_SIZE = 512
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
DATA_DIR = './data'

# Optimizer parameters
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.05

# Fewer epochs for fine-tuning and linear probing
EPOCHS = 80

# Augmentation parameters
IMAGE_SIZE = 32
PATCH_SIZE = 2
MASK_PROPORTION = 0.75

# Encoder and Decoder parameters
LAYER_NORM_EPS = 1e-6

train_dataloader, test_dataloader, train_set, test_set = prepare_data_cifar(DATA_DIR, INPUT_SHAPE, IMAGE_SIZE, BATCH_SIZE)


def classification(experiment_name, mask_ratio=0.75, decoder_depth=4):
    # for now the input only takes mask_ratio and decoder_depth.
    # for more experiments coming remember to also chang the name for model path, history name and so on.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path_pre = './model'
    if not os.path.exists(model_path_pre):
        os.makedirs(model_path_pre)
    model_name = f'mae_pretrain_maskratio_{mask_ratio}_dec_depth_{decoder_depth}.pt'
    model_path = os.path.join(model_path_pre, model_name)

    if experiment_name == 'linear_probe' or experiment_name == 'fine_tune':
        read_model = torch.load(model_path)
        encoder = read_model.module.encoder
    elif experiment_name == 'from_scratch':
        init_model = MAE_ViT(decoder_layer=decoder_depth, mask_ratio=mask_ratio)
        encoder = init_model.encoder

    model = ViT_Classifier(encoder, num_classes=NUM_CLASSES).to(device)

    print('model_loaded')

    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE * BATCH_SIZE / 256,
                              betas=(0.9, 0.999),
                              weight_decay=WEIGHT_DECAY)

    total_steps = int((len(train_set) / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)

    scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()

    history = {
        'experiment': experiment_name,
        'model': model_name,
        'train_loss': [],
        'val_loss': [],
        'acc_train': [],
        'acc_val': [],
        'best_val_acc': 0
    }

    for e in range(EPOCHS):
        if experiment_name == 'linear_probe':
            model.module.patchify.eval()
            model.module.transformer.eval()
            model.module.layer_norm.eval()
            model.module.head.train()
        else:
            model.train()

        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
        history['train_loss'].append(avg_train_loss)
        history['acc_train'].append(avg_train_acc)

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(test_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
            history['val_loss'].append(avg_val_loss)
            history['acc_val'].append(avg_val_acc)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            history['best_val_acc'] = best_val_acc

    history_json = json.dumps(history)
    save_history_to_gcs(history_json, experiment_name + '_' + model_name)
    # print(history)
    print(f'Experiment {experiment_name}_{model_name} is done!')
    print('-----------------------------------------------')


if __name__ == '__main__':
    experiment_name = 'from_scratch'
    classification(experiment_name)

