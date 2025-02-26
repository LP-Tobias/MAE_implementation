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
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

# Pretraining parameters. Epochs here.
EPOCHS = 100

# Augmentation parameters
IMAGE_SIZE = 32
PATCH_SIZE = 2
MASK_PROPORTION = 0.75

# Encoder and Decoder parameters
LAYER_NORM_EPS = 1e-6


train_dataloader, test_dataloader, train_set, test_set = prepare_data_cifar(DATA_DIR, INPUT_SHAPE, IMAGE_SIZE, BATCH_SIZE)


def pre_train(experiment_name, mask_ratio=0.75, decoder_depth=4):
    # for now the input only takes mask_ratio and decoder_depth.
    # for more experiments coming remember to also chang the name for model path, history name and so on.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT(decoder_layer=decoder_depth, mask_ratio=mask_ratio)
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE * BATCH_SIZE / 256,
                              betas=(0.9, 0.95),
                              weight_decay=WEIGHT_DECAY)

    total_steps = int((len(train_set) / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)

    scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

    model_path_pre = './model'
    if not os.path.exists(model_path_pre):
        os.makedirs(model_path_pre)
    model_name = f'large_lr_maskratio_{mask_ratio}_dec_depth_{decoder_depth}.pt'
    model_path = os.path.join(model_path_pre, model_name)

    image_path = experiment_name + '/images'

    step_count = 0
    optim.zero_grad()

    history = {
        'experiment': experiment_name,
        'loss': []
    }

    for e in range(EPOCHS):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / MASK_PROPORTION
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        history['loss'].append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_img = torch.stack([test_set[i][0] for i in range(8)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            image = tensor_to_image((img + 1) / 2)
            image_buffer = io.BytesIO()
            image.save(image_buffer, format='JPEG')
            image_buffer.seek(0)
            upload_blob_from_memory(image_buffer, image_path + f'epoch_{e}.jpg', 'image/jpeg')
            # this saves the image to the bucket, inside a folder.

        ''' save '''
        torch.save(model, model_path)

    history_json = json.dumps(history)
    save_history_to_gcs(history_json, experiment_name)


if __name__ == '__main__':
    # here is where I do the pretraining.

    # mask_ratios = [0.3, 0.5, 0.75, 0.85]
    # decoder_depths = [2, 6, 8]
    mask_ratios = [0.75]
    #
    for mask_ratio in mask_ratios:
        experiment_name = f'large_lr_e_{EPOCHS}_pretrain_mask_ratio_{mask_ratio}_decoder_depth_4'
        pre_train(experiment_name, mask_ratio)
        print(f'Experiment {experiment_name} is done!')
        print('-----------------------------------------------')

    # for decoder_depth in decoder_depths:
    #     experiment_name = f'pretrain_mask_ratio_0.75_decoder_depth_{decoder_depth}'
    #     pre_train(experiment_name, 0.75, decoder_depth)
    #     print(f'Experiment {experiment_name} is done!')
    #     print('-----------------------------------------------')




