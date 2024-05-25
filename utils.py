from google.cloud import storage
from PIL import Image

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
import io


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    tensor = tensor.cpu().detach()  # Move tensor to CPU and detach from gradients
    # tensor = (tensor + 1) / 2  # Normalize if required, adjust based on your normalization
    tensor = tensor.clamp(0, 1)  # Clamp values to valid image range
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image


def upload_blob_from_memory(blob_data, destination_blob_name, content_type):
    """Uploads a file to the bucket from memory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket('experiment_results_23_5')
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(blob_data, content_type=content_type)

    print(f"Data uploaded to {destination_blob_name}.")


def save_history_to_gcs(history_json, destination_blob_name):
    client = storage.Client()
    bucket_name = 'experiment_results_23_5'
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(history_json, content_type='application/json')
    print("History object saved to GCS")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
