import torch
import torch.nn as nn
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim


class WarmUpCosine(LRScheduler):
    def __init__(self, optimizer: optim,
                 total_steps: int,
                 warmup_steps: int,
                 learning_rate_base: float,
                 warmup_learning_rate: float,
                 last_epoch: int = -1):
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = np.pi
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        cos_annealed_lr = 0.5 * self.learning_rate_base * (1 + np.cos(self.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))

        if step < self.warmup_steps:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_learning_rate
            return [warmup_rate for _ in self.base_lrs]
        else:
            return [cos_annealed_lr for _ in self.base_lrs]


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def block_indexes(size: int):
    w = int(np.sqrt(size))
    img_grid = np.arange(size).reshape(w, w)
    forward_indexes = select_random_block(img_grid)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def select_random_block(grid, block_size=128):
    total_size = grid.size
    rows, cols = grid.shape

    block_rows = 8 if np.random.rand() < 0.5 else 16
    block_cols = 16 if block_rows == 8 else 8

    start_row = np.random.randint(rows - block_rows + 1)
    start_col = np.random.randint(cols - block_cols + 1)

    flat_indices = np.arange(total_size)
    block_indices = flat_indices.reshape(rows, cols)

    block_mask = np.zeros_like(grid, dtype=bool)
    block_mask[start_row:start_row + block_rows, start_col:start_col + block_cols] = True
    in_block = block_indices[block_mask]
    outside_block = block_indices[~block_mask]

    new_indices = np.concatenate([outside_block, in_block])
    return new_indices


def grid_indexes(size: int):
    w = int(np.sqrt(size))
    img_grid = np.arange(size).reshape(w, w)
    forward_indexes, backward_indexes = skip_rows_cols(img_grid)
    return forward_indexes, backward_indexes


def skip_rows_cols(grid):
    reduced_grid = grid[::2, ::2]
    reduced_grid_flat = reduced_grid.flatten()
    grid_flat = grid.flatten()

    mask = np.isin(grid_flat, reduced_grid_flat)

    forward_indexes = np.concatenate((reduced_grid_flat, grid_flat[~mask]))
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio=0.75, emb_dim=192, with_mask_token=False, mask_strategy='random'):
        super().__init__()
        self.ratio = ratio
        self.with_mask_token = with_mask_token
        self.mask_strategy = mask_strategy
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        # total patches, bach size, embedding dimension
        remain_T = int(T * (1 - self.ratio))

        if self.mask_strategy == 'random':
            indexes = [random_indexes(T) for _ in range(B)]
            forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
            backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

            patches = take_indexes(patches, forward_indexes)
            patches = patches[:remain_T]

            if self.with_mask_token:
                patches = torch.cat([patches, self.mask_token.expand(forward_indexes.shape[0] - patches.shape[0], patches.shape[1], -1)], dim=0)
        elif self.mask_strategy == 'Block':
            indexes = [block_indexes(T) for _ in range(B)]
            forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
            backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

            patches = take_indexes(patches, forward_indexes)
            patches = patches[:128] # fixed experiment of 50%

        elif self.mask_strategy == 'Grid':
            indexes = [grid_indexes(T) for _ in range(B)]
            forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
            backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

            patches = take_indexes(patches, forward_indexes)
            patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, num_layer=12, num_head=3, mask_ratio=0.75, with_mask_token=True, mask_strategy='random'):
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio, emb_dim, with_mask_token, mask_strategy)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        # if go as original paper
        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embedding, std=.02)

        # or use xiavier initialization
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, img):
        # print('img shape', img.shape)
        patches = self.patchify(img)
        # print('patches shape', patches.shape)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, num_layer=6, num_head=3):
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                                   p1=patch_size, p2=patch_size, h=image_size // patch_size)
        self.init_weight()

    def init_weight(self):
        # if go as original paper
        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embedding, std=.02)

        # or use xiavier initialization
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, features, backward_indexes):
        if features.shape[0] == 257:
            T = 65
        else:
            T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove cls token

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 with_mask_token=False,
                 mask_strategy='random'
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, with_mask_token, mask_strategy)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

