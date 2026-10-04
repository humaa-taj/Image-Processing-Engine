"""Balanced batch sampler: every batch contains exactly batch_size/4 images of each condition.

Random 1-in-4 sampling is only balanced *on average*; a batch of 32 could easily get 12 blur and
4 clean. The assignment asks for balanced batches for the classifier, and the Task 3 balance loss
assumes a balanced batch, so we force it here.
"""
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Yields lists of (image_index, condition) pairs for PetTrainDataset."""

    def __init__(self, n_images: int, batch_size: int, n_conditions: int = 4, seed: int = 0,
                 drop_last: bool = True):
        if batch_size % n_conditions:
            raise ValueError(f"batch_size {batch_size} must be divisible by {n_conditions}")
        self.n_images, self.batch_size, self.n_conditions = n_images, batch_size, n_conditions
        self.seed, self.drop_last, self.epoch = seed, drop_last, 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so each epoch gets a new shuffle (still reproducible)."""
        self.epoch = epoch

    def __len__(self):
        full, rest = divmod(self.n_images, self.batch_size)
        return full if self.drop_last or rest == 0 else full + 1

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.n_images, generator=g).tolist()
        per_class = self.batch_size // self.n_conditions
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            # Conditions [0,0,..,1,1,..,2,..,3,..] shuffled, so each image gets a random one.
            conds = torch.arange(self.n_conditions).repeat_interleave(per_class)
            conds = conds[torch.randperm(len(conds), generator=g)][:len(idx)].tolist()
            yield list(zip(idx, conds))
