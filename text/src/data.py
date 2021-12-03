import math
import random

from torch.utils.data import DataLoader


class BucketLoader(object):
    """DataLoader for sequence bucketing.

    Parameters
    ----------
    bucket_size : int
        How many batches to be sorted together.

    See also
    --------
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        bucket_size=100,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        self.data_size = len(dataset)
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.bucket_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size * bucket_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.nop_collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )

    @staticmethod
    def nop_collate_fn(batch):
        return batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.data_size // self.batch_size
        return math.ceil(self.data_size / self.batch_size)

    def __iter__(self):
        for large_batch in self.bucket_loader:
            large_batch = sorted(large_batch, key=lambda x: len(x[1]["input_ids"]))

            small_batches = []
            for start_idx in range(0, len(large_batch), self.batch_size):
                end_idx = min(len(large_batch), start_idx + self.batch_size)
                small_batch = large_batch[start_idx:end_idx]
                if end_idx - start_idx == self.batch_size or not self.drop_last:
                    small_batches.append(self.collate_fn(small_batch))
            random.shuffle(small_batches)

            for small_batch in small_batches:
                yield small_batch
