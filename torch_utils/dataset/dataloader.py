""" 
CUDA Prefetcher Loader
changed from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
"""
import torch


class PrefetchLoader:

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        data, target = None, None
        for next_data, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_data = next_data.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)

            if not first:
                yield data, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
            target = next_target

        yield data, target
