""" 
CUDA Prefetcher Loader
changed from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
"""

class PrefetchLoader:
    
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)

            if not first:
                yield input, target
            else:
                first = False
