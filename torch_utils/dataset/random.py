import torch


class random():
    """
    random functions in pytorch
    """
    @staticmethod
    def randint(low, high):
        return int(torch.randint(low, high + 1, (1,)).numpy())

    @staticmethod
    def random():
        return float(torch.rand(1)[0].numpy())

    @staticmethod
    def uniform(low, high):
        low = torch.FloatTensor([low])
        high = torch.FloatTensor([high])
        m = torch.distributions.uniform.Uniform(low, high)
        sample = float(m.sample()[0].numpy())
        return sample

    @staticmethod
    def choice(samples):
        idx = torch.randint(len(samples), (1,))
        return samples[idx]

    @staticmethod
    def choices(samples, k=1, replacement=True):
        if replacement:
            idxs = torch.randint(len(samples), (k,))
        else:
            idxs = torch.randperm(len(samples))[:k]
        result = []
        for idx in idxs:
            result.append(samples[idx])
        return result
