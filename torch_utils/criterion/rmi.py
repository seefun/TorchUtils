# from https://github.com/RElbers/region-mutual-information-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 0.0005


class RMILoss(nn.Module):
    """
    PyTorch Module which calculates the Region Mutual Information loss (https://arxiv.org/abs/1910.12037).
    """

    def __init__(self,
                 with_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='max',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=EPSILON):
        """
        :param with_logits:
            If True, apply the sigmoid function to the prediction before calculating loss.
        :param radius:
            RMI radius.
        :param bce_weight:
            Weight of the binary cross entropy. Must be between 0 and 1.
        :param downsampling_method:
            Downsampling method used before calculating RMI. Must be one of ['avg', 'max', 'region-extraction'].
            If 'region-extraction', then downscaling is done during the region extraction phase.
            Meaning that the stride is the spacing between consecutive regions.
        :param stride:
            Stride used for downsampling.
        :param use_log_trace:
            Whether to calculate the log of the trace, instead of the log of the determinant. See equation (15).
        :param use_double_precision:
            Calculate the RMI using doubles in order to fix potential numerical issues.
        :param epsilon:
            Magnitude of the entries added to the diagonal of M in order to fix potential numerical issues.
        """
        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):
        # Calculate BCE if needed
        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        # Apply sigmoid to get probabilities. See final paragraph of section 4.
        if self.with_logits:
            input = torch.sigmoid(input)

        # Calculate RMI loss
        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def rmi_loss(self, input, target):
        """
        Calculates the RMI loss between the prediction and target.

        :return:
            RMI loss
        """

        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        # Get region vectors
        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        # Convert to doubles for better precision
        if self.use_double_precision:
            y = y.double()
            p = p.double()

        # Small diagonal matrix to fix numerical issues
        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        # Subtract mean
        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        # Covariances
        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        # Approximated posterior covariance matrix of Y given P
        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        # Lower bound of RMI
        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        # Normalize
        rmi = rmi / float(vector_size)

        # Sum over classes, mean over samples.
        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        """
        Downsamples and extracts square regions from x.
        Returns the flattened vectors of length radius*radius.
        """

        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):
        # Skip if stride is 1
        if self.stride == 1:
            return x

        # Skip if we pool during region extraction.
        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.linalg.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)
