# %%
import importlib.util
import os
import sys
import tempfile
import urllib.request

if (
    importlib.util.find_spec("arena_lib") is None
):  # Colab, or a notebook outside a set-up clone of the repo
    url = "https://raw.githubusercontent.com/ARENA-education/ARENA_materials/main/arena_lib/__init__.py"
    code = urllib.request.urlopen(
        url
    ).read()  # raises on a 404 or no network, before anything is written
    fetched = os.path.join(
        tempfile.gettempdir(), "arena_lib_fetch"
    )  # not next to the notebook, so never in a clone
    os.makedirs(os.path.join(fetched, "arena_lib"), exist_ok=True)
    open(os.path.join(fetched, "arena_lib", "__init__.py"), "wb").write(code)
    sys.path.append(fetched)  # last, so it never shadows an arena_lib that's set up properly
# %%
import arena_lib

exercises_dir, section_dir = arena_lib.setup("0.2")
# %%
import json
from collections import namedtuple
from dataclasses import dataclass

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm


import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from arena_lib.plot import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))


# %%
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(t.tensor(0), x)


tests.test_relu(ReLU)


# %%


#
# Input: receives activations from the previous layer (in_features).
# Weights: It creates a matrix of weights connecting every input feature to every output feature.
# Biases: It optionally creates one bias per output feature.
# Output: It calculates the weighted sums and adds the biases, producing out_features new activations.
#
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        scalingFactor = 1 / np.sqrt(in_features)

        # here t.rand generates random matrix with values between 0 and 1
        # of dimension out_features by in features. The -1 makes it go
        # from 0, 1 to -1, 1

        # each row of self.weight represents all the weights of one output neuron
        # nn.parameter tells pytorch these are values you can change during training
        self.weight = nn.Parameter(scalingFactor * (2 * t.rand(out_features, in_features) - 1))

        if bias:
            # each row of self.bias contains the bias of one output neuron
            self.bias = nn.Parameter(scalingFactor * (2 * t.rand(out_features) - 1))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


# %%
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28 * 28, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(
        exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()
# %%
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)

model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        # entropy measures uncertainty
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}", refresh=False)

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)


# %%
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains & returns the model, using training parameters from the `args` object. Returns the loss
    list, and the model.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    amount_correctly_guessed = 0
    accuracy_list = []

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)
        pbarTest = tqdm(mnist_testloader)

        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}", refresh=False)

            amount_correctly_guessed = 0

        for imgs, labels in pbarTest:
            imgs, labels = imgs.to(device), labels.to(device)

            with t.inference_mode():
                output = model(imgs)
                prediction = output.argmax(dim=1)
                correct = prediction == labels
                amount_correctly_guessed += correct.sum().item()

        accuracy = amount_correctly_guessed / len(mnist_testset)
        accuracy_list.append(accuracy)

    return loss_list, accuracy_list, model


args = SimpleMLPTrainingArgs()
loss_list, accuracy_list, model = train(args)

line(
    y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
)


# %%
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.
        Name your weight field `self.weight` for compatibility with the PyTorch version.
        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # YOUR CODE HERE - define & initialize `self.weight`
        kernel_width = kernel_height = kernel_size

        scalingFactor = 1 / np.sqrt(in_channels * kernel_width * kernel_height)

        # a filter is a small collection of learnable weights
        # out_channels: How many filters we have.
        # in_channels: How many input channels each filter processes.
        # a channel is like the number of ways we separate an image
        # for example for RGB we have 3 channels 1 per color
        # kernel_height: Height of each individual kernel.
        # kernel_width: Width of each individual kernel.
        self.weight = nn.parameter(
            scalingFactor * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
        )

        # so kernel is the 2D grid of weights
        # the filter is the collection of kernels for each input channel grouped together

        # we can imagine an RGB image with a 3x3 kernel
        # we thus have 3 channels
        # and then we have x many filters where x represents the amount of features a CNN can learn
        # this is semi equivalent to the amount of neurons per layer
        # Each kernel produces an entire output matrix. Then you add those matrices element by element
        # to obtain the output feature map for that filter its just a summation

        # in a cnn the weights are teh numbers within the kernels

        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # a matrix sliding over the output matrix gotten from convolution that just takes the maximum
        # value it encounters

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join(
            [f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]]
        )


# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        # nm_features = channels in the image
        self.num_features = num_features

        # epsilon
        self.eps = eps

        # for the running mean each new batch nudges the running mean and running
        # var by this much %
        self.momentum = momentum

        # the use of a running mean and running deviation is so that when we use
        # a new image that it hasnt seen that image can be normalized using
        # those values

        # the weight and bias here are 2 learnable parameters that the
        # neural net can tweak in case a higher or lower standard deviation
        # benefits us more and the bias is the same but for the median
        # also we initialize the weight (adjustable deivation as 1) because thats
        # what the standard deviation in a perfect normalization would be
        # and we initialize the bias (adjustable mean as 0) for essentially the
        # same reason
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        In training mode, normalize using the biased variance `x.var(..., correction=0)`, but update
        `running_var` using the unbiased variance `x.var(..., correction=1)`.
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        # the process is:
        # 1. Take the activations from the previous layer.
        # 2. Calculate their mean.
        # 3. Subtract the mean from every activation, centering them around zero.
        # 4. Divide by their standard deviation, giving them a standard deviation of 1

        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), correction=0)
            with t.no_grad():
                var_unbiased = x.var(dim=(0, 2, 3), correction=1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean

                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var_unbiased

                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Rearranging these so they can be broadcasted
        reshape = lambda x: einops.rearrange(x, "channels -> 1 channels 1 1")

        # Normalize, then apply affine transformation from self.weight & self.bias
        x_normed = (x - reshape(mean)) / (reshape(var) + self.eps).sqrt()
        x_affine = x_normed * reshape(self.weight) + reshape(self.bias)
        return x_affine

    def extra_repr(self) -> str:
        return ", ".join(
            [f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]]
        )


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)
tests.test_batchnorm2d_running_stats_detached(BatchNorm2d)


# %%
class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        raise NotImplementedError()


tests.test_averagepool(AveragePool)
