import math
import string
from typing import Callable, Iterable, List, Optional, Tuple

from einops.layers.torch import Rearrange, Reduce
from src.dataset.json_files import Json
from src.dataset.tokeniser import Tokeniser
from src.dataset.utils import dfs_unpack_json
import torch
import torch.nn as nn


def char_preprocessing(input: str) -> List[str]:
    """Returns a list of the characters in the string."""
    return list(input)


class JSONKeyValueResnet_EmbeddingEncoder(nn.Module):
    NAME = "json-resnet"

    def __init__(
        self,
        num_classes: int,
        block: Optional[nn.Module] = None,
        layers: Optional[List[int]] = None,
        vocab: Optional[Iterable[str]] = None,
        token_preprocessing: Callable[[str], List[str]] = char_preprocessing,
        embed_dim: int = 4,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        super().__init__()
        self.num_classes = num_classes

        self.block = block or Bottleneck
        self.layers = layers or [3, 4, 6, 3]
        self.vocab = vocab or list(string.printable)

        self.tokeniser = Tokeniser(
            self.vocab,
            token_preprocessing,
            pad_token,
            unk_token,
            bos_token,
            eos_token,
        )

        self.features = torch.nn.Sequential(
            nn.Embedding(len(self.tokeniser.vocab), embed_dim),
            Rearrange("batch seq_len channels -> batch channels seq_len"),
            nn.Conv1d(embed_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self.make_layer(64, 64, self.block, self.layers[0], stride=1),
            self.make_layer(
                64 * self.block.expansion, 128, self.block, self.layers[1], stride=2
            ),
            self.make_layer(
                128 * self.block.expansion, 256, self.block, self.layers[2], stride=2
            ),
            self.make_layer(
                256 * self.block.expansion, 512, self.block, self.layers[3], stride=2
            ),
        )

        self.reshape = torch.nn.Sequential(
            Reduce("batch channels seq -> batch channels", "mean"),
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(512 * self.block.expansion, self.num_classes),
        )

        # initialization
        for m in self.features.modules():
            if isinstance(m, nn.Conv1d):
                n = torch.numel(torch.Tensor(m.kernel_size)) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(
        self,
        inplanes: int,
        planes: int,
        block: nn.Module,
        n_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        # if output size won't match input, just adjust residual
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        return nn.Sequential(
            block(inplanes, planes, stride, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, n_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.reshape(out)
        out = self.classifier(out)
        return out

    def preprocessing(self, json_input: Json) -> torch.LongTensor:
        """
        1. flatten json into a list of key and values
        2. tokenise the complete key in keys
        3. break up the values into characters and tokenise individually
        4. return flattened sequence of ints for py.embedding in the model
        """
        flattened_json_input: List[Tuple[str, str]] = list(dfs_unpack_json(json_input))

        keys = (
            torch.LongTensor([self.tokeniser.convert_token_to_id(key)])
            for key, _ in flattened_json_input
        )

        values = (
            torch.LongTensor(self.tokeniser(value)) for _, value in flattened_json_input
        )

        output = torch.cat([torch.cat([k, v]) for k, v in zip(keys, values)]).flatten()

        return output


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    """3x3 convolution with padding

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    # Expansion is defined here so we can access before module is instantiated
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.0)) * groups

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.residual_block = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width),
            self.relu,
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width),
            self.relu,
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
