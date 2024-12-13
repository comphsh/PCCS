import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mmseg import module as md
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
    ):
        super().__init__()
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=True, attention_type=None)
        blocks = []
        for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels):
            blocks.append(  DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)  )

        self.blocks_list = nn.ModuleList(blocks)

    def forward(self, *features):  #传指针，因为会改变？？？
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        x = head
        x = self.blocks_list[0](x, skips[0])
        x = self.blocks_list[1](x, skips[1])
        x = self.blocks_list[2](x, skips[2])
        x = self.blocks_list[3](x, skips[3])
        x = self.blocks_list[4](x, None)
        return x