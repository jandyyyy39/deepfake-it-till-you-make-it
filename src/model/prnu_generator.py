import torch
import torch.nn as nn

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

class PRNUGenerator(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project the 128-d latent vector into a small 8x8 spatial map
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        
        # Upsample iteratively from 8x8 up to 512x512
        self.upsample_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            UpsampleBlock(512, 256), # 8x8   -> 16x16
            UpsampleBlock(256, 128), # 16x16 -> 32x32
            UpsampleBlock(128, 64),  # 32x32 -> 64x64
            UpsampleBlock(64, 32),   # 64x64 -> 128x128
            UpsampleBlock(32, 16),   # 128x128 -> 256x256
            UpsampleBlock(16, 8)     # 256x256 -> 512x512
        )
        
        # Final Output Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(8, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # Flattened projection
        out = self.l1(z)
        # Reshape into a 2D feature map: (Batch_Size, Channels, Height, Width)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        # Upsample
        out = self.upsample_blocks(out)
        # Final convolution to get the PRNU residual
        img = self.final_conv(out)
        return img
    
class PRNUCritic(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        def critic_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters, affine=True))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            # 512x512 -> 256x256
            *critic_block(input_channels, 64, bn=False),
            # 256x256 -> 128x128
            *critic_block(64, 128),
            # 128x128 -> 64x64
            *critic_block(128, 256),
            # 64x64 -> 32x32
            *critic_block(256, 512),
            # 32x32 -> 16x16
            *critic_block(512, 1024),
            # 16x16 -> 8x8
            *critic_block(1024, 2048),
        )
        
        # Flatten and output a single continuous score
        self.final_layer = nn.Linear(131072, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.final_layer(out)
        return validity
