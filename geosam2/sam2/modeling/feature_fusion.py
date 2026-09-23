import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.fusion_high = nn.Conv2d(256*2, out_channels, kernel_size=3, padding=1)
        self.fusion_mid = nn.Conv2d(256*2, out_channels, kernel_size=3, padding=1)
        self.fusion_low = nn.Conv2d(256*2, out_channels, kernel_size=3, padding=1)

        for conv in [self.fusion_high, self.fusion_mid, self.fusion_low]:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, features1, features2,):
        """
        Args:
            features1: List from backbone_out['backbone_fpn']
            features2: List from pos_map_out['backbone_fpn']
        Returns:
            outputs: List of fused features at each scale
        """
        
        outputs = []
        for i, (feat1, feat2, ) in enumerate(zip(features1, features2, )):
            concat_feat=  torch.cat([feat1, feat2, ], dim=1)

            if i == 0:
                fused = self.fusion_high(concat_feat)
            elif i == 1:
                fused = self.fusion_mid(concat_feat)
            else:
                fused = self.fusion_low(concat_feat)

            fused = fused + feat1
            outputs.append(fused)
            
        return outputs