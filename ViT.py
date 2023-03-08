# https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch
import torch
from torch import nn
from vit_pytorch import ViT
from vit_pytorch import SimpleViT
from vit_pytorch.pit import PiT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small

# https://github.com/lucidrains/vit-pytorch

class Vit_neck(nn.Module):
    def __init__(self, batch_size, image_size, out_chanels=256, device="cuda"):
        super(Vit_neck, self).__init__()

        # self.v = SimpleViT(
        #     image_size = image_size,
        #     patch_size = batch_size,
        #     num_classes = out_chanels,
        #     dim = 128,
        #     depth = 8,
        #     heads = 16,
        #     mlp_dim = 1024,
        #     # dropout = 0.1,
        #     # emb_dropout = 0.1
        # )

        self.v = DeepViT(
            image_size = image_size,
            patch_size = batch_size,
            num_classes = out_chanels,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.device = device

    def forward(self, x) -> torch.Tensor:
        self.v.to(self.device)
        preds = self.v(x)
        return preds