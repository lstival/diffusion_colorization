import torch
from torch import nn
# from vit_pytorch import ViT
# from vit_pytorch import SimpleViT
# from vit_pytorch.pit import PiT
# from vit_pytorch.deepvit import DeepViT
# from vit_pytorch.vit_for_small_dataset import ViT as ViT_small

from transformers import ViTModel, ViTConfig

class Vit_neck(nn.Module):
    def __init__(self, batch_size, image_size, out_chanels=256):
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

        # self.v = DeepViT(
        #     image_size = image_size,
        #     patch_size = batch_size,
        #     num_classes = out_chanels,
        #     dim = 256,
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 1024,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )

        self.configuration = ViTConfig(image_size=image_size, hidden_size=384,intermediate_size=1536,
                                        num_hidden_layers=6, num_attention_heads=6)
        self.v = ViTModel(self.configuration)
        # self.v = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.v.eval()

        self.conv_out = nn.Sequential(
            nn.Conv2d(65, 124, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 124),
            nn.GELU(),
            nn.Conv2d(124, 62, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 62),
        )
        self.lin_out = nn.Sequential(
            nn.Linear(23808, out_chanels),
            nn.GELU(),
        )

    def forward(self, x) -> torch.Tensor:
        preds = self.v(x).last_hidden_state
        x = self.conv_out(preds.unsqueeze(-1))
        x = x.view(x.size(0), -1)
        x = self.lin_out(x)

        return x
    
if __name__ == '__main__':
    image_size = 224
    batch_size = 16
    out_chanels = 1024

    img = torch.ones((batch_size,3,image_size,image_size)).to("cuda")
    model = Vit_neck(batch_size, image_size, out_chanels).to("cuda")
    out = model(img)
    print(out.shape)






