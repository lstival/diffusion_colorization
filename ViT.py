import torch
from torch import nn
from torchvision.models import vit_b_16

class Vit_neck(nn.Module):
    def __init__(self, batch_size, image_size, out_chanels=256):
        super(Vit_neck, self).__init__()

        self.linear_size = (image_size//16 * image_size//16 * 768)

        v = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1").to("cuda")
        self.v = torch.nn.Sequential(*(list(v.children())[:-2]))
        for param in self.v.parameters():
            param.requires_grad = False


        self.lin_out = nn.Sequential(
            nn.Linear(self.linear_size, out_chanels),
            nn.GELU(),
        )

    def forward(self, x) -> torch.Tensor:
        preds = self.v(x)
        x = preds.view(preds.size(0), -1)
        x = self.lin_out(x)

        return x
    
if __name__ == '__main__':
    model = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1").to("cuda")

    image_size = 128
    batch_size = 16
    out_chanels = 1024

    img = torch.ones((batch_size,3,image_size,image_size)).to("cuda")
    model = Vit_neck(batch_size, image_size, out_chanels).to("cuda")
    out = model(img)
    print(out.shape)






