import torch
from torchvision.models import resnet
# define the model
model = resnet.resnet34(pretrained=True)
model.eval()
# trace model with a dummy input
traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
traced_model.save('resnet34.pt')
