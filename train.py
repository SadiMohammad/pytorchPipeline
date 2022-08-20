import models
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.UNet(1, 1).to(device)