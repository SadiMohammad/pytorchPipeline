from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from losses import *

class Dataset_RAM(Dataset):
    def __init__(self, image_paths, mask_paths, size, convert='L'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.convert = convert
        self.transforms = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert(self.convert)
        image = image.filter(ImageFilter.BLUR)
        mask = Image.open(self.mask_paths[index]).convert('L')
        t_image = self.transforms(image)
        t_mask = self.transforms(mask)
        return t_image, t_mask

    def __len__(self):
        return len(self.image_paths)


def evalModel(model, validDataset, device):
    totValDice = 0
    for i_valid, sample_valid in enumerate(validDataset):
        images = sample_valid[0].to(device)
        trueMasks = sample_valid[1].to(device)

        preds = model(images)
        predMasks = preds
        # preds = preds['out'][:, 0, :, :] 
        # predMasks = torch.sigmoid(preds)
        predMasks = (predMasks > 0.5).float()

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        # print(valDice)
        totValDice += valDice.item()
    return totValDice / (i_valid + 1)


if __name__ == "__main__":
    print('dummy')
