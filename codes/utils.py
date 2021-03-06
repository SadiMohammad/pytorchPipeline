from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from losses import *
import numpy as np

def to_one_hot(value,size):
    np_one_hot = np.zeros(shape= size)
    np_one_hot[value]=1
    return np_one_hot


class Dataset_ROM(Dataset):
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


class Dataset_RAM(Dataset):
    def __init__(self, dir_data, files, filepath_labels_csv, convert='RGB', transform=None,N_VAR_1 = 168,N_VAR_2 = 11,N_VAR_3 = 7):
        self.dir_data = dir_data
        self.convert = convert
        self.transform = transform
        self.files = files
        self.N_VAR_1 = N_VAR_1
        self.N_VAR_2 = N_VAR_2
        self.N_VAR_3 = N_VAR_3
        self.df = pd.read_csv(filepath_labels_csv)
        self.id2label = {x:[t1, t2, t3] for x,t1,t2,t3 in zip(self.df['fileID'].values,
                                                              self.df['targetVar1'].values,
                                                              self.df['targetVar2'].values,self.df['targetVar3'].values)
                         }
        self.image=[]
        self.images = []
        self.labels1 = []
        self.labels2 = []
        self.labels3 = []
        for file in tqdm(self.files):
            image, label1, label2, label3 = self.get_item(file)
            self.images.append(image)
            self.labels1.append(label1)
            self.labels2.append(label2)
            self.labels3.append(label3)

    def get_item(self,file):
        path_img = os.path.join(self.dir_data, file + '.png')
        image = Image.open(path_img).convert(self.convert)
        label1 = to_one_hot(
            self.id2label[file][0],
            self.N_VAR_1
        )
        label2 = to_one_hot(
            self.id2label[file][1],
            self.N_VAR_2
        )
        label3 = to_one_hot(
            self.id2label[file][2],
            self.N_VAR_3
        )
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
        label3 = torch.tensor(label3)
        if self.transform:
            image = self.transform['image'](image)
        return image, label1, label2, label3

    def __len__(self):
        size = len(self.files)
        return size

    def __getitem__(self, idx):
        return self.images[idx].to(device), self.labels1[idx].to(device), self.labels2[idx].to(device), self.labels3[idx].to(device)

def evalModel(model, validDataset, device):
    totValDice = 0
    for i_valid, sample_valid in enumerate(validDataset):
        images = sample_valid[0].to(device)
        trueMasks = sample_valid[1].to(device)

        preds = model(images)
        predMasks = preds
        # preds = preds['out']
        # predMasks = torch.sigmoid(preds)
        predMasks = (predMasks > 0.5).float()

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        # print(valDice)
        totValDice += valDice.item()
    return totValDice / (i_valid + 1)


class Dataset_RAM_TEST(Dataset):
    def __init__(self, image_paths, size, convert='L'):
        self.image_paths = image_paths
        self.convert = convert
        self.transforms = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert(self.convert)
        image = image.filter(ImageFilter.BLUR)
        t_image = self.transforms(image)
        return t_image

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    print('dummy')
