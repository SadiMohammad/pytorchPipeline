import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm


class Dataset_ROM(Dataset):
    def __init__(
        self,
        experiment,
        image_path,
        image_ids,
        gt_path,
        gt_ids,
        img_convert="L",
        gt_convert="L",
        transformers=None,
        device="cpu",
    ):
        self.experiment = experiment
        self.image_path = image_path
        self.gt_path = gt_path
        self.img_convert = img_convert
        self.gt_convert = gt_convert
        self.transformers = transformers
        df = pd.read_csv(image_ids, header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(gt_ids, header=None)
        self.gt_files = df[df.columns[0]].tolist()
        self.device = device

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.image_path, self.image_files[idx])
        ).convert(self.img_convert)
        gt = Image.open(os.path.join(self.gt_path, self.gt_files[idx])).convert(
            self.gt_convert
        )
        if self.transformers:
            t_image = self.transformers["image"](image)
            if self.experiment == "train":
                t_gt = self.transformers["gt"](gt)
        return t_image.to(self.device), t_gt.to(self.device)

    def __len__(self):
        return len(self.image_files)


class Dataset_RAM(Dataset):
    def __init__(
        self,
        experiment,
        image_path,
        image_ids,
        gt_path,
        gt_ids,
        img_convert="L",
        gt_convert="L",
        transformers=None,
        device="cpu",
    ):
        self.experiment = experiment
        self.image_path = image_path
        self.gt_path = gt_path
        self.img_convert = img_convert
        self.gt_convert = gt_convert
        self.transformers = transformers
        df = pd.read_csv(image_ids, header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(gt_ids, header=None)
        self.gt_files = df[df.columns[0]].tolist()
        self.device = device

        self.images = []
        self.gts = []
        for img_file, gt_file in tqdm(
            zip(self.image_files, self.gt_files), total=len(self.image_files)
        ):
            image, gt = self.get_item(img_file, gt_file)
            self.images.append(image)
            self.gts.append(gt)

    def get_item(self, img_file, gt_file):
        image = Image.open(os.path.join(self.image_path, img_file)).convert(
            self.img_convert
        )
        gt = Image.open(os.path.join(self.gt_path, gt_file)).convert(self.gt_convert)
        if self.transformers:
            t_image = self.transformers["image"](image)
            if self.experiment == "train":
                t_gt = self.transformers["gt"](gt)
        return t_image, t_gt

    def __getitem__(self, idx):
        return self.images[idx].to(self.device), self.gts[idx].to(self.device)

    def __len__(self):
        return len(self.image_files)


if __name__ == "__main__":
    image_path = "/media/sadi/Vol_2/US/DSPRL-Ultrasoundseg/data/data_for_pipeline_check/test/images"
    image_ids = "/media/sadi/Vol_2/US/DSPRL-Ultrasoundseg/data/data_for_pipeline_check/test/img_list.txt"
    gt_path = (
        "/media/sadi/Vol_2/US/DSPRL-Ultrasoundseg/data/data_for_pipeline_check/test/gt"
    )
    gt_ids = "/media/sadi/Vol_2/US/DSPRL-Ultrasoundseg/data/data_for_pipeline_check/test/gt_list.txt"
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformers = {
        "image": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "gt": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    }

    dataset_train = Dataset_RAM(
        "train",
        image_path,
        image_ids,
        gt_path,
        gt_ids,
        img_convert="RGB",
        gt_convert="L",
        transformers=transformers,
        device=device,
    )
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
    for i_train, sample_train in enumerate(tqdm(loader_train)):
        images = sample_train[0]
        labels = sample_train[1]
