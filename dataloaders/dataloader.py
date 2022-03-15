import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd


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

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.image_path, self.image_files[idx])
        ).convert(self.img_convert)
        gt = Image.open(os.path.join(self.image_path, self.image_files[idx])).convert(
            self.gt_convert
        )
        if self.transformers:
            t_image = self.transformers["image"](image)
            if self.experiment == "train":
                t_gt = self.transformers["gt"](gt)
        return t_image, t_gt

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

    dataset_train = Dataset_ROM(
        "train",
        image_path,
        image_ids,
        gt_path,
        gt_ids,
        img_convert="RGB",
        gt_convert="L",
        transformers=transformers,
    )
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
    for i_train, sample_train in enumerate(tqdm(loader_train)):
        images = sample_train[0]
        labels = sample_train[1]
