import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class Dataset_ROM(Dataset):
    def __init__(self, cfgs, subset, transformers):
        self.cfgs = cfgs
        self.subset = subset
        self.transformers = transformers
        df = pd.read_csv(cfgs["dataset"][self.subset + "_image_ids"], header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(cfgs["dataset"][self.subset + "_label_ids"], header=None)
        self.label_files = df[df.columns[0]].tolist()

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.cfgs["dataset"]["image_path"], self.image_files[idx])
        ).convert(self.cfgs["dataset"]["img_convert"])
        label = Image.open(
            os.path.join(self.cfgs["dataset"]["label_path"], self.label_files[idx])
        ).convert(self.cfgs["dataset"]["label_convert"])
        if self.transformers:
            t_image = self.transformers["image"](image)
            if "train" in self.cfgs["experiment_name"].lower():
                t_label = self.transformers["label"](label)
        return t_image, t_label

    def __len__(self):
        return len(self.image_files)


class Dataset_RAM(Dataset):
    def __init__(self, cfgs, subset, transformers):
        self.cfgs = cfgs
        self.subset = subset
        self.transformers = transformers
        df = pd.read_csv(cfgs["dataset"][self.subset + "_image_ids"], header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(cfgs["dataset"][self.subset + "_label_ids"], header=None)
        self.label_files = df[df.columns[0]].tolist()

        self.images = []
        self.labels = []
        for img_file, label_file in tqdm(
            zip(self.image_files, self.label_files), total=len(self.image_files)
        ):
            image, label = self.get_item(img_file, label_file)
            self.images.append(image)
            self.labels.append(label)

    def get_item(self, img_file, label_file):
        image = Image.open(
            os.path.join(self.cfgs["dataset"]["image_path"], img_file)
        ).convert(self.cfgs["dataset"]["img_convert"])
        label = Image.open(os.path.join(self.cfgs["dataset"]["label_path"], label_file)).convert(
            self.cfgs["dataset"]["label_convert"]
        )
        if self.transformers:
            t_image = self.transformers["image"](image)
            if "train" in self.cfgs["experiment_name"].lower():
                t_label = self.transformers["label"](label)
        return t_image, t_label

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.image_files)
