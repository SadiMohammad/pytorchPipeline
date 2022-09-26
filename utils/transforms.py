import torchvision.transforms as transforms


def get_transformers(cfgs):
    transformers = {
        "image": transforms.Compose(
            [
                transforms.Resize(
                    (cfgs["dataset"]["input_size"], cfgs["dataset"]["input_size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "label": transforms.Compose(
            [
                transforms.Resize(
                    (cfgs["dataset"]["input_size"], cfgs["dataset"]["input_size"])
                ),
                transforms.ToTensor(),
            ]
        ),
    }
    return transformers
