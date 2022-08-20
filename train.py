# from models.model import Model
# import torch
# from utils.config import Config
# import argparse
# arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument(
#     "--config_file", help="config file name/experiment name", required=True
# )
# args = arg_parser.parse_args()
# Cfgs = Config(args)
# cfgs = Cfgs.cfgs
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Model(cfgs).get_model()
# print(model)

# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch
# from utils.config import Config
# from dataloaders import Dataset_ROM
# import argparse

# arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument(
#     "--config_file", help="config file name/experiment name", required=True
# )
# args = arg_parser.parse_args()
# Cfgs = Config(args)
# cfgs = Cfgs.cfgs

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# transformers = {
#     "image": transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     ),
#     "gt": transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ]
#     ),
# }

# dataset_train = Dataset_ROM(cfgs, transformers=transformers)
# loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
# for i_train, sample_train in enumerate(tqdm(loader_train)):
#     images = sample_train[0]
#     labels = sample_train[1]
#     print(images.size())
#     break
