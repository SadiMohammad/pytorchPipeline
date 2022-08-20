import yaml
import os


def func(dic):
    re = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            re.update(v)
        else:
            re.update({k: v})
    return re


class Config:
    def __init__(self, args):
        self.config_file = args.config_file
        cfg_file_path = os.path.join("../configs", self.config_file + ".yaml")
        with open(cfg_file_path, "r") as file:
            self.cfgs = yaml.safe_load(file)
        self.cfgs_single_dict = func(self.cfgs)

        # config
        for key, value in self.cfgs_single_dict.items():
            setattr(self, key, value)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file", help="config file name/experiment name", required=True
    )
    args = arg_parser.parse_args()
    Cfgs = Config(args)
    print(Cfgs.cfgs["model"]["model_name"])
