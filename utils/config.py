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
    def __init__(self, config_file):
        self.config_file = config_file
        cfg_file_path = os.path.join("../configs", self.config_file + ".yaml")
        with open(cfg_file_path, "r") as file:
            cfgs = yaml.safe_load(file)
        self.cfgs_single_dict = func(cfgs)

        # config
        for key, value in self.cfgs_single_dict.items():
            setattr(self, key, value)


if __name__ == "__main__":
    config_file = "train"
    Cfgs = Config(config_file)
    print(Cfgs.__dict__)
