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
    def __init__(self, experimanet_name):
        self.exp_name = experimanet_name
        cfg_file_path = os.path.join("../configs", self.exp_name + ".yaml")
        with open(cfg_file_path, "r") as file:
            cfgs = yaml.safe_load(file)
        cfgs_single_dict = func(cfgs)

        # config
        for key, value in cfgs_single_dict.items():
            setattr(self, key, value)


if __name__ == "__main__":
    experimanet_name = "train"
    Cfgs = Config(experimanet_name)
    print(Cfgs.__dict__)
