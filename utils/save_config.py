import yaml
import os


class RunHistory:
    def __init__(self, time_stamp, cfgs, filepath):
        self.time_stamp = time_stamp
        self.filepath = filepath
        self.cfgs = cfgs

    def save_run_history(self):
        dir = os.path.join(self.filepath, self.cfgs["experiment_name"])
        if not (os.path.exists(dir)):
            os.makedirs(dir)
        with open(os.path.join(dir, "{}.yaml".format(self.time_stamp)), "w") as outfile:
            yaml.dump(self.cfgs, outfile, default_flow_style=False)
