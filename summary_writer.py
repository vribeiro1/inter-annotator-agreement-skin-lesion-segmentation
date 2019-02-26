import os
import json


class SummaryWriter:
    def __init__(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.dir = dir
        self.dict = {}

    def add_scalar(self, tag, value, key):
        if tag not in self.dict:
            self.dict[tag] = {}

        self.dict[tag][key] = value

    def commit(self):
        with open(os.path.join(self.dir, "writer.json"), "w+") as f:
            json.dump(self.dict, f)
