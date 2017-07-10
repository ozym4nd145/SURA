from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import test
import create_result
import create_reference
import pickle
from configuration import DataConfig

data_config = DataConfig().config

create_reference.create_reference()

all_generated = [x for x in os.listdir(data_config["result_dir"]) if "result.json" in x]

all_generated.sort(key = lambda x: int(x.split("_")[0].split("-")[1]))

models = [md.split("_")[0] for md in all_generated]

for model in models:
    create_result.create(model+"_generated.json",model+"_result.json",data_config)
