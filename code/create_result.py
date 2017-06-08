from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import configuration
import sys

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

data_config = configuration.DataConfig().config

video_to_id_file = os.path.join(data_config["result_dir"],"video_to_id.json")
gen_file = os.path.join(data_config["result_dir"],"generated_caption.json")

if not os.path.isfile(video_to_id_file):
    print("video_to_id.json not found in %s\nExiting" %(data_config["result_dir"]))
    sys.exit()

if not os.path.isfile(gen_file):
    print("generated_caption.json not found in %s\nExiting" %(data_config["result_dir"]))
    sys.exit()

with open(video_to_id_file,"r") as fl:
    video_to_id = json.loads(fl.read())
with open(gen_file,"r") as fl:
    gen = json.loads(fl.read())

gen.sort(key = lambda x: int(x["file_name"][5:][:-8]))

generated = []

for data in gen:
    dictionary = {}
    dictionary["image_id"] = video_to_id[data["file_name"][:-8]]
    dictionary["caption"] = data["gen_caption"]
    if is_ascii(dictionary['caption']):
        generated.append(dictionary)

with open(os.path.join(data_config["result_dir"],"generated.json"),"w") as fl:
    fl.write(json.dumps(generated,sort_keys=True,indent=4))