from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import configuration
from data_generator import Data_Generator

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

data_config = configuration.DataConfig().config
data_gen = Data_Generator(data_config["processed_video_dir"],
                        data_config["caption_file"],
                        data_config["unique_frequency_cutoff"],
                        data_config["max_caption_length"])

data_gen.load_vocabulary(data_config["caption_data_dir"])
data_gen.load_dataset(data_config["caption_data_dir"])


ref = {'info':{},'images':[],'licenses':[],'type':'captions','annotations':[]}

ref['info'] = {'contributor': 'Suyash Agrawal',
               'date_created': '2017-06-08',
               'description': 'test',
               'url': 'https://github.com/ozym4nd145',
               'version': '1.0',
               'year': 2017}

ref['licenses'].append({'id': 1,
                        'name': 'Attribution-NonCommercial-ShareAlike License',
                        'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'})

video_to_id = {}
id_to_video = {}

all_data = []

for i in data_gen.dataset.keys():
    all_data += data_gen.dataset[i]

video_ids = list(set([i['file_name'][:-4] for i in all_data]))

video_ids.sort(key= lambda x: int(x[5:][:-4]))

for idx,name in enumerate(video_ids):
    dictionary = {}
    dictionary['file_name'] = name[:-4]
    dictionary['id'] = idx
    dictionary['license'] = 1
    ref['images'].append(dictionary)
    video_to_id[name[:-4]] = idx
    id_to_video[idx] = name[:-4]

all_data.sort(key= lambda x: int(x["file_name"][5:][:-8]))

for idx,data in enumerate(all_data):
    dictionary = {}
    dictionary['caption'] = data['caption']
    dictionary['id'] = idx
    dictionary['image_id'] = video_to_id[data['file_name'][:-8]]

    if is_ascii(dictionary['caption']):
        ref['annotations'].append(dictionary)

os.makedirs(data_config["result_dir"],exist_ok=True)

with open(os.path.join(data_config["result_dir"],"reference.json"),"w") as fl:
    fl.write(json.dumps(ref, indent=4, sort_keys=True))

with open(os.path.join(data_config["result_dir"],"video_to_id.json"),"w") as fl:
    fl.write(json.dumps(video_to_id, indent=4, sort_keys=True))

with open(os.path.join(data_config["result_dir"],"id_to_video.json"),"w") as fl:
    fl.write(json.dumps(id_to_video, indent=4, sort_keys=True))