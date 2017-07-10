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

all_models = list(set([i.split(".")[0] for i in os.listdir(data_config["checkpoint_dir"]) if "model" in i]))
all_models.sort(key=lambda x: int(x[6:]))

all_model_paths = [os.path.join(data_config["checkpoint_dir"],i) for i in all_models]
results = []

for batch in range(0,len(all_model_paths),5):
    result_batch = test.main(all_model_paths[batch:(min(batch+1,len(all_model_paths)))],"test",64,)
    results += result_batch
    with open(os.path.join(data_config["result_dir"],"results.pkl"),"wb") as fl:
        pickle.dump(results,fl)
    print("Saving Results for :")
    print(all_models[batch:(min(batch+1,len(all_model_paths)))])
    for model,result in zip(all_models[batch:(min(batch+1,len(all_model_paths)))],result_batch):
        with open(os.path.join(data_config["result_dir"],model+"_result.json"),"w") as fl:
            fl.write(json.dumps(result, indent=4, sort_keys=True))
        create_result.create(model+"_generated.json",model+"_result.json",data_config)
