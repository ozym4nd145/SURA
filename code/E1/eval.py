from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
from coco_eval.pycocotools.coco import COCO
from coco_eval.pycocoevalcap.eval import COCOEvalCap
import configuration
data_config = configuration.DataConfig().config

ref_file = os.path.join(data_config["result_dir"],"reference.json")
gen_file = os.path.join(data_config["result_dir"],"generated.json")

if not os.path.isfile(ref_file):
    print("reference.json not found in %s\nExiting" %(data_config["result_dir"]))
    sys.exit()

if not os.path.isfile(gen_file):
    print("generated_caption.json not found in %s\nExiting" %(data_config["result_dir"]))
    sys.exit()

# create coco object and cocoRes object
coco = COCO(ref_file)
cocoRes = coco.loadRes(gen_file)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))


