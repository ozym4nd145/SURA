from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
from coco_eval.pycocotools.coco import COCO
from coco_eval.pycocoevalcap.eval import COCOEvalCap
import configuration
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

def get_score(ref_filename,gen_filename,data_config):
    ref_file = os.path.join(data_config["result_dir"],ref_filename)
    gen_file = os.path.join(data_config["result_dir"],gen_filename)

    if not os.path.isfile(ref_file):
        print("%s not found in %s\nExiting" %(ref_filename,data_config["result_dir"]))
        sys.exit()

    if not os.path.isfile(gen_file):
        print("%s not found in %s\nExiting" %(gen_filename,data_config["result_dir"]))
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
    return cocoEval

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_all", help="evaluate score of all the models",
                        action="store_true")
    args = parser.parse_args()

    data_config = configuration.DataConfig().config
    if args.eval_all:
        result_files = [i for i in os.listdir(data_config["result_dir"]) if "_generated.json" in i]
        result_files.sort(key=lambda x: int(x[6:-15]) )
        models = [i[:-15] for i in result_files]
        scores = []
        for i,result in enumerate(result_files):
            start_time = time.time()
            print("---------%d of %d--------%s------------------------" %(i+1,len(models),models[i]))
            res = get_score("reference.json",result,data_config)
            scores.append(res.eval)
            print("-------- Time taken = %.3f ----------------" %(time.time()-start_time))
        
        model_scores = list(zip(models,scores))
        with open(os.path.join(data_config["result_dir"],"model_scores.pkl"),"wb") as fl:
            pickle.dump(model_scores,fl)
        
        with open(os.path.join(data_config["result_dir"],"result.csv"),"w") as fl:
            fl.write("%15s , %8s , %8s , %8s , %8s , %8s , %8s , %8s\n" 
                    %("model","Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L","CIDEr"))

            for model,data in zip(models,scores):
                fl.write("%15s , %8.3f , %8.3f , %8.3f , %8.3f , %8.3f , %8.3f , %8.3f\n"
                                                                        %(model,data["Bleu_1"],
                                                                        data["Bleu_2"],data["Bleu_3"],
                                                                        data["Bleu_4"],data["METEOR"],
                                                                        data["ROUGE_L"],data["CIDEr"]))
        table = {"model":[],"Bleu_1":[],"Bleu_2":[],"Bleu_3":[],"Bleu_4":[],"METEOR":[],"ROUGE_L":[],"CIDEr":[]}
        table["model"] = [i[0] for i in model_scores]
        table["Bleu_1"] = [i[1]["Bleu_1"] for i in model_scores]
        table["Bleu_2"] = [i[1]["Bleu_2"] for i in model_scores]
        table["Bleu_3"] = [i[1]["Bleu_3"] for i in model_scores]
        table["Bleu_4"] = [i[1]["Bleu_4"] for i in model_scores]
        table["METEOR"] = [i[1]["METEOR"] for i in model_scores]
        table["ROUGE_L"] = [i[1]["ROUGE_L"] for i in model_scores]
        table["CIDEr"] = [i[1]["CIDEr"] for i in model_scores]
        df = pd.DataFrame(table)
        plt.figure()
        df.plot(x='model',marker='o',title="Captioning Scores")
        plt.legend(loc='best')
        plt.yticks(np.arange(0.2, 1.0, 0.05))
        figure = plt.gcf()
        figure.set_size_inches(20, 14)
        plt.savefig(os.path.join(data_config["result_dir"],"graph.png"))
        plt.close()
 
    else:
        result = get_score("reference.json","generated.json",data_config)
        # print output evaluation scores
        for metric, score in result.eval.items():
            print('%s: %.3f'%(metric, score))
    
