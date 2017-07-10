from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import time
import json

from model import Caption_Model
from data_generator import Data_Generator
from inference_util import Inference

import inception_base
import configuration

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("batch_size", 64,
                       "Batch size of train data input.")
tf.flags.DEFINE_integer("beam_size", 3,
                       "Beam size.")
tf.flags.DEFINE_string("checkpoint_model", None,
                       "Model Checkpoint to use.")
tf.flags.DEFINE_integer("max_captions", None,
                       "Maximum number of captions to generate")
tf.flags.DEFINE_integer("max_len_captions", None,
                       "Maximum length of captions to generate")
tf.flags.DEFINE_string("dataset", "test",
                       "Dataset to use")
tf.flags.DEFINE_string("outfile_name", "generated_caption.json",
                       "Name of the output result file")

def generate_captions(sess,video_paths,model_path,infer_util,max_iter,max_len,batch_size):
    gen_caption = []    
    saver = tf.train.Saver()

    if model_path != None:
        print("Restoring weights from %s" %model_path)
        saver.restore(sess,model_path)
        
    else:
        print("No checkpoint found. Exiting")
        return

    video_files = list(video_paths.keys())    
    
    iter = 0
    btch = 0
    for btch in range(0,len(video_files),batch_size):
        print("Processing batch %d" %(int(btch/batch_size)+1))
        start = btch
        end = min(len(video_files),btch+batch_size)
        dataset={}
        dataset["video"] = np.asarray([np.load(video_paths[video_files[i]]) for i in range(start,end)])
        dataset["path"] = [video_paths[video_files[i]] for i in range(start,end)]
        dataset["file"] = [video_files[i] for i in range(start,end)]

        dataset["gen_caption"] = infer_util.generate_caption_batch_beam(sess,dataset["video"])
            
        for i in range(len(dataset['gen_caption'])):
            dictionary = {}
            dictionary["gen_caption"] = dataset['gen_caption'][i]
            dictionary["file_name"] = dataset['file'][i]
            dictionary["path"] = dataset['path'][i]
            gen_caption.append(dictionary)
            iter+=1
            if iter >= max_iter:
                break
        if iter >= max_iter:
            break
    return gen_caption

def main(model_paths,dataset,batch_size,max_len_captions=None,max_captions=None):
    with tf.Graph().as_default(): #so that the whole program doesn't get polluted
        data_config = configuration.DataConfig().config
        data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                                caption_file = data_config["caption_file"],
                                unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                                max_caption_len = data_config["max_caption_length"])

        data_gen.load_vocabulary(data_config["caption_data_dir"])
        data_gen.load_dataset(data_config["caption_data_dir"])

        assert dataset in ["val","test","train"]

        if max_len_captions:
            max_len = max_len_captions
        else:
            max_len = data_config['max_caption_length']

        model_config = configuration.ModelConfig(data_gen).config
        model_config["beam_width"] = FLAGS.beam_size
        model = Caption_Model( **model_config,mode="inference")
        model.build()

        infer_util = Inference(model,data_gen.word_to_idx,data_gen.idx_to_word)

        if max_captions:
            max_iter = max_captions
        else:
            max_iter = len(data_gen.dataset[dataset])+10 #+10 is just to be safe ;)
        
        video_paths = {i["file_name"]:i["path"] for i in data_gen.dataset[dataset]}
        
        
        gen_captions = []
        with tf.Session() as sess:
            for model_path in model_paths:
                gen_captions.append(generate_captions(sess,video_paths,model_path,infer_util,max_iter,max_len,batch_size))

        return gen_captions

if __name__ == "__main__":
    FLAGS._parse_flags()
    data_config = configuration.DataConfig().config

    if FLAGS.checkpoint_model:
        model_path = FLAGS.checkpoint_model
    else:
        model_path = tf.train.latest_checkpoint(data_config["checkpoint_dir"])

    gen_caption = main([model_path],FLAGS.dataset,FLAGS.batch_size,FLAGS.max_len_captions,FLAGS.max_captions)[0]


    with open(os.path.join(data_config["result_dir"],FLAGS.outfile_name),"w") as fl:
        fl.write(json.dumps(gen_caption, indent=4, sort_keys=True))
