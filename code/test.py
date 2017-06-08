from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import time
import json

from model import Model_S2VT
from data_generator import Data_Generator
from inference_util import Inference

import inception_base
import configuration

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("batch_size", 64,
                       "Batch size of train data input.")
tf.flags.DEFINE_string("checkpoint_model", None,
                       "Model Checkpoint to use.")
tf.flags.DEFINE_integer("max_captions", None,
                       "Maximum number of captions to generate")
tf.flags.DEFINE_integer("max_len_captions", None,
                       "Maximum length of captions to generate")
tf.flags.DEFINE_string("dataset", "test",
                       "Dataset to use")
def main(unused_argv):
    data_config = configuration.DataConfig().config
    data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                             caption_file = data_config["caption_file"],
                             unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                             max_caption_len = data_config["max_caption_length"])

    data_gen.load_vocabulary(data_config["caption_data_dir"])
    data_gen.load_dataset(data_config["caption_data_dir"])
    #data_gen.build_dataset()

    assert FLAGS.dataset in ["val","test","train"]

    if FLAGS.max_len_captions:
        max_len = FLAGS.max_len_captions
    else:
        max_len = data_config['max_caption_length']

    model_config = configuration.ModelConfig(data_gen).config
    model = Model_S2VT( num_frames = model_config["num_frames"],
                        image_width = model_config["image_width"],
                        image_height = model_config["image_height"],
                        image_channels = model_config["image_channels"],
                        num_caption_unroll = model_config["num_caption_unroll"],
                        num_last_layer_units = model_config["num_last_layer_units"],
                        image_embedding_size = model_config["image_embedding_size"],
                        word_embedding_size = model_config["word_embedding_size"],
                        hidden_size_lstm1 = model_config["hidden_size_lstm1"],
                        hidden_size_lstm2 = model_config["hidden_size_lstm2"],
                        vocab_size = model_config["vocab_size"],
                        initializer_scale = model_config["initializer_scale"],
                        learning_rate = model_config["learning_rate"])
    model.build()

    summary_op = tf.summary.merge(model.summaries)
    
    gen_caption = []

    infer_util = Inference(model,data_gen.word_to_idx,data_gen.idx_to_word)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(data_config["train_log_dir"],sess.graph)
        saver = tf.train.Saver(max_to_keep=20,keep_checkpoint_every_n_hours=0.5)

        if FLAGS.checkpoint_model:
            model_path = FLAGS.checkpoint_model
        else:
            model_path = tf.train.latest_checkpoint(data_config["checkpoint_dir"])

        if model_path != None:
            print("Restoring weights from %s" %model_path)
            saver.restore(sess,model_path)
        else:
            print("No checkpoint found. Exiting")
            return

        if FLAGS.max_captions:
            max_iter = FLAGS.max_captions
        else:
            max_iter = len(data_gen.dataset[FLAGS.dataset])+10 #+10 is just to be safe ;)
        
        iter = 0
        btch = 0
        video_paths = {i["file_name"]:i["path"] for i in data_gen.dataset[FLAGS.dataset]}
        video_files = list(video_paths.keys())
        for btch in range(0,len(video_files),FLAGS.batch_size):
            print("Processing batch %d" %(int(btch/FLAGS.batch_size)+1))
            start = btch
            end = min(len(video_files),btch+FLAGS.batch_size)
            dataset={}
            dataset["video"] = np.asarray([np.load(video_paths[video_files[i]]) for i in range(start,end)])
            dataset["path"] = [video_paths[video_files[i]] for i in range(start,end)]
            dataset["file"] = [video_files[i] for i in range(start,end)]
            dataset["gen_caption"] = infer_util.generate_caption_batch(sess,dataset["video"],max_len=max_len)
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
    with open(os.path.join(data_config["result_dir"],"generated_caption.json"),"w") as fl:
        fl.write(json.dumps(gen_caption, indent=4, sort_keys=True))

if __name__ == "__main__":
  tf.app.run()
