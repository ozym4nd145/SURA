from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import device_lib

from model import Caption_Model
from data_generator import Data_Generator
from inference_util import Inference

import inception_base
import configuration
import argparse

FLAGS = None

def main(unused_argv):
    data_config = configuration.DataConfig().config
    data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                             caption_file = data_config["caption_file"],
                             unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                             max_caption_len = data_config["max_caption_length"])

    data_gen.load_vocabulary(data_config["caption_data_dir"])
    data_gen.load_dataset(data_config["caption_data_dir"])
    #data_gen.build_dataset()

    model_config = configuration.ModelConfig(data_gen).config
    model = Caption_Model( num_frames = model_config["num_frames"],
                           image_width = model_config["image_width"],
                           image_height = model_config["image_height"],
                           image_channels = model_config["image_channels"],
                           num_caption_unroll = model_config["num_caption_unroll"],
                           num_last_layer_units = model_config["num_last_layer_units"],
                           image_embedding_size = model_config["image_embedding_size"],
                           word_embedding_size = model_config["word_embedding_size"],
                           hidden_size_lstm = model_config["hidden_size_lstm"],
                           num_lstm_layer = model_config["num_lstm_layer"],
                           vocab_size = model_config["vocab_size"],
                           initializer_scale = model_config["initializer_scale"],
                           learning_rate = model_config["learning_rate"],
                           mode="train",
                           rnn1_input_keep_prob=model_config["rnn1_input_keep_prob"],
                           rnn1_output_keep_prob=model_config["rnn1_output_keep_prob"],
                           rnn2_input_keep_prob=model_config["rnn2_input_keep_prob"],
                           rnn2_output_keep_prob=model_config["rnn2_output_keep_prob"]
                          )
    model.build()

    summary_op = tf.summary.merge(model.summaries)
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(data_config["train_log_dir"],sess.graph)
        saver = tf.train.Saver(max_to_keep=200,keep_checkpoint_every_n_hours=0.5)

        if FLAGS.checkpoint_model:
            model_path = FLAGS.checkpoint_model
        else:
            model_path = tf.train.latest_checkpoint(data_config["checkpoint_dir"])

        if model_path != None:
            print("Restoring weights from %s" %model_path)
            saver.restore(sess,model_path)
        else:
            print("No checkpoint found. Intializing Variables from scratch")
            sess.run(tf.global_variables_initializer())
        
        data_gen.init_batch(int(FLAGS.batch_size),"train")

        if FLAGS.save_freq:
            iter_to_save = np.int32(FLAGS.save_freq)
        else:
            iter_to_save = int(data_gen.iter_per_epoch["train"]/4)

        for epoch in range(int(FLAGS.num_epochs)):
            for i in range(data_gen.iter_per_epoch["train"]):
                start_time = time.time()
                dataset = data_gen.get_next_batch("train")
                data_gen_time = time.time() - start_time
                
                feed_dict={}
                feed_dict[model.video_mask] = np.ones([dataset["video"].shape[0],dataset["video"].shape[1]],dtype=np.int32)
                feed_dict[model.caption_input] = dataset["indexed_caption"]
                feed_dict[model.caption_mask] = dataset["caption_mask"]
                feed_dict[model.rnn_input] = dataset["video"]
                feed_dict[model.is_training] = True
                
                if np.mod(i+1, int(FLAGS.summary_freq)) == 0:
                    print("Writing Summary")
                    summary,loss,global_step,_ = sess.run([summary_op,model.batch_loss,model.global_step,model.train_step],feed_dict=feed_dict)
                    train_writer.add_summary(summary, global_step)
                    
                    time_global_step = tf.Summary()
                    value = time_global_step.value.add()
                    value.simple_value = (time.time() - start_time)
                    value.tag = "global_step/time_global_step"
                    train_writer.add_summary(time_global_step,global_step)
                else:
                    loss,global_step,_ = sess.run([model.batch_loss,model.global_step,model.train_step],feed_dict=feed_dict)
                
                print("global_step = ",global_step,
                    ", loss = ",loss,
                    ', Elapsed time: %.2f' %(time.time() - start_time))        
                
                if np.mod(i+1,iter_to_save) == 0:
                    print("Saving the model ...")
                    saver.save(sess, os.path.join(data_config["checkpoint_dir"], 'model'), global_step=int(global_step))

        print("Saving the model ...")
        saver.save(sess, os.path.join(data_config["checkpoint_dir"], 'model'), global_step=int(global_step))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="Batch size of train data input",
                        type=int,default=64)
    parser.add_argument("num_epochs", help="Number of epochs to train the model.",
                        type=int)
    parser.add_argument("--checkpoint_model", help="Model Checkpoint to use.",
                        type=str, default=None)
    parser.add_argument("--summary_freq", help="Frequency of writing summary to tensorboard.",
                        type=int, default=100)
    parser.add_argument("--save_freq", help="Frequency of saving model.",
                        type=int, default=None)
    FLAGS = parser.parse_args()
    main(None)
