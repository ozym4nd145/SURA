from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import scipy
import time
from tensorflow.python.client import device_lib

from model import Model_S2VT
from data_generator import Data_Generator
from inference_util import Inference

import inception_base
import configuration

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("batch_size", 64,
                       "Batch size of train data input.")
tf.flags.DEFINE_integer("num_epochs", 10,
                       "Number of epochs to train the model.")
tf.flags.DEFINE_string("checkpoint_model", None,
                       "Model Checkpoint to use.")
tf.flags.DEFINE_string("inception_checkpoint", None,
                       "Inception Checkpoint to use")
tf.flags.DEFINE_integer("summary_freq", 100,
                       "Frequency of writing summary to tensorboard.")
tf.flags.DEFINE_integer("save_freq", None,
                       "Frequency of saving model.")


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
            print("No checkpoint found. Intializing Variables from scratch and restoring from inception checkpoint")
            assert FLAGS.inception_checkpoint, "--Inception checkpoint must be given"
            sess.run(tf.global_variables_initializer())
            saver2 = tf.train.Saver(model.inception_variables)
            saver2.restore(sess,FLAGS.inception_checkpoint)
        
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
                feed_dict[model.caption_input] = dataset["indexed_caption"]
                feed_dict[model.caption_mask] = dataset["caption_mask"]
                feed_dict[model.rnn_input] = dataset["video"]
                
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
  tf.app.run()
