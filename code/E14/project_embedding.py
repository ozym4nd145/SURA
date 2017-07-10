from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model import Caption_Model
from data_generator import Data_Generator
import configuration

data_config = configuration.DataConfig().config
data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                        caption_file = data_config["caption_file"],
                        unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                        max_caption_len = data_config["max_caption_length"])

data_gen.load_vocabulary(data_config["caption_data_dir"])
data_gen.load_dataset(data_config["caption_data_dir"])
model_config = configuration.ModelConfig(data_gen).config
model = Caption_Model(**model_config,mode="inference")
model.build()

proj_config = projector.ProjectorConfig()
embedding = proj_config.embeddings.add()
embedding.tensor_name = model.word_emb.name
embedding.metadata_path = os.path.join(os.path.abspath(data_config["train_log_dir"]),"metadata.tsv")
summary_writer = tf.summary.FileWriter(data_config["train_log_dir"])
projector.visualize_embeddings(summary_writer, proj_config)

keys = list(data_gen.idx_to_word.keys())
keys.sort()
with open(embedding.metadata_path,"w") as fl:
    for key in keys:
        fl.write(data_gen.idx_to_word[key]+"\n")
