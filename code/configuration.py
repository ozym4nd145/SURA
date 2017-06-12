from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inception_base

class DataConfig(object):
    def __init__(self):
        self.config = {}
        self.config["processed_video_dir"] = "../dataset/MSVD_processed/" #folder structure should be of form => 
                                                           # [test/.. , val/... , train/...]
        self.config["caption_file"] = "/home/ozym4nd145/Coding/Notebook/SURA/dataset/MSVD_processed/caption_file.json"
        self.config["unique_frequency_cutoff"] = 1 #words whose frequecy is less than this will be given as <unk>
        self.config["max_caption_length"] = 21 #maximum caption length that is allowed
        self.config["train_log_dir"] = "./s2vt_models/train"
        self.config["checkpoint_dir"] = self.config["train_log_dir"]
        self.config["val_log_dir"] = "./s2vt_models/val"
        self.config["caption_data_dir"] = "./caption_data"
        self.config["inception_pretrained_checkpoint"] = "./inception_v4.ckpt"
        self.config["result_dir"] = "./Results"

class ModelConfig(object):
    def __init__(self,data_gen):
        self.config={}
        self.config["num_frames"] = 100
        self.config["image_width"] = 299
        self.config["image_height"] = 299
        self.config["image_channels"] = 3
        self.config["num_caption_unroll"] = 20
        self.config["num_last_layer_units"] = inception_base.num_end_units_v4
        self.config["image_embedding_size"] = 1000
        self.config["word_embedding_size"] = 1000
        self.config["hidden_size_lstm1"] = 1000
        self.config["hidden_size_lstm2"] = 1000
        self.config["vocab_size"] = data_gen.vocab_size
        self.config["initializer_scale"] = 0.1
        self.config["learning_rate"] = 1e-4
