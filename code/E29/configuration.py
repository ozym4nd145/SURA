from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DataConfig(object):
    def __init__(self):
        self.config = {}
        self.config["processed_video_dir"] = "../../dataset/msrvtt_processed_video/" #folder structure should be of form => 
                                                           # [test/.. , val/... , train/...]
        self.config["caption_file"] = "../../dataset/raw/MSR-VTT_2017/videodatainfo_2017_cleaned.json"
        self.config["unique_frequency_cutoff"] = 2 #words whose frequecy is less than this will be given as <unk>
        self.config["max_caption_length"] = 21 #maximum caption length that is allowed
        self.config["train_log_dir"] = "./models/train"
        self.config["checkpoint_dir"] = self.config["train_log_dir"]
        self.config["val_log_dir"] = "./models/val"
        self.config["caption_data_dir"] = "./caption_data"
        self.config["inception_pretrained_checkpoint"] = "../../dataset/inception_v4.ckpt"
        self.config["result_dir"] = "./Results"

class ModelConfig(object):
    def __init__(self,data_gen):
        self.config={}
        self.config["num_frames"] = 100
        self.config["image_width"] = 299
        self.config["image_height"] = 299
        self.config["image_channels"] = 3
        self.config["num_caption_unroll"] = 20
        self.config["num_last_layer_units"] = 1536
        self.config["word_embedding_size"] = 256
        self.config["hidden_size_lstm"] = 256
        self.config["num_lstm_layer"] = 1
        self.config["vocab_size"] = data_gen.vocab_size
        self.config["initializer_scale"] = 0.3
        self.config["learning_rate"] = 5e-5
        self.config["encoder_input_keep_prob"] = 1.0
        self.config["encoder_output_keep_prob"] = 1.0
        self.config["decoder_input_keep_prob"] = 0.7
        self.config["decoder_output_keep_prob"] = 0.7
        self.config["use_residual_encoder"] = False
        self.config["use_residual_decoder"] = False
        self.config["attention_num_units"] = 100
        self.config["attention_layer_units"] = 100
        self.config["beam_width"] = 3
        self.config["max_gradient_norm"] = 1.25
        self.config["use_gradient_clipping"] = False
        self.config["start_token_id"] = data_gen.word_to_idx["<bos>"]
        self.config["end_token_id"] = data_gen.word_to_idx["<eos>"]
        self.config["embedding_file"] = "./caption_data/embed.npy"
        
