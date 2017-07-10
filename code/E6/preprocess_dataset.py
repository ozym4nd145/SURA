import configuration
from data_generator import Data_Generator
data_config = configuration.DataConfig().config
data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                         caption_file = data_config["caption_file"],
                         unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                         max_caption_len = data_config["max_caption_length"])
data_gen.build_basic_dataset()
data_gen.build_vocabulary()
data_gen.build_process_captions()
data_gen.save_dataset("./caption_data")
data_gen.save_vocabulary("./caption_data")