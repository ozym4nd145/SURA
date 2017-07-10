from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import pysrt
import sys
import argparse
import numpy as np

import tensorflow as tf

from scipy import misc

from inference_util import Inference
from model import Caption_Model
import inception_base
import configuration
from data_generator import Data_Generator



def generate_captions(sess,model,video_path,inception_checkpoint,infer_util,
                        num_frames_per_clip,num_frames_per_sec,max_cap_len,batch_size,beam_size):
    video = cv2.VideoCapture(video_path)

    image_feed = tf.placeholder(dtype=tf.float32,shape=[None,299,299,3],name="image_feed")
    inception_output = inception_base.get_base_model(image_feed)
    inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV4")
    saver = tf.train.Saver(var_list=inception_variables)
    saver.restore(sess,inception_checkpoint)


    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    time_length = length/fps

    num_frames_to_read = int((int(time_length)*(num_frames_per_sec))/num_frames_per_clip)*num_frames_per_clip
    frames_to_read = set(np.linspace(0,length-1,num=num_frames_to_read,dtype=np.int32,endpoint=False))

    num_clips_per_batch = (num_frames_per_clip*batch_size)
    num_batches = int((num_frames_to_read+num_clips_per_batch-1)/num_clips_per_batch)

    captions=[]
    frame_list = []

    start_time=0
    processed_batch=[]
    batch_index=1
    for i in range(length):
        ret, frame = video.read()
        if ret is False:
            break
        if i in frames_to_read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = misc.imresize(frame,[299,299,3])
            frame = ((2*(frame.astype(np.float32) / 255 ))-1)
            frame = sess.run(inception_output,feed_dict={image_feed:[frame]})
            frame_list.append(frame)
            if len(frame_list)%100==0:
                print(len(frame_list))
        if len(frame_list)==(num_frames_per_clip*batch_size):
            print("Processing batch %d of %d" %(batch_index, num_batches))
            batch_index+= 1
            processed_batch = np.array(frame_list,dtype=np.float32)
            embedded_frames = np.reshape(processed_batch,[-1,num_frames_per_clip,inception_base.num_end_units_v4])
            if beam_size==1:
                caption_batch = infer_util.generate_caption_batch(sess,embedded_frames,max_len=max_cap_len)
            else:
                caption_batch = infer_util.generate_caption_batch_beam(sess,beam_size,embedded_frames,max_len=max_cap_len)
            
            for cap in caption_batch:
                caption = {}
                caption["start_time"] = start_time
                caption["end_time"] = start_time+ (num_frames_per_clip/num_frames_per_sec)
                start_time = caption["end_time"]
                caption["caption"] = cap
                captions.append(caption)
            frame_list = []
            del processed_batch
            del embedded_frames
            del caption_batch
    if len(frame_list) != 0:
        print("Processing batch %d of %d" %(batch_index, num_batches))
        processed_batch = np.array(frame_list,dtype=np.float32)
        embedded_frames = np.reshape(processed_batch,[-1,num_frames_per_clip,inception_base.num_end_units_v4])
        if beam_size==1:
            caption_batch = infer_util.generate_caption_batch(sess,embedded_frames,max_len=max_cap_len)
        else:
            caption_batch = infer_util.generate_caption_batch_beam(sess,beam_size,embedded_frames,max_len=max_cap_len)
        
        for cap in caption_batch:
            caption = {}
            caption["start_time"] = start_time
            caption["end_time"] = start_time+ (num_frames_per_clip/num_frames_per_sec)
            start_time = caption["end_time"]
            caption["caption"] = cap
            captions.append(caption)
    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file",
                        type=str)
    parser.add_argument("model_path", help="Path to the model checkpoint",
                        type=str)
    parser.add_argument("inception_path", help="Path to the inception checkpoint",
                        type=str)
    parser.add_argument("--num_fps", help="number of frames per second to be taken of video",
                        type=int, default=10)
    parser.add_argument("--len_clip", help="length of the clip (in sec) to split the video file into",
                        type=float, default=10)
    parser.add_argument("--cap_len", help="Maximum length of each caption",
                        type=int, default=20)
    parser.add_argument("--batch_size", help="Number of clips to be evaluated in one go",
                        type=int, default=64)
    parser.add_argument("--beam_size", help="Beam size to use when generating results",
                        type=int, default=3)
    args = parser.parse_args()


    num_frames_per_sec=args.num_fps
    num_frames_per_clip =int(args.num_fps*args.len_clip)
    max_cap_len=args.cap_len
    batch_size=args.batch_size
    beam_size=args.beam_size

    video_path = args.video_path
    srt_path = os.path.splitext(video_path)[0]+".srt"
    inception_checkpoint = args.inception_path
    model_checkpoint = args.model_path


    ## Building model
    data_config = configuration.DataConfig().config
    data_gen = Data_Generator(processed_video_dir = data_config["processed_video_dir"],
                            caption_file = data_config["caption_file"],
                            unique_freq_cutoff = data_config["unique_frequency_cutoff"],
                            max_caption_len = data_config["max_caption_length"])
    data_gen.load_vocabulary(data_config["caption_data_dir"])
    data_gen.load_dataset(data_config["caption_data_dir"])

    model_config = configuration.ModelConfig(data_gen).config
    model = Caption_Model( num_frames = num_frames_per_clip,
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
                        mode="inference",
                        rnn1_input_keep_prob=model_config["rnn1_input_keep_prob"],
                        rnn1_output_keep_prob=model_config["rnn1_output_keep_prob"],
                        rnn2_input_keep_prob=model_config["rnn2_input_keep_prob"],
                        rnn2_output_keep_prob=model_config["rnn2_output_keep_prob"],
                        embedding_file = model_config["embedding_file"]
                        )
    model.build()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess,model_checkpoint)

    
    infer_util = Inference(model,data_gen.word_to_idx,data_gen.idx_to_word)

    captions = generate_captions(sess,model,video_path,inception_checkpoint,infer_util,
                                num_frames_per_clip,num_frames_per_sec,max_cap_len,batch_size,beam_size)

    subtitles = pysrt.srtfile.SubRipFile()
    index = 1
    for caption in captions:
        sub = pysrt.srtitem.SubRipItem()
        multi_caption = "\n".join([cap[1] for cap in caption["caption"]])
        sub.start.seconds = caption["start_time"]
        sub.end.seconds = caption["end_time"]
        sub.text = multi_caption
        sub.index = index
        index += 1
        subtitles.append(sub)

    subtitles.save(srt_path)

if __name__=="__main__":
    main()
