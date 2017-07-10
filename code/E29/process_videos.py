from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
from scipy import misc
import time
import inception_base
import cv2

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("num_frames", 100,
                       "Number of frames after sampling.")
tf.flags.DEFINE_string("video_dir", None,
                       "Directory containing the video files.")
tf.flags.DEFINE_string("output_dir", None,
                       "Directory containing the output files.")
tf.flags.DEFINE_string("checkpoint_path", None,
                       "Checkpoint containing the InceptionV4 model.")

def read_video(video_path,num_frames):
    cap  = cv2.VideoCapture( video_path )
    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1
    frame_list = np.array(frame_list)
    if frame_count > num_frames:
        frame_indices = np.linspace(0, frame_count-1, num=num_frames, endpoint=True).astype(int)
        frame_list = frame_list[frame_indices]
    frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
    frame_list_resized = [misc.imresize(frame,[299,299,3]) for frame in frame_list]
    frame_list_processed = [((2*(frame.astype(np.float32) / 255 ))-1) for frame in frame_list_resized]
    frame_list = np.array(frame_list_processed)
    return frame_list

def main(unused_args):
    assert FLAGS.video_dir, "--video_dir is required"
    assert FLAGS.output_dir, "--output_dir is required"
    assert FLAGS.checkpoint_path, "--checkpoint_path is required"
    
    ## Building model
    image_feed = tf.placeholder(dtype=tf.float32,shape=[None,299,299,3],name="image_feed")
    inception_output = inception_base.get_base_model(image_feed)
    inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV4")
    
    ## Getting files
    supported_extensions = [".mp4",".avi"]
    input_files = []

    for root, dirs, files in os.walk(FLAGS.video_dir):
        input_files += [os.path.join(root,fl) for fl in files if os.path.splitext(fl)[1] in supported_extensions]

    # print(input_files[:10])    
    # print(FLAGS.output_dir)
    # print(FLAGS.checkpoint_path)
    # print(FLAGS.video_dir)
    output_files = [os.path.join(FLAGS.output_dir,fl[len(FLAGS.video_dir)+1:]) for fl in input_files]
    # print(output_files[:10])
    output_dirs = list(set([os.path.dirname(fl) for fl in output_files]))

    for dr in output_dirs:
        print(dr)
        os.makedirs(dr,exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(var_list=inception_variables)
        saver.restore(sess,FLAGS.checkpoint_path)

        for i in range(len(input_files)):
            if i%100 == 0:
                print("Processed 100 videos, currently at "+str(i)+" of "+str(len(input_files))+"\n")
            if os.path.exists(output_files[i]+".npy"):
                print("Skipping, already exists: "+(output_files[i]+".npy"))
                continue
            video = read_video(input_files[i],FLAGS.num_frames)
            embedded_frames = sess.run(inception_output,feed_dict={image_feed:video})
            if(embedded_frames.shape[0] < FLAGS.num_frames):
                num_zeros = (FLAGS.num_frames-embedded_frames.shape[0])
                end_zeros = np.zeros([num_zeros,embedded_frames.shape[1]],dtype=np.float32)
                mask = np.asarray([1]*embedded_frames.shape[0]+[0]*(num_zeros),dtype=np.uint8)                
                embedded_frames = np.concatenate([embedded_frames,end_zeros],axis=0)
            else:
                mask = np.ones([FLAGS.num_frames],dtype=np.uint8)
            np.save(output_files[i]+".npy",embedded_frames)
            np.save(output_files[i]+".mask.npy",mask)

if __name__ == "__main__":
  tf.app.run()
