from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class Inference(object):
    def __init__ (self,model,word_to_idx,idx_to_word):
        self.model=model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
    def feed_video(self, sess, encoded_video):
        initial_state_1,initial_state_2 = sess.run([self.model.infer_hidden_state_1,
                                                    self.model.infer_hidden_state_2,],
                                                    feed_dict = {self.model.rnn_input: encoded_video})
        return [initial_state_1,initial_state_2]
    def inference_step(self,sess,input_feed,state_feed):
        feed_dict = {}
        feed_dict[self.model.state_feed_1] = state_feed[0]
        feed_dict[self.model.state_feed_2] = state_feed[1]
        feed_dict[self.model.word_input] = input_feed
        
        preds,next_state_1,next_state_2 = sess.run(
                                    [self.model.infer_predictions,
                                     self.model.infer_last_state_l1,
                                     self.model.infer_last_state_l2],
                                     feed_dict=feed_dict)
        return preds,[next_state_1,next_state_2]
    def generate_caption_batch(self,sess,video_batch,max_len=20):
        input_batch = np.array([[self.word_to_idx["<bos>"]]]*video_batch.shape[0])
        state = self.feed_video(sess,video_batch)
        eos_batch = np.array([[self.word_to_idx["<eos>"]]]*video_batch.shape[0])
        finished_batch = np.array([[False]]*video_batch.shape[0])
        caption_generated = ["" for i in range(video_batch.shape[0])]

        for i in range(max_len):
            pred,state = self.inference_step(sess,input_batch,state)
            input_batch = np.argmax(pred,axis=2)
            finished_batch = np.logical_or(input_batch==eos_batch,finished_batch)
            is_end = np.all(finished_batch)
            pred = np.squeeze(pred)
            if is_end:
                break
            for j,p in enumerate(pred):
                if not finished_batch[j]:
                    caption_generated[j] += " "+self.idx_to_word[np.argmax(p)]
        return [i[1:] for i in caption_generated]
    
