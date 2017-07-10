from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class Inference(object):
    def __init__ (self,model,word_to_idx,idx_to_word):
        self.model=model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
    def generate_caption_batch_beam(self,sess,video_batch):
        feed_dict={}
        feed_dict[self.model.video_mask] = np.ones([video_batch.shape[0],video_batch.shape[1]],dtype=np.int32)
        feed_dict[self.model.rnn_input] = video_batch
        feed_dict[self.model.is_training] = False
        predictions = sess.run(self.model.infer_predictions,feed_dict = feed_dict)
        batch_captions = [[] for _ in range(video_batch.shape[0])]
        for video_caption,preds in zip(batch_captions,predictions):
            for beam in range(self.model.beam_width):
                caption = []
                for pred in preds:
                    if pred[beam] == self.word_to_idx["<eos>"]:
                        break
                    caption.append(self.idx_to_word[pred[beam]])
                video_caption.append((0," ".join(caption)))
        return batch_captions
