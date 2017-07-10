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
        initial_state = sess.run(self.model.infer_hidden_state,
                                                    feed_dict = {self.model.rnn_input: encoded_video, self.model.is_training: False})
        return initial_state
    def inference_step(self,sess,input_feed,state_feed):
        feed_dict = {}
        feed_dict[self.model.state_feed] = state_feed
        feed_dict[self.model.word_input] = input_feed
        
        preds,next_state = sess.run(
                                    [self.model.infer_predictions,
                                     self.model.infer_last_state_decoding],
                                     feed_dict=feed_dict)
        return preds,next_state
    def generate_caption_batch(self,sess,video_batch,max_len=20):
        input_batch = np.array([[self.word_to_idx["<bos>"]]]*video_batch.shape[0])
        state = self.feed_video(sess,video_batch)
        # state is of shape num_layer * 2 * batch_size * hidden_size
        eos_batch = np.array([[self.word_to_idx["<eos>"]]]*video_batch.shape[0])
        finished_batch = np.array([[False]]*video_batch.shape[0])
        caption_generated = ["" for i in range(video_batch.shape[0])]
        loss = [0.0 for d in range(video_batch.shape[0])] 
        for i in range(max_len):
            pred,state = self.inference_step(sess,input_batch,state)
            input_batch = np.argmax(pred,axis=2)
            pred = np.squeeze(pred,axis=1)            
            pred_values = np.max(pred,axis=1)
            is_end = np.all(finished_batch)
            if is_end:
                break
            for idx,prob in enumerate(pred):
                if not finished_batch[idx]:
                    loss[idx] -= np.log(pred_values[idx])
                    if self.word_to_idx["<eos>"] != np.argmax(prob):
                        caption_generated[idx] += " "+self.idx_to_word[np.argmax(prob)]
            finished_batch = np.logical_or(input_batch==eos_batch,finished_batch)
            
        return [[(l/(len(i[1:].split(" ")) + 1),i[1:])] for l,i in zip(loss,caption_generated)]
    def generate_caption_batch_beam(self, sess, beam_size,video_batch,max_len=20):
        assert beam_size >= 2
        input_batch = np.array([[self.word_to_idx["<bos>"]]]*video_batch.shape[0])
        state = np.array(self.feed_video(sess,video_batch))
        # state is of shape num_layer * 2 * batch_size * hidden_size

        state = np.split(state,state.shape[2],axis=2)
        # state is list of size batch_size with each element of shape num_layer * 2 * 1 *hidden_size        
        batch_of_beams = []
        for i in range(video_batch.shape[0]):
            beam = [] # {st: , current_cap: , loss: , prev_word:}
            #for j in range(beam_size):
            beam.append({"st":state[i],
                             "current_cap":"" ,
                             "loss":0,
                             "prev_word":self.word_to_idx["<bos>"] })
            batch_of_beams.append(beam)

        completed_captions = [[] for d in range(video_batch.shape[0])]
        for i in range(max_len):
            beam_squared_list = [[] for d in range(video_batch.shape[0])] 
            for vv in range(len(batch_of_beams[0])):
                input_batch = [[video[vv]["prev_word"]] for video in batch_of_beams]
                state = [video[vv]["st"] for video in batch_of_beams] #batchsize * 2 * 2000
                state = np.concatenate(state,axis=2)
                pred,state = self.inference_step(sess,input_batch,state)
                state = np.array(self.feed_video(sess,video_batch))
                state = np.split(state,state.shape[2],axis=2)
                pred = np.squeeze(pred,axis=1)
                for j in range(len(beam_squared_list)):
                    #print("-------- Vid ------",j, " Current cap ", batch_of_beams[j][vv]["current_cap"] )
                    for pred_word in pred[j].argsort()[-beam_size:][::-1]:
                        #print(pred_word, " Pred Word-- ", self.idx_to_word[pred_word])
                        new_loss = batch_of_beams[j][vv]["loss"] - np.log(pred[j][pred_word])
                        if (pred_word == self.word_to_idx["<eos>"]) : 
                            completed_captions[j].append(( new_loss / (i+1) , # did  +1 avoiding divide by 0
                                                          batch_of_beams[j][vv]["current_cap"]))
                        else:
                            beam_squared_list[j].append({"st":state[j] ,
                                                         "current_cap":batch_of_beams[j][vv]["current_cap"] + 
                                                            " " + self.idx_to_word[pred_word],
                                                         "loss":new_loss,
                                                         "prev_word":pred_word })
                        
            for j in range(len(beam_squared_list)):
                beam_squared_list[j].sort(key = lambda x: x["loss"])
                batch_of_beams[j] = beam_squared_list[j][:beam_size].copy()
        for j in range(len(beam_squared_list)):
            completed_captions[j] += ([ ( t["loss"] / (max_len+1) ,  t["current_cap"] )   for t in beam_squared_list[j]])
            completed_captions[j].sort(key = lambda x: x[0])                                                                
        retList = []
        for t in completed_captions:
            to_append = []
            for tup in t[:beam_size]:
                to_append.append((tup[0],tup[1].strip()))
            retList.append(to_append)

        return retList  # List of batch size lists of beam size tuples with loss and caption
