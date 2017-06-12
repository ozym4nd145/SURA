from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import json
import collections
import string
import re
import pickle

class Data_Generator(object):
    def __init__ (self,processed_video_dir,caption_file,unique_freq_cutoff=2,max_caption_len=21):
        self.processed_video_dir=processed_video_dir
        self.caption_file=caption_file
        self.unique_frequency_cutoff = unique_freq_cutoff
        self.max_caption_len=max_caption_len
        self.vocab_size = 0
        self.vocabulary=None
        self.word_to_idx = None
        self.idx_to_word = None
        self.dataset = {"train":[],"val":[],"test":[]}
        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"
        self.all_words = [] #all the words in the caption dataset
        
        self.iter_last_index = {"test":None , "train":None, "val":None}
        self.iter_per_epoch = {"test":None , "train":None, "val":None}
        self.iterators = {"test":None , "train":None, "val":None}
        self.batch_size = {"test":None , "train":None, "val":None}
    def build_basic_dataset(self):
        TEST_DIR = os.path.join(self.processed_video_dir,"test")
        TRAIN_DIR = os.path.join(self.processed_video_dir,"train")
        VAL_DIR = os.path.join(self.processed_video_dir,"val")
        
        with open(self.caption_file,"r") as f:
            caption_file = json.loads(f.read())
        
        test_video_files = set(os.listdir(TEST_DIR))
        train_video_files = set(os.listdir(TRAIN_DIR))
        val_video_files = set(os.listdir(VAL_DIR))
        
        punctuation_process_table = str.maketrans({key: " " for key in string.punctuation})
        
        for i in range(len(caption_file['sentences'])):
            real_caption = caption_file['sentences'][i]['caption'].lower()
            caption = real_caption.translate(punctuation_process_table)
            caption = re.sub("\s\s+" , " ", caption)
            words = [t for t in caption.split(" ") if len(t) > 0 ]
            caption = " ".join(words)
            self.all_words += words
            name = caption_file['sentences'][i]['video_id']+".avi.npy"
            if name in test_video_files:
                self.dataset["test"].append({"file_name":name,"caption":caption,"real_caption": real_caption,"path": os.path.join(TEST_DIR,name)})
            if name in train_video_files:
                self.dataset["train"].append({"file_name":name,"caption":caption,"real_caption": real_caption,"path": os.path.join(TRAIN_DIR,name)})
            if name in val_video_files:
                self.dataset["val"].append({"file_name":name,"caption":caption,"real_caption": real_caption,"path": os.path.join(VAL_DIR,name)})
        
    def build_vocabulary(self):
        words = collections.Counter(self.all_words)
        count_pairs = sorted(words.items(), key=lambda x: -x[1])
        
        #PAD should be first element as we want its id to be 0
        additional_words = [self.PAD,self.BOS,self.EOS,self.UNK]
        
        self.word_to_idx = {additional_words[i]:i for i in range(len(additional_words))}
        self.idx_to_word = {i: additional_words[i] for i in range(len(additional_words))}
        self.vocabulary = set(additional_words)
        
        for i in range(len(count_pairs)):
            if count_pairs[i][1] <= self.unique_frequency_cutoff :
                wrd = self.UNK
                self.word_to_idx[count_pairs[i][0]] = self.word_to_idx[wrd]
            else:
                wrd = count_pairs[i][0]
                self.word_to_idx[wrd] = len(self.vocabulary)
                self.idx_to_word[len(self.vocabulary)] = wrd
                self.vocabulary.add(wrd)
                
        self.vocab_size = len(self.vocabulary)
    def build_process_captions(self):
        max_len = self.max_caption_len
        for i in (self.dataset).keys():
            data = self.dataset[i]
            for j in range(len(data)):
                words = [self.BOS]+data[j]['caption'].split(" ")+[self.EOS]
                processed_caption = []
                if len(words) < max_len:
                    processed_caption += words
                    processed_caption += [self.PAD]*(max_len - len(processed_caption))
                else:
                    processed_caption += words[:max_len-1]
                    processed_caption += [self.EOS]
                data[j]['processed_caption'] = " ".join(processed_caption)
                data[j]['indexed_caption'] = np.asarray([self.word_to_idx[i] for i in processed_caption],np.int32)
                target =(data[j]['indexed_caption'][1:])
                data[j]['caption_mask']=np.ndarray.astype((target!=0),np.int32)
    def build_dataset(self):
        self.build_basic_dataset()
        self.build_process_captions()
    def save_vocabulary(self,path="./"):
        os.makedirs(path,exist_ok=True)
        word2idx_path = os.path.join(path,"word_to_idx.json")
        idx2word_path = os.path.join(path,"idx_to_word.json")
        vocabulary_path = os.path.join(path,"vocabulary.json")
        with open(word2idx_path,"w") as fl:
            fl.write(json.dumps(self.word_to_idx))
        with open(idx2word_path,"w") as fl:
            fl.write(json.dumps(self.idx_to_word))
        with open(vocabulary_path,"w") as fl:
            fl.write(json.dumps(list(self.vocabulary)))
    def load_dataset(self,path):
        dataset_path = os.path.join(path,"dataset_info.pkl")
        with open(dataset_path,"rb") as fl:
            self.dataset = pickle.load(fl)
    def save_dataset(self,path):
        os.makedirs(path,exist_ok=True)
        dataset_path = os.path.join(path,"dataset_info.pkl")
        with open(dataset_path,"wb") as fl:
            pickle.dump(self.dataset, fl)
    def load_vocabulary(self,path):
        word2idx_path = os.path.join(path,"word_to_idx.json")
        idx2word_path = os.path.join(path,"idx_to_word.json")
        vocabulary_path = os.path.join(path,"vocabulary.json")
        with open(word2idx_path,"r") as fl:
            self.word_to_idx = json.loads(fl.read())
        with open(idx2word_path,"r") as fl:
            temp = json.loads(fl.read())
            self.idx_to_word = {int(k):temp[k] for k in temp.keys()}
        with open(vocabulary_path,"r") as fl:
            self.vocabulary = set(json.loads(fl.read()))
        self.vocab_size = len(self.vocabulary)
    def init_batch(self,batch_size,dataset_to_use):
        assert dataset_to_use in ["test","train","val"]  
        np.random.shuffle(self.dataset[dataset_to_use])
        self.iterators[dataset_to_use] = 0
        self.iter_last_index[dataset_to_use] = len(self.dataset[dataset_to_use])
        self.batch_size[dataset_to_use] = batch_size
        self.iter_per_epoch[dataset_to_use] = int((len(self.dataset[dataset_to_use])+batch_size-1)/batch_size)
    def get_next_batch(self,dataset_to_use):
        assert dataset_to_use in ["test","train","val"]
        iterator = self.iterators[dataset_to_use]
        data = self.dataset[dataset_to_use]
        if(iterator == self.iter_last_index[dataset_to_use]):
            self.iterators[dataset_to_use] = 0
            iterator = 0
            np.random.shuffle(self.dataset[dataset_to_use])
        batch_size = min(self.batch_size[dataset_to_use],self.iter_last_index[dataset_to_use]-iterator)
        batch = {"video":[],"path":[],"file":[],"caption":[],"caption_mask":[],"indexed_caption":[]}
        batch["video"] = np.asarray([np.load(data[i]['path']) for i in range(iterator,iterator+batch_size)])
        batch["indexed_caption"] = np.asarray([data[i]['indexed_caption'] for i in range(iterator,iterator+batch_size)])
        batch["caption_mask"] = np.asarray([data[i]['caption_mask'] for i in range(iterator,iterator+batch_size)])
        batch["path"] = [data[i]['path'] for i in range(iterator,iterator+batch_size)]
        batch["file"] = [data[i]['file_name'] for i in range(iterator,iterator+batch_size)]
        batch["caption"] = [data[i]['caption'] for i in range(iterator,iterator+batch_size)]
        self.iterators[dataset_to_use] += batch_size
        return batch
    
    
    