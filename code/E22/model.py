from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import inception_base
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

class Caption_Model(object):
    def __init__(self,num_frames,
                 image_width,
                 image_height,
                 image_channels,
                 num_caption_unroll,
                 num_last_layer_units,
                 word_embedding_size,
                 hidden_size_lstm,
                 num_lstm_layer,
                 vocab_size,
                 initializer_scale,
                 learning_rate,
                 mode,
                 rnn1_input_keep_prob ,
                 rnn1_output_keep_prob ,
                 rnn2_input_keep_prob ,
                 rnn2_output_keep_prob ,
                 embedding_file,
                 attention_num_units,
                 attention_layer_units,
                 beam_width,
                 start_token_id,
                 end_token_id,
                 ):
        assert mode in ["train","val","inference"]
        self.num_frames=num_frames
        self.image_width=image_width
        self.image_height=image_height
        self.image_channels=image_channels
        self.num_caption_unroll=num_caption_unroll
        self.num_last_layer_units=num_last_layer_units
        # self.image_embedding_size=image_embedding_size
        self.word_embedding_size=word_embedding_size
        self.hidden_size_lstm=hidden_size_lstm
        self.num_lstm_layer = num_lstm_layer
        self.vocab_size=vocab_size
        self.initializer = tf.random_uniform_initializer(minval=-initializer_scale,maxval=initializer_scale)
        self.learning_rate=learning_rate
        self.mode=mode
        self.rnn1_input_keep_prob=rnn1_input_keep_prob
        self.rnn1_output_keep_prob=rnn1_output_keep_prob
        self.rnn2_input_keep_prob=rnn2_input_keep_prob
        self.rnn2_output_keep_prob=rnn2_output_keep_prob
        self.embedding_file = embedding_file
        self.attention_num_units = attention_num_units
        self.attention_layer_units = attention_layer_units
        self.summaries = []
        self.beam_width = beam_width
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
    def build(self):
        self.build_inputs()
        #self.build_inception_output()
        self.build_embeddings()
        self.build_train_op()
        self.build_inference_op()
        self.build_output_logits()
        self.build_loss()
        self.setup_global_step()
        self.build_optimizer()
        
    def build_inputs(self):
        with tf.variable_scope("inputs") as scope:
            self.processed_video_feed = tf.placeholder(tf.float32,
                                        [None,self.num_frames,self.image_width,self.image_height,self.image_channels],
                                        name="video_input")
            self.video_mask = tf.placeholder(tf.int32, [None, self.num_frames],name="video_mask")
            self.caption_input = tf.placeholder(tf.int32,[None,self.num_caption_unroll+1],name="caption_input")
            self.caption_mask = tf.placeholder(tf.int32,[None,self.num_caption_unroll],name="caption_mask")
            self.is_training = tf.placeholder(tf.bool, name='phase')
            lengths = tf.add(tf.reduce_sum(self.caption_mask, 1), 1)
            self.summaries.append(tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths)))
            self.summaries.append(tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths)))
            self.summaries.append(tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths)))

    def build_inception_output(self):
        interm_inputs = tf.reshape(self.processed_video_feed,[-1,self.image_width,self.image_height,self.image_channels])
        inception_output = inception_base.get_base_model(interm_inputs)
        self.inception_output = tf.reshape(inception_output,[-1,self.num_frames,self.num_last_layer_units])
        self.inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV4") 
    def initialize_inception_pretrained(self,session,checkpoint_path):
        saver = tf.train.Saver(var_list=self.inception_variables)
        saver.restore(session,checkpoint_path)
                                           
    def build_embeddings(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("inputs") as scope:
                self.rnn_input = tf.placeholder(dtype=tf.float32,
                                                shape = [None,self.num_frames,self.num_last_layer_units],
                                                name="rnn_input")

            # converting to shape (batch_size*num_frames,dim_output_inception)
            interm_input = tf.reshape(self.rnn_input,[-1,self.num_last_layer_units])
            
            # with tf.variable_scope("image_encoding") as scope:
            #     h1 = tf.contrib.layers.fully_connected(inputs=interm_input,
            #                                             num_outputs=self.image_embedding_size,
            #                                             activation_fn = None,
            #                                             weights_initializer=self.initializer,
            #                                             biases_initializer=self.initializer,
            #                                             scope = scope)
            #     image_encoded = tf.contrib.layers.batch_norm(h1, 
            #                                     center=True, scale=True, 
            #                                     is_training=self.is_training,
            #                                     scope=scope)
            #     # image_encoded =  tf.nn.relu(h2, 'relu')
                
            #     # converting to shape (batch_size,num_frames,embedding_size)
            #     self.image_encoded = tf.reshape(image_encoded,[-1,self.num_frames,self.image_embedding_size])
            
            with tf.variable_scope("embedding") as scope:
                ## Pretrained Variable Embedding
                self.special_emb = tf.get_variable(name="spc_embedding",
                                                shape=[4, self.word_embedding_size],
                                                initializer = self.initializer)
                self.pretrained_emb = tf.Variable(initial_value=np.load(self.embedding_file),
                                            dtype=tf.float32,name="glove_embedding",trainable=True)
                self.word_embedding = tf.concat([self.special_emb,self.pretrained_emb],axis=0, name="word_embedding")
                
                ## Pretrained Constant Embedding
                # self.word_emb = tf.constant(np.load(self.embedding_file),
                #                             shape=[self.vocab_size-4, self.word_embedding_size],
                #                             dtype=tf.float32,name="glove_embedding",
                #                             verify_shape=True)
                # self.word_emb = tf.placeholder(dtype=tf.float32, name="word_embedding",
                #                                 shape=[self.vocab_size-4, self.word_embedding_size]
                #                                 )
                
                ## Untrained embedding
                # self.word_embedding = tf.get_variable(dtype=tf.float32, name="word_embedding",
                #                                 shape=[self.vocab_size, self.word_embedding_size]
                #                                 )
            
            with tf.variable_scope("rnn_units"):
                self.cell_1 = [tf.contrib.rnn.BasicLSTMCell(self.hidden_size_lstm,reuse=tf.get_variable_scope().reuse) for _ in range(self.num_lstm_layer)]
                self.cell_2 = [tf.contrib.rnn.BasicLSTMCell(self.hidden_size_lstm,reuse=tf.get_variable_scope().reuse) for _ in range(self.num_lstm_layer)]
            
            self.caption_encoded = tf.nn.embedding_lookup(self.word_embedding,self.caption_input,name="lookup_caption_embedding")
            #caption_encoded is of shape(batch_size,self.num_caption_unroll+1,self.word_embedding_size)

            self.rnn_transform_layer = Dense(self.vocab_size)
    def build_inference_op(self):
        with tf.name_scope("Inference"):
            cell_1 = tf.contrib.rnn.MultiRNNCell(self.cell_1,state_is_tuple=True)
            cell_2 = tf.contrib.rnn.MultiRNNCell(self.cell_2,state_is_tuple=True)
            
            ############################## INFERENCE GRAPH #############################################
            with tf.name_scope("Video_Encoding"):
                with tf.variable_scope("RNN_1",reuse=True) as scope:
                    outputs_encoding, last_state_encoding = tf.nn.dynamic_rnn(
                                                cell=cell_1,
                                                # inputs=self.image_encoded,
                                                inputs=self.rnn_input,
                                                dtype=tf.float32)

                batch_size_tensor = tf.shape(outputs_encoding)[0]
                self.infer_hidden_state = last_state_encoding

            with tf.name_scope("Decoding_step"):
                encoder_outputs = tf.contrib.seq2seq.tile_batch(
                                    outputs_encoding, multiplier=self.beam_width)
                encoder_last_state = nest.map_structure(
                            lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), last_state_encoding)
                with tf.variable_scope("Attention",reuse=True) as scope:
                    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units = self.attention_num_units, # depth of query mechanism
                                    memory = encoder_outputs, # hidden states to attend (output of RNN)
                                    #memory_sequence_length= T,#tf.sequence_mask(seq_lengths, T), # masks false memories
                                    normalize=False, # normalize energy term
                                    name='BahdanauAttention')
                    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    cell = cell_2,# Instance of RNNCell
                                    attention_mechanism = attn_mech, # Instance of AttentionMechanism
                                    attention_layer_size = self.attention_layer_units, # Int, depth of attention (output) tensor
                                    alignment_history=True, # whether to store history in final output
                                    output_attention=False,
                                    name="attention_wrapper")

                start_tokens = tf.fill(dims=[batch_size_tensor],value=self.start_token_id)
                attn_zero_state = attn_cell.zero_state(batch_size=batch_size_tensor*self.beam_width,dtype=tf.float32)
                initial_decoder_state = attn_zero_state.clone(cell_state=encoder_last_state)

                with tf.variable_scope("RNN_2",reuse=True) as scope:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                    cell=attn_cell,
                                    embedding=self.word_embedding,
                                    start_tokens=start_tokens,
                                    end_token=self.end_token_id,
                                    initial_state=initial_decoder_state,
                                    beam_width=self.beam_width,
                                    output_layer=self.rnn_transform_layer)
                    outputs_decoding, last_state_decoding, _ = tf.contrib.seq2seq.dynamic_decode(
                                                            decoder,maximum_iterations=self.num_caption_unroll)
                self.infer_last_state_decoding = last_state_decoding
                self.infer_output_rnn = outputs_decoding
            ############################################################################################
    def build_train_op(self):
        with tf.name_scope("Train"):
            ############################## TRAINING GRAPH #############################################
            # batch_size_tensor = tf.shape(self.image_encoded)[0]
            batch_size_tensor = tf.shape(self.rnn_input)[0]

            cell_1 = self.cell_1
            cell_2 = self.cell_2

            if self.mode=="train":
                cell_1 = [ tf.contrib.rnn.DropoutWrapper( cell,
                                                        input_keep_prob=self.rnn1_input_keep_prob,
                                                        output_keep_prob=self.rnn1_output_keep_prob)
                                                        for cell in cell_1]
                cell_2 = [ tf.contrib.rnn.DropoutWrapper( cell,
                                                        input_keep_prob=self.rnn2_input_keep_prob,
                                                        output_keep_prob=self.rnn2_output_keep_prob)
                                                        for cell in cell_2]

            cell_1 = tf.contrib.rnn.MultiRNNCell(cell_1,state_is_tuple=True)
            cell_2 = tf.contrib.rnn.MultiRNNCell(cell_2,state_is_tuple=True)
            
            with tf.name_scope("Encoding_stage"):
                sequence_length = tf.reduce_sum(self.video_mask, 1)
                with tf.variable_scope("RNN_1") as scope:            
                    outputs_encoding,last_state_encoding = tf.nn.dynamic_rnn(
                                                                cell=cell_1,
                                                                sequence_length=sequence_length,
                                                                # inputs=self.image_encoded,
                                                                inputs=self.rnn_input,
                                                                dtype=tf.float32)
                
            with tf.name_scope("Decoding_state"):
                
                sequence_length = tf.reduce_sum(self.caption_mask, 1)
                caption_needed = self.caption_encoded[:,:-1,:]

                with tf.variable_scope("Attention") as scope:
                    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units = self.attention_num_units, # depth of query mechanism
                                    memory = outputs_encoding, # hidden states to attend (output of RNN)
                                    #memory_sequence_length= T,#tf.sequence_mask(seq_lengths, T), # masks false memories
                                    normalize=False, # normalize energy term
                                    name='BahdanauAttention')
                    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    cell = cell_2,# Instance of RNNCell
                                    attention_mechanism = attn_mech, # Instance of AttentionMechanism
                                    attention_layer_size = self.attention_layer_units, # Int, depth of attention (output) tensor
                                    alignment_history=True, # whether to store history in final output
                                    output_attention=False,
                                    name="attention_wrapper")
                
                helper = tf.contrib.seq2seq.TrainingHelper(
                                inputs = caption_needed, # decoder inputs
                                sequence_length = sequence_length, # decoder input length
                                name = "decoder_training_helper")
                
                attn_zero_state = attn_cell.zero_state(batch_size=batch_size_tensor,dtype=tf.float32)
                initial_decoder_state = attn_zero_state.clone(cell_state=last_state_encoding)

                with tf.variable_scope("RNN_2") as scope:
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                                    cell = attn_cell,
                                    helper = helper,
                                    initial_state = initial_decoder_state,
                                    output_layer = self.rnn_transform_layer)
                    outputs_decoding, last_state_decoding, output_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)
            self.STATE = last_state_decoding
            self.train_output_len = output_seq_len
            self.train_output_rnn = outputs_decoding.rnn_output
            ############################################################################################
    def build_output_logits(self):
            # gives directly the word ids, is of shape [batch_size,sentence_length,beam_size]
            self.infer_predictions = self.infer_output_rnn.predicted_ids
            self.train_predictions = tf.nn.softmax(self.train_output_rnn)

    def build_loss(self):
        with tf.variable_scope("loss") as scope:
            self.max_decoder_length = tf.reduce_max(self.train_output_len)
            correct_predictions = self.caption_input[:,1:self.max_decoder_length+1] #shape = (batch_size,self.max_decoder_length) int32
            self.mask = tf.sequence_mask(lengths=tf.reduce_sum(self.caption_mask, 1),
                                    maxlen=self.max_decoder_length,dtype=tf.float32,name="caption_mask")
            self.train_pred = tf.argmax(self.train_output_rnn, axis=-1,
                                                    name='decoder_pred_train')
            self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_output_rnn,
                                                                targets=correct_predictions,
                                                                weights=self.mask,
                                                                average_across_timesteps=True,
                                                                average_across_batch=True)
            self.summaries.append(tf.summary.scalar('Batch_Loss', self.batch_loss))
            self.summaries.append(tf.summary.histogram('Loss_Histogram', self.batch_loss))
            for var in tf.trainable_variables():
                self.summaries.append(tf.summary.histogram("parameters/" + var.op.name, var))

    def build_optimizer(self):
        with tf.name_scope("Optimizer") as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            with tf.control_dependencies(update_ops):
                self.train_step = optimizer.minimize(self.batch_loss,global_step=self.global_step)
    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step
        
