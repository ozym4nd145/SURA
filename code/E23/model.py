from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import inception_base
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


class Caption_Model(object):
    def __init__(self, num_frames,
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
                 encoder_input_keep_prob,
                 encoder_output_keep_prob,
                 decoder_input_keep_prob,
                 decoder_output_keep_prob,
                 use_residual_encoder,
                 use_residual_decoder,
                 embedding_file,
                 attention_num_units,
                 attention_layer_units,
                 beam_width,
                 start_token_id,
                 end_token_id,
                 max_gradient_norm,
                 use_gradient_clipping,
                 ):
        assert mode in ["train", "val", "inference"]
        self.num_frames = num_frames
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.num_caption_unroll = num_caption_unroll
        self.num_last_layer_units = num_last_layer_units
        # self.image_embedding_size=image_embedding_size
        self.word_embedding_size = word_embedding_size
        self.hidden_size_lstm = hidden_size_lstm
        self.num_lstm_layer = num_lstm_layer
        self.vocab_size = vocab_size
        self.initializer = tf.random_uniform_initializer(
            minval=-initializer_scale, maxval=initializer_scale)
        self.learning_rate = learning_rate
        self.mode = mode
        self.encoder_input_keep_prob = encoder_input_keep_prob
        self.encoder_output_keep_prob = encoder_output_keep_prob
        self.decoder_input_keep_prob = decoder_input_keep_prob
        self.decoder_output_keep_prob = decoder_output_keep_prob
        self.use_residual_encoder = use_residual_encoder
        self.use_residual_decoder = use_residual_decoder
        self.embedding_file = embedding_file
        # self.attention_num_units = attention_num_units
        self.attention_num_units = self.hidden_size_lstm
        # self.attention_layer_units = attention_layer_units
        self.attention_layer_units = self.hidden_size_lstm
        self.summaries = []
        self.beam_width = beam_width
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.is_using_beamsearch = (mode=="inference" and (beam_width>1))
        self.max_gradient_norm = max_gradient_norm
        self.use_gradient_clipping = use_gradient_clipping

    def build(self):
        self.build_inputs()
        # self.build_inception_output()
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
                                        [None, self.num_frames, self.image_width,
                                            self.image_height, self.image_channels],
                                        name="video_input")
            self.video_mask = tf.placeholder(
                tf.int32, [None, self.num_frames], name="video_mask")
            self.caption_input = tf.placeholder(
                tf.int32, [None, self.num_caption_unroll + 1], name="caption_input")
            self.caption_mask = tf.placeholder(
                tf.int32, [None, self.num_caption_unroll], name="caption_mask")
            self.is_training = tf.placeholder(tf.bool, name='phase')
            lengths = tf.add(tf.reduce_sum(self.caption_mask, 1), 1)
            self.summaries.append(tf.summary.scalar(
                "caption_length/batch_min", tf.reduce_min(lengths)))
            self.summaries.append(tf.summary.scalar(
                "caption_length/batch_max", tf.reduce_max(lengths)))
            self.summaries.append(tf.summary.scalar(
                "caption_length/batch_mean", tf.reduce_mean(lengths)))

    def build_inception_output(self):
        interm_inputs = tf.reshape(
            self.processed_video_feed, [-1, self.image_width, self.image_height, self.image_channels])
        inception_output = inception_base.get_base_model(interm_inputs)
        self.inception_output = tf.reshape(
            inception_output, [-1, self.num_frames, self.num_last_layer_units])
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV4")

    def initialize_inception_pretrained(self, session, checkpoint_path):
        saver = tf.train.Saver(var_list=self.inception_variables)
        saver.restore(session, checkpoint_path)

    def build_encoder_cell(self):
        encoder_cells = self.encoder_cells
        if self.mode == "train":
            encoder_cells = [tf.contrib.rnn.DropoutWrapper(cell,
                                                            input_keep_prob=self.encoder_input_keep_prob,
                                                            output_keep_prob=self.encoder_output_keep_prob)
                                for cell in encoder_cells]
        if self.use_residual_encoder:
            encoder_cells = [tf.contrib.rnn.ResidualWrapper(cell) for cell in encoder_cells]
        
        return tf.contrib.rnn.MultiRNNCell(encoder_cells, state_is_tuple=True)
    
    def build_decoder_cell(self,outputs_encoding,last_state_encoding,callee):
        
        batch_size_tensor = tf.shape(outputs_encoding)[0]

        is_using_beamsearch = (self.is_using_beamsearch and callee == "infer_op")
        decoder_cells = [cell for cell in self.decoder_cells]
        if self.mode == "train":
            decoder_cells = [tf.contrib.rnn.DropoutWrapper(cell,
                                input_keep_prob=self.decoder_input_keep_prob,
                                output_keep_prob=self.decoder_output_keep_prob)
                                for cell in decoder_cells]
        if self.use_residual_decoder:
            decoder_cells = [tf.contrib.rnn.ResidualWrapper(cell) for cell in decoder_cells]
        
        with tf.variable_scope("Attention") as scope:
            # Tiling if using beam search
            if is_using_beamsearch:
                outputs_encoding = tf.contrib.seq2seq.tile_batch(
                    outputs_encoding, multiplier=self.beam_width)
                last_state_encoding = nest.map_structure(
                    lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), last_state_encoding)

            attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                                num_units=self.attention_num_units,  # depth of query mechanism
                                memory=outputs_encoding,
                                # memory_sequence_length= T,#tf.sequence_mask(seq_lengths, T), # masks false memories
                                normalize=True,  # normalize energy term
                                name='BahdanauAttention')

            def attn_decoder_input_fn(inputs, attention):
                if not self.use_residual_decoder:
                    return array_ops.concat([inputs, attention], -1)

                # Essential when use_residual=True
                return self.attention_dense_layer(array_ops.concat([inputs, attention], -1))

            decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(
                                cell=decoder_cells[-1],
                                attention_mechanism=attn_mech,
                                attention_layer_size=self.attention_layer_units,
                                cell_input_fn=attn_decoder_input_fn,
                                initial_cell_state=last_state_encoding[-1],
                                alignment_history=False,
                                name='Attention_Wrapper')

            batch_size = batch_size_tensor if not is_using_beamsearch else batch_size_tensor * self.beam_width
            initial_state = [state for state in last_state_encoding]
            attn_state = decoder_cells[-1].zero_state(batch_size=batch_size, dtype=tf.float32)
            initial_state[-1] = attn_state.clone(cell_state=initial_state[-1])

            decoder_initial_state = tuple(initial_state)
            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells, state_is_tuple=True)
            return decoder_cell,decoder_initial_state

    def build_embeddings(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("inputs") as scope:
                self.rnn_input = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.num_frames,
                                                    self.num_last_layer_units],
                                                name="rnn_input")
            if self.use_residual_encoder:
                with tf.variable_scope("image_encoding") as scope:
                    # converting to shape (batch_size*num_frames,dim_output_inception)
                    interm_input = tf.reshape(self.rnn_input, [-1, self.num_last_layer_units])
                    image_encoded = tf.contrib.layers.fully_connected(inputs=interm_input,
                                                            num_outputs=self.hidden_size_lstm,
                                                            activation_fn = None,
                                                            weights_initializer=self.initializer,
                                                            biases_initializer=self.initializer,
                                                            scope = scope)
                    # image_encoded = tf.contrib.layers.batch_norm(image_encoded,
                    #                                 center=True, scale=True,
                    #                                 is_training=self.is_training,
                    #                                 scope=scope)
                    # # image_encoded =  tf.nn.relu(h2, 'relu')

                    # converting to shape (batch_size,num_frames,embedding_size)
                    self.encoder_inputs = tf.reshape(
                        image_encoded, [-1, self.num_frames, self.hidden_size_lstm])
            else:
                self.encoder_inputs = self.rnn_input

            with tf.variable_scope("embedding") as scope:
                # Pretrained Variable Embedding
                self.special_emb = tf.get_variable(name="spc_embedding",
                                                shape=[4, self.word_embedding_size],
                                                initializer = self.initializer)
                self.pretrained_emb = tf.Variable(initial_value=np.load(self.embedding_file),
                                            dtype=tf.float32,name="glove_embedding",trainable=False)
                self.word_embedding = tf.concat([self.special_emb,self.pretrained_emb],axis=0, name="word_embedding")

                # Untrained embedding
                # self.word_embedding = tf.get_variable(dtype=tf.float32, name="word_embedding",
                #                                 shape=[self.vocab_size,
                #                                     self.word_embedding_size]
                #                                 )

            with tf.variable_scope("rnn_units"):
                self.encoder_cells = [tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size_lstm, reuse=tf.get_variable_scope().reuse) for _ in range(self.num_lstm_layer)]
                self.decoder_cells = [tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size_lstm, reuse=tf.get_variable_scope().reuse) for _ in range(self.num_lstm_layer)]

            self.caption_encoded = tf.nn.embedding_lookup(
                self.word_embedding, self.caption_input, name="lookup_caption_embedding")
            # caption_encoded is of shape(batch_size,self.num_caption_unroll+1,self.word_embedding_size)

            self.rnn_transform_layer = Dense(self.vocab_size,
                                             kernel_initializer=self.initializer,
                                             bias_initializer=self.initializer,
                                             dtype=tf.float32,
                                             name="rnn_transform_layer")
            self.word_transform_layer = Dense(self.hidden_size_lstm,
                                              kernel_initializer=self.initializer,
                                              bias_initializer=self.initializer,
                                              dtype=tf.float32,
                                              name="word_transform_layer")
            self.attention_dense_layer = Dense(self.hidden_size_lstm,
                                               kernel_initializer=self.initializer,
                                               bias_initializer=self.initializer,
                                               dtype=tf.float32,
                                               name='attn_input_feeding_decoder')

    def build_train_op(self):
        with tf.name_scope("Train"):
            ############################## TRAINING GRAPH #############################################
            batch_size_tensor = tf.shape(self.encoder_inputs)[0]
            
            with tf.name_scope("Encoding_stage"):
                sequence_length = tf.reduce_sum(self.video_mask, 1)
                self.train_encoder_cell = self.build_encoder_cell()
                with tf.variable_scope("Encoder") as scope:
                    outputs_encoding,last_state_encoding = tf.nn.dynamic_rnn(
                                                                cell=self.train_encoder_cell,
                                                                sequence_length=sequence_length,
                                                                inputs=self.encoder_inputs,
                                                                dtype=tf.float32)
                
            with tf.name_scope("Decoding_state"):
                sequence_length = tf.reduce_sum(self.caption_mask, 1)
                caption_needed = self.caption_encoded[:,:-1,:]

                if self.use_residual_decoder:
                    flattened_captions = tf.reshape(caption_needed,[-1, self.word_embedding_size])
                    caption_needed = self.word_transform_layer(flattened_captions)
                    caption_needed=tf.reshape(caption_needed, [-1, self.num_caption_unroll,self.hidden_size_lstm])


                self.train_decoder_cell,self.train_initial_decoder_state = self.build_decoder_cell(
                                                                outputs_encoding,last_state_encoding,"train_op")

                helper = tf.contrib.seq2seq.TrainingHelper(
                                inputs = caption_needed, # decoder inputs
                                sequence_length = sequence_length, # decoder input length
                                name = "decoder_training_helper")
                
                with tf.variable_scope("Decoder") as scope:
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                                    cell = self.train_decoder_cell,
                                    helper = helper,
                                    initial_state = self.train_initial_decoder_state,
                                    output_layer = self.rnn_transform_layer)
                    outputs_decoding, last_state_decoding, output_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)
            self.train_output_len = output_seq_len
            self.train_output_rnn = outputs_decoding.rnn_output
            ############################################################################################

    def build_inference_op(self):
        with tf.name_scope("Inference"):
            ############################## INFERENCE GRAPH #############################################
            with tf.name_scope("Video_Encoding"):
                self.infer_encoder_cell = self.build_encoder_cell()
                with tf.variable_scope("Encoder", reuse=True) as scope:
                    outputs_encoding, last_state_encoding = tf.nn.dynamic_rnn(
                        cell=self.infer_encoder_cell,
                        inputs=self.encoder_inputs,
                        dtype=tf.float32)

                batch_size_tensor = tf.shape(outputs_encoding)[0]
                self.infer_hidden_state = last_state_encoding
                self.infer_outputs_encoding = outputs_encoding

            with tf.name_scope("Decoding_step"):
                with tf.variable_scope("",reuse=True):
                    self.infer_decoder_cell,self.infer_initial_decoder_state = self.build_decoder_cell(
                                                            outputs_encoding, last_state_encoding,"infer_op")
                start_tokens = tf.fill(
                    dims=[batch_size_tensor], value=self.start_token_id)


                def embed_and_input_proj(inputs):
                    if self.use_residual_decoder:
                        return self.word_transform_layer(tf.nn.embedding_lookup(self.word_embedding, inputs))
                    else:
                        return tf.nn.embedding_lookup(self.word_embedding, inputs)
                with tf.variable_scope("Decoder", reuse=True) as scope:
                    
                    if not self.is_using_beamsearch:
                        decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                        end_token=self.end_token_id,
                                                                        embedding=embed_and_input_proj)
                        decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.infer_decoder_cell,
                                                                helper=decoding_helper,
                                                                initial_state=self.infer_initial_decoder_state,
                                                                output_layer=self.rnn_transform_layer)
                    else:
                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                                        cell=self.infer_decoder_cell,
                                                        embedding=embed_and_input_proj,
                                                        start_tokens=start_tokens,
                                                        end_token=self.end_token_id,
                                                        initial_state=self.infer_initial_decoder_state,
                                                        beam_width=self.beam_width,
                                                        output_layer=self.rnn_transform_layer)
                    outputs_decoding, last_state_decoding, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder, maximum_iterations=self.num_caption_unroll)
                self.infer_last_state_decoding = last_state_decoding
                self.infer_output_rnn = outputs_decoding
            ############################################################################################
    def build_output_logits(self):
            # gives directly the word ids, is of shape [batch_size,sentence_length,beam_size]
            if not self.is_using_beamsearch:
                self.infer_predictions = tf.expand_dims(self.infer_output_rnn.sample_id,-1)
            else:
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

            if self.use_gradient_clipping:
                trainable_params = tf.trainable_variables()
                # Compute gradients of loss w.r.t. all trainable variables
                gradients = tf.gradients(self.batch_loss, trainable_params)

                # Clip gradients by a given maximum_gradient_norm
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

                # Update the model
                with tf.control_dependencies(update_ops):
                    self.train_step = optimizer.apply_gradients(zip(gradients, trainable_params),
                                                                global_step=self.global_step)
            else:
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
        
