from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import inception_base

class Model_S2VT(object):
    def __init__(self,num_frames=100,
                 image_width=299,
                 image_height=299,
                 image_channels=3,
                 num_caption_unroll=20,
                 num_last_layer_units=inception_base.num_end_units_v4,
                 image_embedding_size=1000,
                 word_embedding_size=1000,
                 hidden_size_lstm1=500,
                 hidden_size_lstm2=500,
                 vocab_size=18000,
                 initializer_scale=1,
                 learning_rate=1e-3,
                 mode="train",
                 rnn1_input_keep_prob = 1.0,
                 rnn1_output_keep_prob = 1.0,
                 rnn2_input_keep_prob = 1.0,
                 rnn2_output_keep_prob = 1.0,
                 ):
        assert mode in ["train","val","inference"]
        self.num_frames=num_frames
        self.image_width=image_width
        self.image_height=image_height
        self.image_channels=image_channels
        self.num_caption_unroll=num_caption_unroll
        self.num_last_layer_units=num_last_layer_units
        self.image_embedding_size=image_embedding_size
        self.word_embedding_size=word_embedding_size
        self.hidden_size_lstm1=hidden_size_lstm1
        self.hidden_size_lstm2=hidden_size_lstm2
        self.vocab_size=vocab_size
        self.initializer = tf.random_uniform_initializer(minval=-initializer_scale,maxval=initializer_scale)
        self.learning_rate=learning_rate
        self.mode=mode
        self.rnn1_input_keep_prob=rnn1_input_keep_prob
        self.rnn1_output_keep_prob=rnn1_output_keep_prob
        self.rnn2_input_keep_prob=rnn2_input_keep_prob
        self.rnn2_output_keep_prob=rnn2_output_keep_prob
        self.summaries = []
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
            self.video_mask = tf.placeholder(tf.float32, [None, self.num_frames],name="video_mask")
            self.caption_input = tf.placeholder(tf.int32,[None,self.num_caption_unroll+1],name="caption_input")
            self.word_input = tf.placeholder(tf.int32,[None,1],name="word_input_inference")
            self.caption_mask = tf.placeholder(tf.float32,[None,self.num_caption_unroll],name="caption_mask")
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
        with tf.variable_scope("inputs") as scope:
            self.rnn_input = tf.placeholder(dtype=tf.float32,
                                            shape = [None,self.num_frames,self.num_last_layer_units],
                                            name="rnn_input")

        # converting to shape (batch_size*num_frames,dim_output_inception)
        interm_input = tf.reshape(self.rnn_input,[-1,self.num_last_layer_units])
        
        with tf.variable_scope("image_encoding") as scope:
            h1 = tf.contrib.layers.fully_connected(inputs=interm_input,
                                                    num_outputs=self.image_embedding_size,
                                                    activation_fn = None,
                                                    weights_initializer=self.initializer,
                                                    biases_initializer=self.initializer,
                                                    scope = scope)
            h2 = tf.contrib.layers.batch_norm(h1, 
                                            center=True, scale=True, 
                                            is_training=self.is_training,
                                            scope=scope)
            image_encoded =  tf.nn.relu(h2, 'relu')
            
            # converting to shape (batch_size,num_frames,embedding_size)
            self.image_encoded = tf.reshape(image_encoded,[-1,self.num_frames,self.image_embedding_size])
        
        with tf.variable_scope("embedding") as scope:
            self.word_emb = tf.get_variable(name="word_embedding",
                                            shape=[self.vocab_size, self.word_embedding_size],
                                            initializer = self.initializer)
        
        with tf.variable_scope("rnn_units"):
            self.cell_1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size_lstm1,reuse=tf.get_variable_scope().reuse)
            self.cell_2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size_lstm2,reuse=tf.get_variable_scope().reuse)
        
        self.caption_encoded = tf.nn.embedding_lookup(self.word_emb,self.caption_input,name="lookup_caption_embedding")
        #caption_encoded is of shape(batch_size,self.num_caption_unroll+1,self.word_embedding_size)

        self.word_encoded = tf.nn.embedding_lookup(self.word_emb,self.word_input,name="lookup_word_embedding")
        #word_encoded is of shape(batch_size,1,self.word_embedding_size)
        
    def build_inference_op(self):
        with tf.name_scope("Inference"):
            ############################## INFERENCE GRAPH #############################################
            with tf.name_scope("Video_Encoding"):
                with tf.variable_scope("RNN_1",reuse=True) as scope:
                    outputs_l1, last_state_l1 = tf.nn.dynamic_rnn(
                                                cell=self.cell_1,
                                                inputs=self.image_encoded,
                                                dtype=tf.float32)
                
                batch_size_tensor = tf.shape(outputs_l1)[0]
                caption_padding = tf.zeros([batch_size_tensor,self.num_frames,self.word_embedding_size])
                inputs_l2 = tf.concat([caption_padding,outputs_l1],axis=2)

                with tf.variable_scope("RNN_2",reuse=True) as scope:
                    outputs_l2,last_state_l2 = tf.nn.dynamic_rnn(
                                                cell=self.cell_2,
                                                inputs=inputs_l2,dtype=tf.float32)
                
                self.infer_encoding_output_1 = outputs_l1
                self.infer_encoding_input_1 = self.image_encoded
                self.infer_encoding_output_2 = outputs_l2
                self.infer_encoding_input_2 = inputs_l2
                
                self.infer_hidden_state_1 = tf.concat(last_state_l1,axis=1,name="hidden_state_1_concat")
                self.infer_hidden_state_2 = tf.concat(last_state_l2,axis=1,name="hidden_state_2_concat")


            self.state_feed_1 = tf.placeholder(dtype=tf.float32,
                                          shape=[None, sum(self.cell_1.state_size)],
                                          name="state_feed_1")
            self.state_feed_2 = tf.placeholder(dtype=tf.float32,
                                          shape=[None, sum(self.cell_2.state_size)],
                                          name="state_feed_2")

            with tf.name_scope("Decoding_step"):
                state_tuple_1 = tf.split(value=self.state_feed_1, num_or_size_splits=2, axis=1)
                state_tuple_2 = tf.split(value=self.state_feed_2, num_or_size_splits=2, axis=1)

                state_tuple_1 = tf.contrib.rnn.LSTMStateTuple(*state_tuple_1)
                state_tuple_2 = tf.contrib.rnn.LSTMStateTuple(*state_tuple_2)


                batch_size_tensor = tf.shape(self.word_encoded)[0]
                image_padding = tf.zeros([batch_size_tensor,1,self.image_embedding_size])
                with tf.variable_scope("RNN_1",reuse=True) as scope:
                    outputs_l1, last_state_l1 = tf.nn.dynamic_rnn(
                                                cell=self.cell_1,
                                                inputs=image_padding,
                                                initial_state=state_tuple_1,
                                                dtype=tf.float32)

                inputs_l2 = tf.concat([self.word_encoded,outputs_l1],axis=2)
                with tf.variable_scope("RNN_2",reuse=True) as scope:
                    outputs_l2, last_state_l2 = tf.nn.dynamic_rnn(
                                                cell=self.cell_2,
                                                inputs=inputs_l2,
                                                initial_state=state_tuple_2,
                                                dtype=tf.float32)
                
                self.infer_decoding_output_1 = outputs_l1
                self.infer_decoding_input_1 = image_padding
                self.infer_decoding_output_2 = outputs_l2
                self.infer_decoding_input_2 = inputs_l2
                
                self.infer_last_state_l1 = tf.concat(last_state_l1,axis=1,name="last_state_1_concat")
                self.infer_last_state_l2 = tf.concat(last_state_l2,axis=1,name="last_state_2_concat")

            self.infer_output_rnn = outputs_l2
            ############################################################################################
    def build_train_op(self):
        with tf.name_scope("Train"):
            ############################## TRAINING GRAPH #############################################
            batch_size_tensor = tf.shape(self.image_encoded)[0]

            cell_1 = self.cell_1
            cell_2 = self.cell_2

            if self.mode=="train":
                cell_1 = tf.contrib.rnn.DropoutWrapper( cell_1,
                                                        input_keep_prob=self.rnn1_input_keep_prob,
                                                        output_keep_prob=self.rnn1_output_keep_prob)
                cell_2 = tf.contrib.rnn.DropoutWrapper( cell_2,
                                                        input_keep_prob=self.rnn2_input_keep_prob,
                                                        output_keep_prob=self.rnn2_output_keep_prob)

            with tf.name_scope("Encoding_stage"):
                sequence_length = tf.reduce_sum(self.video_mask, 1)
                with tf.variable_scope("RNN_1") as scope:
                    outputs_encoding_l1,last_state_encoding_l1 = tf.nn.dynamic_rnn(
                                                                cell=cell_1,
                                                                sequence_length=sequence_length,
                                                                inputs=self.image_encoded,
                                                                dtype=tf.float32)
                padded_caption = tf.zeros([batch_size_tensor,self.num_frames,self.word_embedding_size]);
                input_encoding_layer2 = tf.concat([padded_caption,outputs_encoding_l1],axis=2);
                with tf.variable_scope("RNN_2") as scope:
                    output_encoding_l2,last_state_encoding_l2 = tf.nn.dynamic_rnn(
                                                                cell=cell_2,
                                                                sequence_length=sequence_length,
                                                                inputs=input_encoding_layer2,
                                                                dtype=tf.float32)
            with tf.name_scope("Decoding_state"):
                sequence_length = tf.reduce_sum(self.caption_mask, 1)
                padded_caption = tf.zeros([batch_size_tensor,self.num_caption_unroll,self.image_embedding_size]);
                with tf.variable_scope("RNN_1",reuse=True) as scope:
                    outputs_decoding_l1,last_state_decoding_l1 = tf.nn.dynamic_rnn(
                                                                cell = cell_1,
                                                                inputs=padded_caption,
                                                                sequence_length=sequence_length,
                                                                dtype=tf.float32,
                                                                initial_state=last_state_encoding_l1)
                caption_needed = self.caption_encoded[:,:-1,:]
                input_decoding_layer2 = tf.concat([caption_needed,outputs_decoding_l1],axis=2);
                with tf.variable_scope("RNN_2",reuse=True) as scope:
                    outputs_decoding_l2,last_state_decoding_l2 = tf.nn.dynamic_rnn(
                                                                cell = cell_2,
                                                                inputs=input_decoding_layer2,
                                                                sequence_length=sequence_length,
                                                                dtype=tf.float32,
                                                                initial_state=last_state_encoding_l2)
            self.train_output_rnn = outputs_decoding_l2
            ############################################################################################
    def build_output_logits(self):
        # converting to shape (batch_size*num_frames,dim_output_inception)
        train_input = tf.reshape(self.train_output_rnn,[-1,self.hidden_size_lstm2])
        infer_input = tf.reshape(self.infer_output_rnn,[-1,self.hidden_size_lstm2])
        
        with tf.variable_scope("word_decoding") as scope:
            train_logits_decoded = tf.contrib.layers.fully_connected(inputs=train_input,
                                                          num_outputs=self.vocab_size,
                                                          activation_fn = None,
                                                          weights_initializer=self.initializer,
                                                          biases_initializer=self.initializer,
                                                          scope = scope)
            scope.reuse_variables()
            infer_logits_decoded = tf.contrib.layers.fully_connected(inputs=infer_input,
                                                          num_outputs=self.vocab_size,
                                                          activation_fn = None,
                                                          weights_initializer=self.initializer,
                                                          biases_initializer=self.initializer,
                                                          scope = scope)

            # as the output will be a single word
            self.infer_logits = tf.reshape(infer_logits_decoded,[-1,1,self.vocab_size])
            self.infer_predictions = tf.nn.softmax(self.infer_logits)

            self.train_logits = tf.reshape(train_logits_decoded,[-1,self.num_caption_unroll,self.vocab_size])
            self.train_predictions = tf.nn.softmax(self.train_logits)

    def build_loss(self):
        with tf.variable_scope("loss") as scope:
            correct_predictions = self.caption_input[:,1:] #shape = (batch_size,num_caption_unroll) int32
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_predictions,logits=self.train_logits)
            #loss of shape (batch_size,num_caption_unroll) float32

            # Irrelevant now. But leaving for fun's sake ;)
            #    # I belive that this should be
            #    #    self.correct_loss = tf.multiply(self.loss, self.caption_mask[:, 1:]
            #    # because caption_mask denotes whether the ith word was actual caption or padding and as loss
            #    # is calculated using the prediction of next word thus we should multiply it by whether the next
            #    # word was padding or not and not on whether the current word was padding or not

            self.loss = tf.multiply(loss, self.caption_mask) #shape = (batch_size,num_caption_unroll)
            
            batch_size_tensor = tf.shape(self.loss)[0]

            self.individual_loss = tf.reduce_sum(self.loss,axis=1) #shape = (batch_size,)
            
            #### Which loss to use?
            ## Sum of loss of all the words in a sequence (Done in s2vt code that we saw)
            #self.batch_loss = tf.reduce_sum(self.loss)/tf.to_float(batch_size_tensor)
            #self.batch_loss = tf.reduce_mean(self.individual_loss)
            
            ## Avg of loss of all the words in a sequence (Dome in im2txt code that we saw)
            self.batch_loss = tf.div(tf.reduce_sum(self.loss),
                                    tf.reduce_sum(self.caption_mask),
                                    name="batch_loss")

            self.summaries.append(tf.summary.scalar('Batch_Loss', self.batch_loss))
            self.summaries.append(tf.summary.histogram('Loss_Histogram', self.batch_loss))
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
        