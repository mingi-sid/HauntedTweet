#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
import os.path

class Generator():
    def __init__(self, Embedding):
        self.Embedding = Embedding
        self.embedding_size = Embedding.embedding_size()
        
    def nn_init(self, batch_size, timesteps, hidden_size, learning_rate = 0.001, seed=1):
        if seed != None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
        
        #variables
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.input_tensor = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size])
        self.label_tensor = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size])
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.filepos = tf.get_variable("gen_filepos", initializer=tf.constant(0), dtype=tf.int32)
        
        #inputs and outputs
        #Weights and biases for output
        W = tf.Variable(tf.random_normal([hidden_size, self.embedding_size]))
        b = tf.Variable(tf.random_normal([self.embedding_size]))
        
        
        #models
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=hidden_size), output_keep_prob=0.2)
        self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.input_tensor, self.seq_length, dtype=tf.float32)
        
        self.output_tensor = self.output
        
        self.prediction = tf.matmul(tf.reshape(self.output_tensor, [-1, hidden_size]), W) + b
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label_tensor))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    
    def train_real_data(self, num_steps, datafile, save_file_name, restore=True):
        with tf.Session() as session:
            if restore and os.path.isfile(save_file_name+".meta"):
                print("Data restored")
                self.saver.restore(session, save_file_name)
            else:
                session.run(self.tf_init)
            
            total_loss = 0.0
            for step in range(num_steps):
                input_batches = tf.constant([], shape=[0, self.timesteps, self.embedding_size], dtype=tf.float32)
                input_labels = tf.constant([], shape=[0, self.timesteps, self.embedding_size], dtype=tf.float32)
                input_seq_length = tf.constant([], dtype=tf.int32)
                
                for i in range(self.batch_size):
                    #print(line)
                    datafile.seek(self.filepos.eval())
                    line = datafile.readline()
                    if line == "":
                        datafile.seek(0, 0)
                        line = datafile.readline()
                    input_tokens = line.split('\t')
                    #print(input_tokens)
                    input_vectors = [self.Embedding.word2vec(token) for token in input_tokens]
                    #print(input_vectors)
                    input_length = len(input_tokens)
                    input_batch = tf.concat([tf.stack(input_vectors)[:self.timesteps-1, :], tf.zeros([max(self.timesteps-input_length, 1), self.embedding_size], dtype=tf.float32)], 0)
                    
                    eos_vec = self.Embedding.word2vec("('<eos>', 'token')")
                    #eos_tensor = tf.stack([eos_vec] * self.timesteps)
                    
                    #print("input_length=", input_length, "input_batch=", input_batch)
                    #print(input_batch)
                    #print(eos_vec)
                    #print(eos_tensor)
                    input_label = tf.concat([input_batch[1:, :], tf.reshape(eos_vec, [1, -1])], axis=0)
                    
                    input_seq_length = tf.concat([input_seq_length, [input_length]], 0)
                    input_batches = tf.concat([input_batches, tf.expand_dims(input_batch, 0)], 0)
                    input_labels = tf.concat([input_labels, tf.expand_dims(input_label, 0)], 0)
                    
                input_batches_, input_labels_, input_seq_length_ = session.run([input_batches, input_labels, input_seq_length])
                
                _, loss = session.run([self.optimizer, self.loss],\
                                    {self.input_tensor : input_batches_, self.label_tensor : input_labels_, self.seq_length : input_seq_length_})
                self.filepos = tf.assign(self.filepos, datafile.tell())
                total_loss += loss
                if step % 50 == 0 or step == num_steps-1:
                    print("average loss at {} = {}".format(step, total_loss / 50))
                    total_loss = 0.0
            if step % 10 == 0 or step == num_steps-1:
                self.saver.save(session, save_file_name)
            
    def train_against(self):
        pass
        
    def generate(self, save_file_name, num = 1):
        sentences = []
        with tf.Session() as session:
            self.saver.restore(session, save_file_name)
            for i in range(num):
                wordvecs = []
                #random_input = session.run(tf.concat([tf.random_normal([1, 1, self.embedding_size], dtype=tf.float32), tf.zeros([1, self.timesteps-1, self.embedding_size], dtype=tf.float32)], axis=1))
                go_vec = self.Embedding.word2vec("('<go>', 'token')")
                go_input = session.run(tf.concat([tf.expand_dims(tf.expand_dims(go_vec, 0), 0), tf.zeros([1, self.timesteps-1, self.embedding_size], dtype=tf.float32)], axis=1))
                prediction, state = session.run([self.prediction, self.state], feed_dict={self.input_tensor : go_input, self.seq_length : [1]})
                prev_input = go_input
                for j in range(1, self.timesteps):
                    print("result=", prediction[:, 0])
                    prediction = tf.expand_dims(tf.concat([prev_input[0, :j, :], tf.expand_dims(prediction[j-1, :], 0)], axis=0), 0)
                    prediction = tf.concat([prediction, tf.zeros([1, self.timesteps-j-1, self.embedding_size], dtype=tf.float32)], axis=1)
                    prediction = session.run(prediction)
                    print("input =", prediction[0, :, 0])
                    prev_input = prediction
                    prediction, _ = session.run([self.prediction, self.state], feed_dict={self.input_tensor : prediction, self.seq_length : [j+1]})
                    wordvecs.append(prediction[j, :])
                #print([session.run(tf.reshape(y, [-1])) for y in wordvecs])
                sentence = [self.Embedding.closest_word(session.run(tf.reshape(y[0], [-1]))) for y in wordvecs]
            sentences.append(sentence)
        return sentences