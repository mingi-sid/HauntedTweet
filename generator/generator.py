#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
import os.path

class Generator():
    def __init__(self, Embedding, cell_num):
        self.Embedding = Embedding
        self.embedding_size = Embedding.embedding_size()
        self.cell_num = cell_num
        
    def nn_init(self, timesteps, hidden_size, learning_rate = 0.001, seed=1):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        #variables
        self.timesteps = timesteps
        self.input_tensor = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size])
        self.label_tensor = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size])
        
        #inputs and outputs
        #Weights and biases for output
        W = tf.Variable(tf.random_normal([hidden_size, self.embedding_size]))
        b = tf.Variable(tf.random_normal([self.embedding_size]))
        
        
        #models
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=hidden_size), output_keep_prob=0.2)
        self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.input_tensor, dtype=tf.float32)
        
        self.output_tensor = self.output
        
        self.prediction = tf.matmul(self.output_tensor[-1], W) + b
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label_tensor))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    
    def train_real_data(self, num_steps, datafile, save_file_name, restore=True):
        with tf.Session() as session:
            if restore:
                print("Data restored")
                self.saver.restore(session, save_file_name)
            else:
                session.run(self.tf_init)
            
            for step in range(num_steps):
                #print(line)
                line = datafile.readline()
                input_tokens = line.split('\t')
                if len(input_tokens) < 3:
                    continue
                #print(input_tokens)
                input_vectors = [self.Embedding.word2vec(token) for token in input_tokens]
                #print(input_vectors)
                batch_size = len(input_tokens)
                input_batch = tf.expand_dims(tf.concat([tf.stack(input_vectors)[:self.timesteps-1, :], tf.zeros([max(self.timesteps-batch_size, 1), self.embedding_size], dtype=tf.float32)], 0), 0)
                
                eos_vec = self.Embedding.word2vec("('<eos>', 'token')")
                eos_tensor = tf.stack([eos_vec] * self.timesteps)
                
                print("batch_size=", batch_size, "input_batch=", input_batch)
                #print(input_batch)
                #print(eos_vec)
                #print(eos_tensor)
                input_label = tf.concat([input_batch[:, 1:, :], tf.reshape(eos_vec, [1, 1, -1])], axis=1)
                
                input_batch_, input_label_ = session.run([input_batch, input_label])
                
                _, loss = session.run([self.optimizer, self.loss],\
                                    {self.input_tensor : input_batch_, self.label_tensor : input_label_})
                if step % 50 == 0 or step == num_steps-1:
                    print("loss=", loss)
                self.saver.save(session, save_file_name)
            
    def train_against(self):
        pass
    def generate(self, save_file_name, num = 1):
        sentences = []
        with tf.Session() as session:
            self.saver.restore(session, save_file_name)
            for i in range(num):
                wordvecs = []
                random_input = session.run(tf.concat([tf.random_normal([1, 1, self.embedding_size], dtype=tf.float32), tf.zeros([1, self.timesteps-1, self.embedding_size], dtype=tf.float32)], axis=1))
                prediction, _ = session.run([self.prediction, self.state], feed_dict={self.input_tensor : random_input})
                for j in range(self.timesteps):
                    prediction = tf.expand_dims(prediction, 0)
                    #print(prediction)
                    prediction = session.run(prediction)
                    prediction, _ = session.run([self.prediction, self.state], feed_dict={self.input_tensor : prediction})
                    wordvecs.append(prediction)
                print([session.run(tf.reshape(y[0], [-1])) for y in wordvecs])
                sentence = [self.Embedding.closest_word(session.run(tf.reshape(y[0], [-1]))) for y in wordvecs]
            sentences.append(sentence)
        return sentences