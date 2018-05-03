#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
import os.path

class Generator():
    def __init__(self, Embedding):
        self.Embedding = Embedding
        self.embedding_size = Embedding.embedding_size()
        self.word_size = Embedding.word_size()
        
    def nn_init(self, batch_size, timesteps, hidden_size, learning_rate = 0.001, seed=1, use_vector=False):
        if seed != None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
        if type(hidden_size) == int:
            hidden_size = [hidden_size]
        self.use_vector = use_vector
            
        #variables
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        if not self.use_vector:
            self.output_size = self.word_size
        else:
            self.output_size = self.embedding_size
        self.filepos = tf.get_variable("gen_filepos", initializer=tf.constant(0), dtype=tf.int32)
        
        #inputs and outputs
        self.input_vectors = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size])
        self.input_length = tf.placeholder(tf.int32, [None])
        self.input_targets = tf.placeholder(tf.int32, [None, timesteps, self.output_size])
        
        
        #Weights and biases for output
        W = tf.Variable(tf.random_normal([hidden_size[-1], self.output_size]))
        b = tf.Variable(tf.random_normal([self.output_size]))
        
        
        #models
        
        self.keep_input = tf.placeholder_with_default(1.0, shape=())
        self.keep_output = tf.placeholder_with_default(1.0, shape=())
        self.keep_state = tf.placeholder_with_default(1.0, shape=())
        self.cells = [ tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=h_size), \
                    input_keep_prob=self.keep_input, output_keep_prob=self.keep_output, state_keep_prob=self.keep_state) for h_size in hidden_size]
        
        self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cells)
        self.initial_state = tuple([ tf.placeholder_with_default(self.cells[i].zero_state(batch_size, dtype=tf.float32), shape=(batch_size, hidden_size[i])) for i in range(len(self.cells)) ])
        
        self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.input_vectors, self.input_length, initial_state=self.initial_state, dtype=tf.float32)
        
        self.output_tensor = self.output
        #print("self.output", self.output)
        
        if not self.use_vector:
            self.prediction = tf.reshape(tf.matmul(tf.reshape(self.output_tensor, [-1, hidden_size[-1]]), W) + b, [-1, timesteps, self.output_size])
            self.probs = tf.nn.softmax(self.prediction)
            #print("self.probs", self.probs)
            
            zipf_inv_list = [ np.power(float(x) * np.log(self.word_size), 0.7) for x in range(1, self.word_size+1) ]
            self.zipf_inv = tf.constant(zipf_inv_list, dtype=tf.float32)
            
            self.exag_probs = tf.multiply(self.probs, self.zipf_inv)
            #self.exag_labels = tf.multiply(self.input_labels, self.zipf_inv)
        
            #print("self.exag_probs", self.exag_probs)
            #print("self.exag_labels", self.exag_labels)
            self.weights = tf.constant([ [np.power(1/float(j), 0.5) for j in range(1, timesteps+1)] for i in range(batch_size)])
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.probs, labels=self.input_targets))
            #self.loss = tf.reduce_mean(tf.norm(self.exag_probs-self.exag_labels))
        else:
            self.prediction = tf.reshape(tf.matmul(tf.reshape(self.output_tensor, [-1, hidden_size[-1]]), W) + b, [-1, timesteps, self.output_size])
            self.loss = tf.losses.cosine_distance(labels=self.input_targets, predictions=self.prediction, axis=2)
        self.distance = tf.losses.mean_squared_error(self.prediction[:, :-1, :], self.prediction[:, 1:, :])
        self.additional_loss = tf.exp(-(self.distance/1E+03 - 1))
        #self.loss = self.loss + self.additional_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    
    def batch_real_data(self, datafile):
        input_vectors_l = []
        input_length_l = []
        #input_codes_l = []
        input_targets_l = []
        
        datafile.seek(self.filepos.eval())
        count = 0
        while count < self.batch_size:
            line = datafile.readline()
            #print(line.strip())
            if line == "":
                print("One epoch ended.")
                datafile.seek(0, 0)
                line = datafile.readline()
            input_tokens = line.strip().split('\t')
            
            input_length = len(input_tokens)-1 - 1
            eos_vec = self.Embedding.word2vec("('<eos>', 'Token')")
            shifted_tokens = ( input_tokens[1:] + ["('<eos>', 'Token')"]*max(1, self.timesteps-input_length+1) )[:self.timesteps]
            input_labels = [self.Embedding.word2code(token) for token in shifted_tokens]
            input_targets = [ [1 if i==code else 0 for i in range(self.word_size)] for code in input_labels ]
            if 0 in input_targets:
                continue
            
            #Subsampling
            appear_prob = 1
            for token in input_labels:
                appear_prob *= self.Embedding.codefreq(token)
            if np.random.random() < np.power(appear_prob, 1/float(self.timesteps)) / 7E-4 - 1:
                pass#continue
                
            input_vectors = ( [self.Embedding.word2vec(token) for token in input_tokens] + [eos_vec]*max(1, self.timesteps-input_length) )[:self.timesteps]
            input_targets_vectors = input_vectors[1:] + [eos_vec]
            
            input_length_l.append(input_length)
            input_vectors_l.append(input_vectors)
            if not self.use_vector:
                input_targets_l.append(input_targets)
            else:
                input_targets_l.append(input_targets_vectors)
            count += 1
            #input_codes = [ [(1 if num==self.Embedding.word2code(token) else 0) for num in range(self.word_size)] for token in shifted_tokens ]
            #print([code.index(1) for code in input_codes])
            #input_codes_l.append(input_codes)
        self.filepos.load(datafile.tell())
            
        return input_vectors_l, input_length_l, input_targets_l
            
    def train_real_data(self, num_steps, datafile, save_file_name, restore=True):
        with tf.Session() as session:
            if restore and os.path.isfile(save_file_name+".meta"):
                self.saver.restore(session, save_file_name)
                print("Data restored. Resume learning from {}".format(self.filepos.eval()))
            else:
                session.run(self.tf_init)
            
            total_loss = 0.0
            loss_count = 0
            for step in range(num_steps):
                input_vectors_l, input_length_l, input_targets_l = self.batch_real_data(datafile)
                
                _, loss, additional_loss = session.run([self.optimizer, self.loss, self.additional_loss],\
                                    {self.input_vectors : input_vectors_l, self.input_length : input_length_l, self.input_targets : input_targets_l,\
                                    self.keep_input : 0.5, self.keep_output : 0.5, self.keep_state : 0.5})
                #print(additional_loss)
                total_loss += loss
                loss_count += 1
                if step % 10 == 0 or step == num_steps-1:
                    print("average loss at {} = {}".format(step, total_loss / loss_count))
                    total_loss = 0.0
                    loss_count = 0
                    self.saver.save(session, save_file_name)
                if step % 50 == 0:
                    sentences = self.generate(None, 8)
                    for sentence in sentences:
                        print( ' '.join([token.split("'")[1] for token in sentence]) )
            
    def train_against(self):
        pass
        
    def generate(self, save_file_name, size = 1):
        sentences = [ [] for x in range(self.batch_size)]
        if save_file_name == None:
            session = tf.get_default_session()
        else:
            session = tf.Session()
            self.saver.restore(session, save_file_name)
            
        go_vec = self.Embedding.word2vec("('<go>', 'Token')")
        eos_vec = self.Embedding.word2vec("('<eos>', 'Token')")
        initial_input = [ ( [go_vec] + [eos_vec]*max(1, self.timesteps-1) )[:self.timesteps] ] * self.batch_size
        input_words = [ ["('<go>', 'Token')"] for x in range(self.batch_size) ]
        initial_state = [ tf.random_normal([self.batch_size, h_size], stddev=0.1).eval() for h_size in self.hidden_size ]
        
        if not self.use_vector:
            probs, state = session.run([self.probs, self.state], feed_dict={self.input_vectors : initial_input, self.input_length : [1] * self.batch_size, self.initial_state : initial_state})
        else:
            prediction, state = session.run([self.prediction, self.state], feed_dict={self.input_vectors : initial_input, self.input_length : [1] * self.batch_size, self.initial_state : initial_state})
        prev_input = initial_input
        
        print("Highest probs of 1st sentence", end=" ")
        for i in range(self.timesteps):
            if not self.use_vector:
                #print("result=", probs)
                new_word_prob = probs[:, 0, :]
                #print(sum([(1, -1)[i%2] * new_word_prob[:, i] for i in range(len(new_word_prob[0, :]))]))
                #print(new_word_prob[0, 0])
                print(max(new_word_prob[0, :]), end=" ")
            words = []
            for j in range(self.batch_size):
                if not self.use_vector:
                    randfloat = np.random.random()
                    word_cumul_prob = 0
                    prob_sum = sum(new_word_prob[j, :])
                    #print(randfloat * prob_sum)
                    for k in range(len(new_word_prob[j, :])):
                        word_cumul_prob += new_word_prob[j, k]
                        if randfloat * prob_sum < word_cumul_prob: break
                
                    words.append(self.Embedding.code2word(k))
                    sentences[j].append(self.Embedding.code2word(k))
                else:
                    close_words = self.Embedding.closest_word([prediction[j, 0, :]], 5)
                    if j == 0: print(close_words[0][0], end=" ")
                    chosen_word = np.random.choice(close_words[0])
                    words.append(chosen_word)
                    sentences[j].append(chosen_word)
            
            if i == self.timesteps: break
            
            #print(len(sentences[0]))
            input = [ ( [self.Embedding.word2vec(words[j])] + [eos_vec]*max(1, self.timesteps-1) )[:self.timesteps] for j in range(self.batch_size) ]
            for j in range(self.batch_size): input_words[j].append(words[j])
            #for input_ in input:
            #    print(input_)
            if not self.use_vector:
                probs, state = session.run([self.probs, self.state], feed_dict={self.input_vectors : input, self.input_length : [1] * self.batch_size, self.initial_state : state})
            else:
                prediction, state = session.run([self.prediction, self.state], feed_dict={self.input_vectors : input, self.input_length : [1] * self.batch_size, self.initial_state : state})
            
            
            # prediction = tf.expand_dims(tf.concat([prev_input[0, :i, :], tf.expand_dims(prediction[i-1, :], 0)], axis=0), 0)
            # prediction = tf.concat([prediction, tf.zeros([1, self.timesteps-i-1, self.embedding_size], dtype=tf.float32)], axis=1)
            # prediction = session.run(prediction)
            # print("input =", prediction[0, :, 0])
            # prev_input = prediction
            # prediction, _ = session.run([self.prediction, self.state], feed_dict={self.input_tensor : prediction, self.seq_length : [i+1]})
            # wordvecs.append(prediction[i, :])
            #print([session.run(tf.reshape(y, [-1])) for y in wordvecs])
            # sentence = [session.run(tf.reshape(y[0], [-1])) for y in wordvecs]
        print()
        #print(input_words)
        return sentences[:size]