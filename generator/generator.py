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
        
    def nn_init(self, batch_size, timesteps, hidden_size, learning_rate = 1E-05, seed=1, use_vector=False):
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
        self.input_vectors = tf.placeholder(tf.float32, [None, timesteps, self.embedding_size], 'input_vectors')
        self.input_length = tf.placeholder(tf.int32, [None], 'input_length')
        self.input_targets = tf.placeholder(tf.int32, [None, timesteps, self.output_size], 'input_targets')
        
        self.input_vectors_data = tf.data.Dataset.from_tensor_slices(self.input_vectors)
        self.input_length_data = tf.data.Dataset.from_tensor_slices(self.input_length)
        self.input_targets_data = tf.data.Dataset.from_tensor_slices(self.input_targets)
        
        self.input_vectors_iter = self.input_vectors_data.make_initializable_iterator()
        self.input_length_iter = self.input_length_data.make_initializable_iterator()
        self.input_targets_iter = self.input_targets_data.make_initializable_iterator()
        
        #Weights and biases for output
        W = tf.Variable(tf.random_normal([hidden_size[-1], self.output_size]), name='output_weights')
        b = tf.Variable(tf.random_normal([self.output_size]), name='output_biases')
        
        
        #models
        
        self.keep_input = tf.placeholder_with_default(1.0, (), 'keep_input')
        self.keep_output = tf.placeholder_with_default(1.0, (), 'keep_output')
        self.keep_state = tf.placeholder_with_default(1.0, (), 'keep_state')
        self.cells = [ tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=h_size), \
                    input_keep_prob=self.keep_input, output_keep_prob=self.keep_output, state_keep_prob=self.keep_state) for h_size in hidden_size]
        
        self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cells)
        self.initial_state = tuple([ tf.placeholder_with_default(self.cells[i].zero_state(batch_size, dtype=tf.float32), shape=(batch_size, hidden_size[i]), name='initial_state_'+str(i)) for i in range(len(self.cells)) ])
        
        self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.input_vectors, self.input_length, initial_state=self.initial_state, dtype=tf.float32)
        
        self.output_tensor = self.output
        
        if not self.use_vector:
            self.prediction = tf.reshape(tf.matmul(tf.reshape(self.output_tensor, [-1, hidden_size[-1]]), W) + b, [-1, timesteps, self.output_size])
            self.probs = tf.nn.softmax(self.prediction)
            
            zipf_inv_list = [ np.power(float(x) * np.log(self.word_size), 0.7) for x in range(1, self.word_size+1) ]
            self.zipf_inv = tf.constant(zipf_inv_list, dtype=tf.float32)
            
            self.exag_probs = tf.multiply(self.probs, self.zipf_inv)
        
            self.weights = tf.constant([ [np.power(1/float(j), 0.5) for j in range(1, timesteps+1)] for i in range(batch_size)])
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.input_targets)
            mask = tf.sequence_mask(self.input_length, maxlen=timesteps, dtype=tf.float32)
            
            self.loss = tf.reduce_sum(mask*cross_entropy) / tf.reduce_sum(tf.cast(self.input_length, dtype=tf.float32))
            tf.summary.scalar('loss', self.loss)
        else:
            self.prediction = tf.reshape(tf.matmul(tf.reshape(self.output_tensor, [-1, hidden_size[-1]]), W) + b, [-1, timesteps, self.output_size])
            self.loss = tf.losses.cosine_distance(labels=self.input_targets, predictions=self.prediction, axis=2)
            tf.scalar_summary('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        
    
    def batch_real_data(self, datafile):
        input_vectors_l = []
        input_length_l = []
        input_targets_l = []
        
        datafile.seek(self.filepos.eval())
        count = 0
        while count < self.batch_size:
            try:
                line = datafile.readline()
            except UnicodeDecodeError:
                datafile.seek(0, 0)
                line = datafile.readline()
            if line == "":
                print("One epoch ended.")
                datafile.seek(0, 0)
                line = datafile.readline()
            input_tokens = line.strip().split('\t')
            
            input_length = len(input_tokens) - 1
            eos_vec = self.Embedding.word2vec("('<eos>', 'Token')")
            go_vec = self.Embedding.word2vec("('<go>', 'Token')")
            shifted_tokens = ( input_tokens[1:] + ["('<go>', 'Token')"]*max(1, self.timesteps-len(input_tokens)+1) )[:self.timesteps]
            input_labels = [self.Embedding.word2code(token) for token in shifted_tokens]
            input_targets = [ [1 if i==code else 0 for i in range(self.word_size)] for code in input_labels ]
            #Randomly ignore sentence with UNK
            if 0 in input_labels:
                if np.random.random() < 0.0:
                    continue
            
            #Subsampling
            appear_prob = 1.0
            for token in input_labels:
                appear_prob *= self.Embedding.codefreq(token)
            if np.random.random() < np.power(appear_prob, 1/float(self.timesteps)) / 7E-4 - 1:
                pass#continue
                
            input_vectors = ( [self.Embedding.word2vec(token) for token in input_tokens] + [go_vec]*max(1, self.timesteps-input_length) )[:self.timesteps]
            input_targets_vectors = input_vectors[1:] + [go_vec]
            
            input_length_l.append(input_length)
            input_vectors_l.append(input_vectors)
            
            if not self.use_vector:
                input_targets_l.append(input_targets)
            else:
                input_targets_l.append(input_targets_vectors)
            count += 1
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
            summaries = tf.summary.merge_all()
            for step in range(num_steps):
                input_vectors_l, input_length_l, input_targets_l = self.batch_real_data(datafile)
                
                _, loss, summary, __, ___, ____ = session.run([self.optimizer, self.loss, summaries, self.input_vectors_iter.initializer, self.input_length_iter.initializer, self.input_targets_iter.initializer],\
                                    {self.input_vectors : input_vectors_l, self.input_length : input_length_l, self.input_targets : input_targets_l,\
                                    self.keep_input : 1.0, self.keep_output : 0.5, self.keep_state : 1.0})
                total_loss += loss
                loss_count += 1
                if step % 20 == 0 or step == num_steps-1:
                    print("average loss at {} = {}".format(step, total_loss / loss_count))
                    total_loss = 0.0
                    loss_count = 0
                if step % 200 == 0:
                    sentences = self.generate(None, 10)
                    for sentence in sentences:
                        print( ' '.join([token.split("'")[1] for token in sentence]) )
                if step % 10 == 0:
                    self.saver.save(session, save_file_name)
                    train_writer = tf.summary.FileWriter('tmp/gen_log', session.graph)
                    train_writer.add_summary(summary, step)

    def train_against(self):
        pass
        
    def generate(self, save_file_name, size = 1):
        sentences = []
        external_run = save_file_name != None
        if not external_run:
            session = tf.get_default_session()
        else:
            session = tf.Session()
            self.saver.restore(session, save_file_name)
        go_vec = self.Embedding.word2vec("('<go>', 'Token')")
        eos_vec = self.Embedding.word2vec("('<eos>', 'Token')")
        initial_input = [ ( [go_vec] + [eos_vec]*max(1, self.timesteps-1) )[:self.timesteps] ] * self.batch_size
        initial_state = [ tf.random_normal([self.batch_size, h_size], stddev=0.001, name='initial_state_'+str(h_size)).eval(session=session) for h_size in self.hidden_size ]
        
        for k in range((size-1) // self.batch_size + 1):
            batch_sized_sentences = [ [] for x in range(self.batch_size)]
            input_words = [ ["('<go>', 'Token')"] for x in range(self.batch_size) ]
            if not self.use_vector:
                probs, state = session.run([self.probs, self.state], feed_dict={self.input_vectors : initial_input, self.input_length : [1] * self.batch_size, self.initial_state : initial_state})
            else:
                prediction, state = session.run([self.prediction, self.state], feed_dict={self.input_vectors : initial_input, self.input_length : [1] * self.batch_size, self.initial_state : initial_state})
            prev_input = initial_input
            
            if not external_run: print("Highest probs of 1st sentence", end=" ")
            for i in range(self.timesteps):
                if not self.use_vector:
                    new_word_prob = probs[:, 0, :]
                    if not external_run: print("{0:.1f}%".format(max(new_word_prob[0, :]) * 100), end=" ")
                words = []
                for j in range(self.batch_size):
                    if not self.use_vector:
                        #Only if UNK token is dominant, use UNK.
                        if(new_word_prob[j, 0] > 0.99):
                            words.append(self.Embedding.code2word(0))
                            batch_sized_sentences[j].append(self.Embedding.code2word(0))
                            continue
                            
                        randfloat = np.random.random()
                        word_cumul_prob = 0
                        #Give probability weight
                        weighted_prob = np.power(new_word_prob[j, :], 2.0)
                        weighted_prob[self.Embedding.word2code("('<eos>', 'Token')")] /= 16.0
                        prob_sum = sum(weighted_prob[1:])
                        for k in range(1, len(weighted_prob[1:])+1):
                            word_cumul_prob += weighted_prob[k]
                            if randfloat < word_cumul_prob / prob_sum: break
                    
                        words.append(self.Embedding.code2word(k))
                        batch_sized_sentences[j].append(self.Embedding.code2word(k))
                    else:
                        close_words = self.Embedding.closest_word([prediction[j, 0, :]], 5)
                        if j == 0 and not external_run: print(close_words[0][0], end=" ")
                        chosen_word = np.random.choice(close_words[0])
                        words.append(chosen_word)
                        batch_sized_sentences[j].append(chosen_word)
                
                if i == self.timesteps: break
                
                input = [ ( [self.Embedding.word2vec(words[j])] + [eos_vec]*max(1, self.timesteps-1) )[:self.timesteps] for j in range(self.batch_size) ]
                for j in range(self.batch_size): input_words[j].append(words[j])
                if not self.use_vector:
                    probs, state = session.run([self.probs, self.state], feed_dict={self.input_vectors : input, self.input_length : [1] * self.batch_size, self.initial_state : state})
                else:
                    prediction, state = session.run([self.prediction, self.state], feed_dict={self.input_vectors : input, self.input_length : [1] * self.batch_size, self.initial_state : state})
                
            if not external_run: print()
            sentences += batch_sized_sentences
        return sentences[:size]