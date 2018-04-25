#!/usr/bin/env python
"""Referred to TensorFlow word2vec tutorial
(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py), 
which follows Apache License, Ver. 2.0."""
import tensorflow as tf
import numpy as np
import math
import os.path
from six.moves import xrange

class Word2Vec():
    def __init__(self, datafile, freqfile):
        self._file = datafile
        self._freqfile = freqfile
        self._word2code = dict()
        self._code2word = dict()
        self._dataline = [""]
        self._dataindex = -1

    def give_code(self):
    #give each word a number, from 1 to 50000 and save mapping at dictionary
        count = 1
        #If there are more than 50000 words, replace rare words into UNK token
        max_word_count = 50000
        self._code2word[0] = "('UNK', 'UNK')"
        
        for line in self._freqfile:
            word, freq = tuple(line.split('\t'))
            freq = float(freq)
            if count <= max_word_count:
                self._word2code[word] = count
                self._code2word[count] = word
            else:
                self._word2code[word] = 0
            count += 1
        self._vocabulary_size = min(max_word_count + 1, count)
        #print(self._vocabulary_size)
        #print(min(list(self._word2code.values())), max(list(self._word2code.values())), len(list(self._word2code.values())))
        #print(list(self._code2word.items())[0:10])
    
    def generate_batch(self, batch_size, window_size = 3):
    #From "token\ttoken" format file, create batch and labels array
        index = 0
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch.fill(-1)
        labels.fill(-1)
        while index < batch_size:
            if self._dataindex == -1 or self._dataindex >= len(self._dataline):
                self._dataline = self._file.readline()
                if self._dataline == "":
                    self._file.seek(0, 0)
                    self._dataline = self._file.readline()
                elif len(self._dataline.strip().split('\t')) <= 1:
                    while len(self._dataline.strip().split('\t')) <= 1:
                        self._dataline = self._file.readline()
                self._dataline = self._dataline.strip().split('\t')
                self._dataindex = 0
            for j in range(window_size):
                if index >= batch_size:
                    break
                if self._dataindex - j - 1 < 0:
                    break
                labels[index, 0] = self._word2code[ self._dataline[self._dataindex - j - 1] ]
                batch[index] = self._word2code[ self._dataline[self._dataindex] ]
                #print(self._word2code[ self._dataline[self._dataindex - j - 1] ], self._word2code[ self._dataline[self._dataindex] ], index)
                index += 1
            for j in range(window_size):
                if index >= batch_size:
                    break
                if self._dataindex + j + 1 >= len(self._dataline) :
                    break
                labels[index, 0] = self._word2code[ self._dataline[self._dataindex + j + 1] ]
                batch[index] = self._word2code[ self._dataline[self._dataindex] ]
                #print(self._word2code[ self._dataline[self._dataindex + j + 1] ], self._word2code[ self._dataline[self._dataindex] ], index)
                index += 1
            self._dataindex += 1
        #print(batch)
        #print(labels)
        return batch, labels

    def tf_init(self, embedding_size, batch_size, seed=1):
    #Initialize tensorflow variables.
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self._batch_size = batch_size
        self._embedding_size = embedding_size
        
        self._valid_size = 16
        valid_window = 100
        self._valid_examples = np.random.choice(valid_window, self._valid_size, replace=False)
        num_sampled = 64

        self._train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(self._valid_examples, dtype=tf.int32)

        truncated = tf.truncated_normal([self._vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size))
        nce_weights = tf.Variable(truncated)
        nce_biases = tf.Variable(tf.zeros([self._vocabulary_size]))

        embeddings = tf.Variable(tf.random_uniform([self._vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self._train_inputs)

        nce_loss = tf.nn.nce_loss(weights=nce_weights,
                    biases=nce_biases,
                    labels=self._train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=self._vocabulary_size)
        self._loss = tf.reduce_mean(nce_loss)

        self._optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self._loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        self._normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self._normalized_embeddings, valid_dataset)
        self._similarity = tf.matmul(valid_embeddings, self._normalized_embeddings, transpose_b=True)
        
        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

    def tf_run(self, num_steps, save_file_name, restore=True):
    #Run tensorflow session for given steps, and save the result.
        with tf.Session() as session:
            if restore == True:# and os.path.isfile(save_file_name):
                self._saver.restore(session, save_file_name)
                print("Save file loaded.")
            else:
                session.run(self._init)

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = self.generate_batch(self._batch_size)

                feed_dict = {self._train_inputs : batch_inputs, self._train_labels : batch_labels}
                _, loss_val = session.run([self._optimizer, self._loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step {} : {}'.format(step, average_loss))
                    average_loss = 0

                if step % 10000 == 0:
                    sim = self._similarity.eval()
                    for i in xrange(self._valid_size):
                        valid_word = self._code2word[self._valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = self._code2word[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

                self._saver.save(session, save_file_name)
            self._final_embeddings = self._normalized_embeddings.eval()

    def Embeddings(self):
        return Embeddings(self._embedding_size, self._word2code, self._code2word, self._final_embeddings)

class Embeddings():
    def __init__(self, embedding_size, word2code, code2word, embeddings):
        self._embedding_size = embedding_size
        self._word2code = word2code
        self._code2word = code2word
        self._embeddings = embeddings

    def word2code(self, word):
        if word in self._word2code:
            return self._word2code[word]
        else:
            return 0

    def code2word(self, code):
        if code in self._code2word:
            return self._code2word[code]
        else:
            return "('UNK', 'UNK')"

    def word2vec(self, word):
        return self._embeddings[ self.word2code(word) ]

    def closest_word(self, vector):
        with tf.Session() as sess:
            norm_vector = tf.nn.l2_normalize(vector).eval()
            batch_array = tf.constant(norm_vector, shape=[1, self._embedding_size])
            similarity = tf.matmul(batch_array, self._embeddings, transpose_b=True)
            nearest_preeval = tf.argmax(similarity, 1)
            nearest = nearest_preeval.eval()
            return self.code2word(nearest[0])
    
    def embedding_size(self):
        return self._embedding_size