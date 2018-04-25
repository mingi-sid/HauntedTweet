import unittest
import os
import sys
from data.word2vec import Word2Vec
from generator.generator import Generator

class GeneratorTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_short_run(self):
        token_data = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_dataset.txt")
        freq_data = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_result.txt")
        with open(token_data) as data:
            with open(freq_data) as freq:
                W = Word2Vec(data, freq)
                print("Made Word2Vec instance")
                
                w2v_save_file = os.path.join(os.path.dirname(sys.argv[0]), "test_word2vec.tfsav")
                gen_save_file = os.path.join(os.path.dirname(sys.argv[0]), "test_generator.tfsav")
                
                W.give_code()
                print("Gave code for tokens")
                W.tf_init(32, 30, seed=3)
                print("Initialized")
                
                print("1st run (10 steps)....")
                W.tf_run(10, w2v_save_file, restore=True)
                
                E = W.Embeddings()
                print("Made Embeddings instance")
                del W
                
                G = Generator(E, 34)
                print("Made Generator instance")
                G.nn_init(10, 16)
                print("Initialized")
                
                print("1st run (100 steps)....")
                G.train_real_data(100, data, gen_save_file, restore=False)
                
                print("2st run (100 steps)....")
                G.train_real_data(100, data, gen_save_file, restore=True)
                
                print(G.generate(gen_save_file))
                
        
    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()