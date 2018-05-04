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
        with open(token_data, encoding='utf8') as data:
            with open(freq_data, encoding='utf8') as freq:
                W = Word2Vec(data, freq)
                print("Made Word2Vec instance")
                
                w2v_save_file = os.path.join(os.path.dirname(sys.argv[0]), "test_word2vec.tfsav")
                gen_save_file = os.path.join(os.path.dirname(sys.argv[0]), "test_generator.tfsav")
                
                W.give_code()
                print("Gave code for tokens")
                W.tf_init(32, 192)
                print("Initialized")
                
                print("1st run (100 steps)....")
                W.tf_run(100, w2v_save_file, restore=True)
                
                E = W.Embeddings()
                print("Made Embeddings instance")
                del W
                
                G = Generator(E)
                print("Made Generator instance")
                G.nn_init(batch_size=32, timesteps=32, hidden_size=64)
                print("Initialized")
                
                print("1st run (5 steps)....")
                G.train_real_data(5, data, gen_save_file, restore=False)
                
                print("2st run (5 steps)....")
                G.train_real_data(5, data, gen_save_file)
                
                print(G.generate(gen_save_file))
                
        
    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()