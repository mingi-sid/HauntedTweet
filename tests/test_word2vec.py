#Running tests : python3 -m tests.test_***
import unittest
import os
import sys
from data.word2vec import Word2Vec

class Word2VecTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_short_run(self):
        token_data = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_dataset.txt")
        freq_data = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_result.txt")
        with open(token_data, encoding='utf8') as data:
            with open(freq_data, encoding='utf8') as freq:
                self.W = Word2Vec(data, freq)
                print("Made Word2Vec instance\n")
                
                save_file = os.path.join(os.path.dirname(sys.argv[0]), "test_word2vec.tfsav")
                
                self.W.give_code()
                print("Gave code for tokens\n")
                self.W.tf_init(32, 192)
                print("Initialized\n")
                
                print("1st run (100 steps)....")
                self.W.tf_run(100, save_file, restore=False)
                print("done.\n")
                print("2nd run (100 steps)....")
                self.W.tf_run(100, save_file)
                print("done.\n")
                
                self.E = self.W.Embeddings()
                print("Made Embeddings instance")
                del self.W
                
                s = ""
                for i in range(5):
                    w = self.E.code2word(i)
                    s += w + " : " + str(self.E.word2vec(w)) + "\n"
                print(s)
                
                self.E.closest_word([self.E.word2vec(self.E.code2word(self.E.word2code("('이', 'Josa')")))])
             
                

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
