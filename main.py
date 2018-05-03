import os
import sys
from data.preparser import Preparser
from data.parser import Parser
from data.word2vec import Word2Vec
from generator.generator import Generator
import gc

def join_filenames(*names):
    return os.path.join(os.path.dirname(sys.argv[0]), *names)
    
help_message = '''-h : Prints this message
-i : Initial run. Parse tweets.csv
-w N : Run Word2Vec learning N steps
-W [filename] : Load and save Word2Vec data from [filename]
-g N : Run Generator NN learning N steps
-G [filename] : Load and save Generator data from [filename]
-V : Run Generator word2vec mode. Otherwise one-hot-vector mode is used.'''

def main():
    argdict = dict(zip(sys.argv, sys.argv[1:] + ['']))
    if "-h" in argdict:
        print(help_message)
        return
    raw_filename = join_filenames("data", "tweets.csv")
    filtered_filename = join_filenames("data", "_tweets_filtered.txt")
    stat_filename = join_filenames("data", "tweets_stat.txt")
    tokenized_filename = join_filenames("data", "tweets_tokenized.txt")
    
    embedding_size = 64
    word2vec_batch_size = 640
    gen_batch_size = 32
    gen_seq_length = 32
    gen_hidden_size = [64, 128]
    
    if "-i" in argdict:
        proceed = True
        if os.path.isfile(tokenized_filename):
            proceed = (input("Erasing old data. OK to proceed? (Y/N)") == "Y")
        if proceed:
            with open(raw_filename, "r", encoding='utf8') as raw_file_r:
                #Filter actual tweets
                preparser = Preparser(raw_file_r)
                preparser.extract(filter=True)
                with open(filtered_filename, "w", encoding='utf8') as filtered_file_w:
                    preparser.save(filtered_file_w)
                
                #Tokenize tweets
                with open(filtered_filename, "r", encoding='utf8') as filtered_file_r:
                    parser = Parser(filtered_file_r)
                    with open(stat_filename, "w", encoding='utf8') as stat_file_w:
                        parser.get_stats(stat_file_w)
                    with open(tokenized_filename, "w", encoding='utf8') as tokenized_file_w:
                        parser.get_data(tokenized_file_w)
                del parser
                
    if "-w" in argdict and int(argdict["-w"]) >= 0:
        word2vec_num_step = int(argdict["-w"])
        word2vec_save_filename = join_filenames("saves", argdict["-W"])
        word2vec_restore = os.path.isfile(word2vec_save_filename+".meta")
        
        word2vec = Word2Vec(tokenized_filename, stat_filename)
        word2vec.give_code()
        word2vec.tf_init(embedding_size=embedding_size, batch_size=word2vec_batch_size, seed=None)
        '''
        word2vec.tf_run(word2vec_num_step, word2vec_save_filename, restore=word2vec_restore)
        '''
        word2vec.tf_run(word2vec_num_step, word2vec_save_filename, restore=word2vec_restore)
        
        if "-g" in argdict and int(argdict["-g"]) >= 0:
            with open(stat_filename, "r", encoding='utf8') as stat_file_r, open(tokenized_filename, "r", encoding='utf8') as tokenized_file_r:
                embeddings = word2vec.Embeddings()
                gen_save_filename = join_filenames("saves", argdict["-G"])
                gen_restore = os.path.isfile(gen_save_filename+".meta")
                generator = Generator(embeddings)
                generator.nn_init(gen_batch_size, gen_seq_length, gen_hidden_size, learning_rate = 1E-04, seed=None, use_vector=("-V" in argdict))
                generator.train_real_data(int(argdict["-g"]), tokenized_file_r, gen_save_filename, restore=gen_restore)
                print(generator.generate(gen_save_filename, 2))
    
    
    
if __name__ == '__main__':
    main()