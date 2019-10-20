import os
import sys
from data.preparser import Preparser
from data.parser import Parser
from data.word2vec import Word2Vec
from generator.generator import Generator
from generator.unparser import Unparser
import gc

def join_filenames(*names):
    return os.path.join(os.path.dirname(sys.argv[0]), *names)
    
help_message = '''-h : Prints this message
-i : Initial run. Parse tweets.csv
-w N : Run Word2Vec learning N steps
-W [filename] : Load and save Word2Vec data from [filename]
-g N : Run Generator NN learning N steps
-G [filename] : Load and save Generator data from [filename]
-s N : Save ~=N sentences made by Generator.
-S [filename] : Save sentences at [filename]'''

def main():
    argdict = dict(zip(sys.argv, sys.argv[1:] + ['']))
    if "-h" in argdict:
        print(help_message)
        return
    
    #Set of filenames to data files.
    raw_filename = join_filenames("data", "tweets.csv")
    filtered_filename = join_filenames("data", "_tweets_filtered.txt")
    stat_filename = join_filenames("data", "tweets_stat.txt")
    tokenized_filename = join_filenames("data", "tweets_tokenized.txt")
    
    #Dimension of the model
    embedding_size = 128
    word2vec_batch_size = 640
    gen_batch_size = 128
    gen_seq_length = 32
    gen_hidden_size = [128, 256]

    #Hyper-parameter of the model
    learning_rate = 3E-02
    
    if "-i" in argdict:
        #Filter valid tweets from data file, and use nlp parser to tokenize tweets
        if os.path.isfile(tokenized_filename):
            proceed = (input("Erasing old data. OK to proceed? (Y/N)") == "Y")
        else:
            proceed = True
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
                
    if "-w" in argdict and int(argdict["-w"]) >= 0:
        #Start or continue word2vec optimization
        word2vec_num_step = int(argdict["-w"])
        word2vec_save_filename = join_filenames("saves", argdict["-W"])
        word2vec_restore = os.path.isfile(word2vec_save_filename+".meta")
        
        word2vec = Word2Vec(tokenized_filename, stat_filename)
        word2vec.give_code()
        word2vec.tf_init(embedding_size=embedding_size, batch_size=word2vec_batch_size, seed=None)
        word2vec.tf_run(word2vec_num_step, word2vec_save_filename, restore=word2vec_restore)
        
        if "-g" in argdict and int(argdict["-g"]) >= 0:
        #Start or continue generator learning
            with open(stat_filename, "r", encoding='utf8') as stat_file_r, open(tokenized_filename, "r", encoding='utf8') as tokenized_file_r:
                embeddings = word2vec.Embeddings()
                gen_save_filename = join_filenames("saves", argdict["-G"])
                gen_restore = os.path.isfile(gen_save_filename+".meta")
                generator = Generator(embeddings)
                generator.nn_init(gen_batch_size, gen_seq_length, gen_hidden_size, learning_rate = learning_rate, seed=None, use_vector=("-V" in argdict))
                generator.train_real_data(int(argdict["-g"]), tokenized_file_r, gen_save_filename, restore=gen_restore)
                
                if "-s" in argdict and int(argdict["-s"]) >= 0:
                    result_filename = join_filenames(argdict["-S"])
                    unparser = Unparser(result_filename)
                    sentences = generator.generate(gen_save_filename, int(argdict["-s"]))
                    unparser.save(sentences)
                
                
    
    
    
if __name__ == '__main__':
    main()