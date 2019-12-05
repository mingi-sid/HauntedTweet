import os
import sys
from data.preparser import Preparser
from data.parser import Parser
from data.word2vec import Word2Vec
from generator.generator import Generator
from generator.unparser import Unparser
import gc
import configparser

def join_filenames(*names):
    return os.path.join(os.path.dirname(sys.argv[0]), *names)

def open_utf8(filename, option):
    return open(filename, option, encoding='utf8')

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
    raw_filename = join_filenames("data", "tweets.csv")
    filtered_filename = join_filenames("data", "_tweets_filtered.txt")
    stat_filename = join_filenames("data", "tweets_stat.txt")
    tokenized_filename = join_filenames("data", "tweets_tokenized.txt")
    
    session_config = configparser.ConfigParser()
    session_config.read('session.ini')

    word2vec_batch_size = 640
    embedding_size = int(session_config['dimension']['embedding_size'])
    gen_batch_size = 128
    gen_seq_length = int(session_config['dimension']['gen_seq_length'])
    gen_hidden_size = [int(x) for x in session_config['dimension']['gen_hidden_size'].split(',')]
    learning_rate = 1E-03
    

    if "-i" in argdict:
        proceed = True
        if os.path.isfile(tokenized_filename):
            proceed = (input("Erasing old data. OK to proceed? (Y/N)") == "Y")
        if proceed:
            with open_utf8(raw_filename, "r") as raw_file_r:
                #Filter actual tweets
                preparser = Preparser(raw_file_r)
                preparser.extract(filter=True)
                with open_utf8(filtered_filename, "w") as filtered_file_w:
                    preparser.save(filtered_file_w)
                
                #Tokenize tweets
                with open_utf8(filtered_filename, "r") as filtered_file_r:
                    parser = Parser(filtered_file_r)
                    with open_utf8(stat_filename, "w") as stat_file_w:
                        parser.get_stats(stat_file_w)
                    with open_utf8(tokenized_filename, "w") as tokenized_file_w:
                        parser.get_data(tokenized_file_w)
                del parser
                
    if "-w" in argdict and int(argdict["-w"]) >= 0:
        word2vec_num_step = int(argdict["-w"])
        if "-W" in argdict:
            word2vec_save_filename = join_filenames("saves", argdict["-W"])
        else:
            word2vec_save_filename = join_filenames(
                "saves", session_config['save_file']['word2vec_save'])
        word2vec_restore = os.path.isfile(word2vec_save_filename+".meta")
        
        word2vec = Word2Vec(tokenized_filename, stat_filename)
        word2vec.give_code()
        word2vec.tf_init(embedding_size=embedding_size,
                         batch_size=word2vec_batch_size, seed=None)
        word2vec.tf_run(word2vec_num_step, word2vec_save_filename, restore=word2vec_restore)
        
        if "-g" in argdict and int(argdict["-g"]) >= 0:
            with open_utf8(stat_filename, "r") as stat_file_r, open_utf8(tokenized_filename, "r") as tokenized_file_r:
                embeddings = word2vec.Embeddings()
                if "-G" in argdict:
                    gen_save_filename = join_filenames("saves", argdict["-G"])
                else:
                    gen_save_filename = join_filenames(
                        "saves", session_config['save_file']['generator_save'])
                gen_restore = os.path.isfile(gen_save_filename+".meta")
                generator = Generator(embeddings)
                generator.nn_init(
                    gen_batch_size, gen_seq_length, gen_hidden_size,
                    learning_rate=learning_rate, seed=None,
                    use_vector=("-V" in argdict))
                generator.train_real_data(int(argdict["-g"]), tokenized_file_r,
                    gen_save_filename, restore=gen_restore)
                
                if "-s" in argdict and int(argdict["-s"]) >= 0:
                    result_filename = join_filenames(argdict["-S"])
                    unparser = Unparser(result_filename)
                    sentences = generator.generate(gen_save_filename,
                                                   int(argdict["-s"]))
                    unparser.save(sentences)
# end main()                
    
if __name__ == '__main__':
    main()