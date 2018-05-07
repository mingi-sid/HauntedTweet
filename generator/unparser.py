#!/usr/bin/env python
import os.path
import sys
class Unparser():
    def __init__(self, filename):
        self.filename = filename
        self.delimiter = "========"
        
    def unparse(self, sentences):
        words_to_compress = ["('<go>', 'Token')", "('<eos>', 'Token')", "('……', 'Foreign')"]
        tags_to_glue = ['Token', 'Josa', 'Eomi', 'Punctuation', 'Suffix', 'PreEomi']
        replace_dict = {'<go>' : '', '<eos>' : '\n'}
        
        result = []
        for sentence in sentences:
            prev_token = ""
            word_count = []
            cnt = 1
            for token in sentence:
                if prev_token == token:
                    cnt += 1
                else:
                    word_count.append( (prev_token, cnt) )
                    cnt = 1
                prev_token = token
            word_count.append( (token, cnt) )
            word_count = word_count[1:]
            
            #Compress words
            sentence_compressed = [ ((token[0], 1) if token[0] in words_to_compress else token) for token in word_count]
            
            #Make into string
            string = ""
            maximum_sentence_count = 3
            sentence_count = maximum_sentence_count
            for token in sentence_compressed:
                #print(token)
                tagged_word, count = token
                word, tag = eval(tagged_word)
                replaced_word = replace_dict[word] if (word in replace_dict) else word
                
                if word == '<eos>':
                    sentence_count -= 1
                if sentence_count <= 0:
                    break
                    
                
                if tag in tags_to_glue:
                    string += replaced_word
                else:
                    string += " " + replaced_word
                    
                if count > 1:
                    string += "*" + str(count)
                    
            string = '\n'.join([x.strip() for x in string.split('\n') if x != '']).strip()
            result.append(string)
        return result
                
    def save(self, sentences):
        with open(os.path.join(os.path.dirname(sys.argv[0]), self.filename), 'a', encoding='utf8') as f:
            result = self.unparse(sentences)
            for string in result:
                f.write(string)
                f.write('\n' + self.delimiter + '\n')