#!/usr/bin/env python
import os.path
import sys
class Unparser():
    def __init__(self, filename):
        self.filename = filename
        self.delimiter = "========"
        
    def unparse(self, sentences):
        tags_to_glue = ['Token', 'Josa', 'Eomi', 'Punctuation', 'Suffix', 'PreEomi']
        replace_dict = {'<go>' : '', '<eos>' : ''}
        
        result = []
        for sentence in sentences:
            string = ""
            for token in sentence:
                tagged_word = token
                word, tag = eval(tagged_word)
                replaced_word = replace_dict[word] if (word in replace_dict) else word
                
                if tag in tags_to_glue:
                    string += replaced_word
                else:
                    string += " " + replaced_word
                    
            string = '\n'.join([x.strip() for x in string.split('\n') if x != '']).strip()
            result.append(string)
        return result
                
    def save(self, sentences):
        with open(os.path.join(os.path.dirname(sys.argv[0]), self.filename), 'a', encoding='utf8') as f:
            result = self.unparse(sentences)
            for string in result:
                f.write(string)
                f.write('\n' + self.delimiter + '\n')