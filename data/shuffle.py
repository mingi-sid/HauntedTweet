#!/usr/bin/env python
import sys, os
import random

with open(os.path.join(os.path.dirname(sys.argv[0]), 'tweets_tokenized2.txt'), "r", encoding='utf8') as f:
    with open(os.path.join(os.path.dirname(sys.argv[0]), 'tweets_tokenized_shuffled.txt'), 'w', encoding='utf8') as fw:
        lines = f.readlines()
        f.seek(0)
        print(f.readline())
        random.shuffle(lines)
        for line in lines:
            fw.write(line)
