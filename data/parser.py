#!/usr/bin/env python
from konlpy.tag import Twitter
import random

class Parser():
    def __init__(self, openfile):
        self._totalcount = 0.0
        self._count = {}
        self._freq = {}
        self._tokens = []
        if openfile != None:
            self.file = openfile

    def get_stats(self, targetfile):
        tagger = Twitter()
        for line in self.file:
            if line == "timestamp\ttext":
                continue
            timestamp, text = tuple(line.split("\t"))
            text = text.replace("\\n", "\n").strip()
            if text == "":
                continue
            tokens = tagger.pos(text, norm=True)
            tokens = [('<go>', 'Token')] + tokens + [('<eos>', 'Token')]
            self._tokens.append( [str(token) for token in tokens] )
            for token in tokens:
                self._totalcount += 1
                if token in self._count:
                    self._count[token] += 1
                else:
                    self._count[token] = 1
        self._freq = {token: cnt / self._totalcount for token, cnt in self._count.items()}
        for key in sorted(sorted(self._freq.keys()), key=self._freq.__getitem__, reverse=True):
            targetfile.write(str(key) + "\t" + str(self._freq[key]) + "\n")

    def get_data(self, targetfile):
        if self._tokens == []:
            tagger = Twitter()
            self.file.seek(0)
            for line in self.file:
                timestamp, text = tuple(line.split("\t"))
                text = text.replace("\\n", "\n").strip()
                if text == "":
                    continue
                tokens = tagger.pos(text, norm=True)
                tokens = [('<go>', 'Token')] + tokens + [('<eos>', 'Token')]
                self._tokens.append( [str(token) for token in tokens] )
        #random.shuffle(self._tokens)
        for line in self._tokens:
            targetfile.write("\t".join(line) + "\n")
