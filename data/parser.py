#!/usr/bin/env python
from konlpy.tag import Twitter

class Parser():
    def __init__(self, openfile):
        self._totalcount = 0.0
        self._count = {}
        self._freq = {}
        if openfile != None:
            self.file = openfile

    def analyze(self, targetfile):
        tagger = Twitter()
        for line in self.file:
            if line == "timestamp\ttext":
                continue
            timestamp, text = tuple(line.split(\t))
            text = text.replace("\\n", "\n").strip()
            if text == "":
                continue
            tokens = tagger.pos(text, norm=True)
            for token in tokens:
                self._totalcount += 1
                if token in self._freq:
                    self._count[token] += 1
                else:
                    self._count[token] = 1
        self._freq = {token: cnt / self._totalcount for token, cnt in self._count.iteritems()}
            
