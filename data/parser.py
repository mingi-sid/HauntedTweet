#!/usr/bin/env python
from konlpy.tag import Twitter

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
            self._tokens.append(tokens[:])
            for token in tokens:
                self._totalcount += 1
                if token in self._count:
                    self._count[token] += 1
                else:
                    self._count[token] = 1
        self._freq = {token: cnt / self._totalcount for token, cnt in self._count.items()}
        for key in sorted(self._freq.keys()):
            targetfile.write(str(key) + "\t" + str(self._freq[key]) + "\n")

    def get_dataset(self, targetfile, window_size = 3):
        if self._tokens == []:
            tagger = Twitter()
            self.file.seek(0)
            for line in self.file:
                if line == "timestamp\ttext":
                    continue
                timestamp, text = tuple(line.split("\t"))
                text = text.replace("\\n", "\n").strip()
                if text == "":
                    continue
                tokens = tagger.pos(text, norm=True)
                self._tokens.append(tokens[:])
        for line in self._tokens:
            for i in range(len(line)):
                for j in range(window_size):
                    if i-j-1 < 0:
                        break
                    targetfile.write(str(line[i]) + "\t" + str(line[i-j-1]) + "\n")
                for j in range(window_size):
                    if i+j+1 >= len(line):
                        break
                    targetfile.write(str(line[i]) + "\t" + str(line[i+j+1]) + "\n")
                
