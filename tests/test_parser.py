#!/usr/bin/env python
import unittest
import os
import sys
from data.parser import Parser

class ParserTest(unittest.TestCase):
    def test_Parser_freq(self):
        with open(os.path.join(os.path.dirname(sys.argv[0]), "test_parser.txt")) as fr:
            p = Parser(fr)
            with open(os.path.join(os.path.dirname(sys.argv[0]), "test_parser_result.txt"), "w") as fw:
                p.get_stats(fw)
        result = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_result.txt")
        compare = os.path.join(os.path.dirname(sys.argv[0]), "test_parser_compare.txt")
        with open(result) as fresult:
            with open(compare) as fcompare:
                for lineresult in fresult:
                    linecompare = fcompare.readline()
                    self.assertEqual(lineresult, linecompare)

    def test_Parser_freq(self):
        with open(os.path.join(os.path.dirname(sys.argv[0]), "test_parser.txt")) as fr:
            p = Parser(fr)
            with open(os.path.join(os.path.dirname(sys.argv[0]), "test_parser_dataset.txt"), "w") as fw:
                p.get_dataset(fw, 3)

if __name__ == '__main__':
    unittest.main()
