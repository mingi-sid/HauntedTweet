#!/usr/bin/env python
import unittest
import os
import sys
from generator.unparser import Unparser

class UnparserTest(unittest.TestCase):
    def test_Unparser_stats(self):
        filename = "test_unparser.txt"
        data = [ "('<go>', 'Token')	('<go>', 'Token')	('하루', 'Noun')	('라도', 'Josa')	('전', 'Noun')	('에', 'Josa')	('지진', 'Noun')	('이', 'Josa')	('있었', 'Adjective')	('으면', 'Eomi')	('모르는', 'Verb')	('데', 'Eomi')	('고사', 'Noun')	('장', 'Suffix')	('안전', 'Noun')	('도', 'Josa')	('확인', 'Noun')	('이', 'Josa')	('안', 'Noun')	('된', 'Verb')	('상황', 'Noun')	('은', 'Josa')	('곤란하긴', 'Adjective')	('하지', 'Verb')	('<eos>', 'Token')".split('\t'),
        "('<go>', 'Token')	('일단', 'Noun')	('아', 'Exclamation')	('무', 'Noun')	('상관', 'Noun')	('없지', 'Adjective')	('만', 'Eomi')	('이', 'Determiner')	('거', 'Noun')	('보고', 'Noun')	('갑', 'Verb')	('시', 'PreEomi')	('다', 'Eomi')	('<eos>', 'Token')	('추가', 'Noun')	(\"?'\", 'Punctuation')	('<eos>', 'Token')".split('\t'),
        ["('서', 'Eomi')", "('서', 'Eomi')", "('서', 'Eomi')", "('<eos>', 'Token')"],
        "('<go>', 'Token')	('악', 'Noun')	('탄산음료', 'Noun')	('<eos>', 'Token')	('<go>', 'Token')	('<eos>', 'Token')	('<go>', 'Token')	('<eos>', 'Token')	('<go>', 'Token')".split('\t'),
        "('<go>', 'Token')	('악', 'Noun')	('탄산음료', 'Noun')	('<eos>', 'Token')	('탄산음료', 'Noun')	('<eos>', 'Token')	('탄산음료', 'Noun')	('<eos>', 'Token')	('탄산음료', 'Noun')	('<eos>', 'Token')	('탄산음료', 'Noun')	('<eos>', 'Token')".split('\t')]
        compare = [ "하루라도 전에 지진이 있었으면 모르는데 고사장 안전도 확인이 안 된 상황은 곤란하긴 하지",
        "일단 아 무 상관 없지만 이 거 보고 갑시다\n추가?'",
        "서*3",
        "악 탄산음료",
        "악 탄산음료\n탄산음료\n탄산음료"]
        u = Unparser(filename)
        result = u.unparse(data)
        self.assertEqual(result, compare)
        u.save(data)

if __name__ == '__main__':
    unittest.main()
