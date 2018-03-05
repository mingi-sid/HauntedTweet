import unittest
import os
import sys
from data.preparser import Preparser

class PreparserTest(unittest.TestCase):
    def test_Preparser_filter_off(self):
        with open(os.path.join(os.path.dirname(sys.argv[0]), "test_preparser.csv")) as fr:
            p = Preparser(fr)
            p.extract(filter=False)
            with open(os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_result_off.txt"), "w") as fwresult:
                p.save(fwresult)
        result = os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_result_off.txt")
        compare = os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_compare_off.txt")
        with open(result) as fresult:
            with open(compare) as fcompare:
                for lineresult in fresult:
                    linecompare = fcompare.readline()
                    self.assertEqual(lineresult, linecompare)

    def test_Preparser_filter_on(self):
        with open(os.path.join(os.path.dirname(sys.argv[0]), "test_preparser.csv")) as fr:
            p = Preparser(fr)
            p.extract(filter=True)
            with open(os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_result_on.txt"), "w") as fwresult:
                p.save(fwresult)
        result = os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_result_on.txt")
        compare = os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_compare_on.txt")
        with open(result) as fresult:
            with open(compare) as fcompare:
                for lineresult in fresult:
                    linecompare = fcompare.readline()
                    self.assertEqual(lineresult, linecompare)

    def tearDown(self):
#os.remove(os.path.join(os.path.dirname(sys.argv[0]), "test_preparser_result.txt"))
        pass

if __name__ == '__main__':
    unittest.main()
