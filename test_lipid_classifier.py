import unittest
from lipid_classifier import (
    classify_lipid,
    extract_carbon_number,
    extract_unsaturation,
    extract_total_cn,
    extract_total_unsat,
    extract_total_cn_unsat,
)

class TestLipidClassifier(unittest.TestCase):
    def test_caep(self):
        s = 'CAEP(d18:2/27:4)(2OH)'
        self.assertEqual(classify_lipid(s), 'CAEP(d18:2)(2OH)')
        self.assertEqual(extract_carbon_number(s), 18)
        self.assertEqual(extract_unsaturation(s), 2)
        # total: 18+27, 2+4
        self.assertEqual(extract_total_cn(s), 45)
        self.assertEqual(extract_total_unsat(s), 6)
        self.assertEqual(extract_total_cn_unsat(s), (45, 6))

    def test_cer(self):
        s = 'Cer(d14:0/20:0)'
        self.assertEqual(classify_lipid(s), 'Cer(d14:0)')
        self.assertEqual(extract_carbon_number(s), 14)
        self.assertEqual(extract_unsaturation(s), 0)
        self.assertEqual(extract_total_cn(s), 34)
        self.assertEqual(extract_total_unsat(s), 0)

    def test_fmc(self):
        s = 'FMC-4(d18:0/15:2)'
        self.assertEqual(classify_lipid(s), 'FMC-4(d18:0)')
        self.assertEqual(extract_carbon_number(s), 18)
        self.assertEqual(extract_unsaturation(s), 0)
        self.assertEqual(extract_total_cn(s), 33)
        self.assertEqual(extract_total_unsat(s), 2)

    def test_ga2(self):
        s = 'GA2(d18:1/16:0)'
        self.assertEqual(classify_lipid(s), 'GA2(d18:1)')
        self.assertEqual(extract_carbon_number(s), 18)
        self.assertEqual(extract_unsaturation(s), 1)
        self.assertEqual(extract_total_cn(s), 34)
        self.assertEqual(extract_total_unsat(s), 1)

    def test_modifiers_multi(self):
        s = 'Cer(d18:1/24:0)(2OH)(acetyl)'
        self.assertEqual(classify_lipid(s), 'Cer(d18:1)(2OH)(acetyl)')

if __name__ == '__main__':
    unittest.main(verbosity=2)
