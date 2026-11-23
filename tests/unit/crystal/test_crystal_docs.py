import unittest

from tests._doc_examples import DocExamplesBoilerPlate


class TestCrystalDocExamples(unittest.TestCase):
    def test_doc_examples(self):
        DocExamplesBoilerPlate("crystal", __file__).run()


if __name__ == "__main__":
    unittest.main()
