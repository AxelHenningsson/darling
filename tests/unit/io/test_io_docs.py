import unittest

from tests._doc_examples import DocExamplesBoilerPlate


class TestIODocExamples(unittest.TestCase):
    def test_doc_examples(self):
        DocExamplesBoilerPlate("io", __file__).run()


if __name__ == "__main__":
    unittest.main()
