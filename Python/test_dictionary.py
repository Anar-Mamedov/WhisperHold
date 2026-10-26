#!/usr/bin/env python3
"""dictionary.py answers None without PyObjC and real answers with it."""
import unittest
import dictionary


class DictionaryTests(unittest.TestCase):
    def test_answers_or_none(self):
        answer = dictionary.is_word("kitap")
        if answer is None:
            self.assertFalse(dictionary.available())
        else:
            self.assertTrue(answer)
            self.assertFalse(dictionary.is_word("Giral"))
            self.assertTrue(dictionary.is_word("çalışırken"))
            self.assertTrue(dictionary.is_word("endpoint"))   # English dictionary
            self.assertTrue(dictionary.is_turkish_word("komite"))
            self.assertFalse(dictionary.is_turkish_word("commit"))

    def test_punctuation_and_numbers_are_not_misspellings(self):
        if dictionary.available():
            self.assertTrue(dictionary.is_word("3."))
            self.assertTrue(dictionary.is_word("kitap,"))


if __name__ == "__main__":
    unittest.main()
