#!/usr/bin/env python3
"""Unit tests for acronyms.py: spelled-out letter names become the acronym."""
import unittest
import acronyms

VOCAB = ["LLM", "API", "GPU", "JSON", "CI", "UI", "Qwen", "build", "s3"]


class AcronymTests(unittest.TestCase):
    def test_turkish_letter_names_with_suffix(self):
        # Whisper wrote "el eleme" for both "LLM'e" and a bare "LLM"; the trailing
        # vowel cannot be told apart from a real dative, so both get the suffix.
        text = "Sonra da teksti el eleme gönderdiğinde, el eleme bu örneklerden alıp düzeltir."
        self.assertEqual(acronyms.restore(text, VOCAB),
                         "Sonra da teksti LLM'e gönderdiğinde, LLM'e bu örneklerden alıp düzeltir.")
        self.assertEqual(acronyms.restore("el el em dedim", VOCAB), "LLM dedim")

    def test_english_letter_names(self):
        self.assertEqual(acronyms.restore("Bunu ey pi ay üzerinden çağırıyoruz.", VOCAB), "Bunu API üzerinden çağırıyoruz.")
        self.assertEqual(acronyms.restore("Ci pi yu'da çalışıyor, ge pe u değil.", VOCAB), "GPU'da çalışıyor, GPU değil.")
        # "ce pe u" spells CPU, which is not in this vocabulary.
        self.assertEqual(acronyms.restore("ce pe u değil.", VOCAB), "ce pe u değil.")

    def test_single_words_and_real_phrases_are_left_alone(self):
        for text in ["Eleme turu başladı.", "Bu iş el ile yapıldı.", "Ay pi değil.",
                     "Ve onun menzili ne kadardır?", "Evet, 5. maddenin 2. kısmı var.", "LLM zaten doğru."]:
            self.assertEqual(acronyms.restore(text, VOCAB), text, text)

    def test_punctuation_between_words_ends_the_run(self):
        self.assertEqual(acronyms.restore("el, eleme", VOCAB), "el, eleme")
        self.assertEqual(acronyms.restore("(el elem)", VOCAB), "(LLM)")

    def test_only_acronyms_from_the_vocabulary_count(self):
        self.assertEqual(acronyms.restore("ce i çalışıyor", ["LLM"]), "ce i çalışıyor")
        self.assertEqual(acronyms.restore("ce i çalışıyor", ["CI"]), "CI çalışıyor")
        self.assertFalse(acronyms.is_acronym("Qwen"))
        self.assertFalse(acronyms.is_acronym("S3"))
        self.assertTrue(acronyms.is_acronym("JSON"))


if __name__ == "__main__":
    unittest.main()
