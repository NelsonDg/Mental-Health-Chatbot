import os
import sys

os.environ.setdefault('GEMINI_API_KEY', 'test-placeholder-key')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from app import AITherapyBot


class TestEmotionDetection(unittest.TestCase):
    def setUp(self):
        self.bot = AITherapyBot()

    def test_detects_happy(self):
        result = self.bot.analyze_emotional_state("I'm feeling really happy today")
        self.assertEqual(result['primary_emotion'], 'happy')

    def test_detects_anxious(self):
        result = self.bot.analyze_emotional_state("I'm so anxious about my exam, I can't relax")
        self.assertEqual(result['primary_emotion'], 'anxious')

    def test_detects_grief(self):
        result = self.bot.analyze_emotional_state("my dog died yesterday and I miss him so much")
        self.assertEqual(result['primary_emotion'], 'grief')

    def test_neutral_default(self):
        result = self.bot.analyze_emotional_state("I need to buy groceries and do laundry today")
        self.assertEqual(result['primary_emotion'], 'neutral')


class TestCrisisDetection(unittest.TestCase):
    def setUp(self):
        self.bot = AITherapyBot()

    def test_detects_direct_crisis_phrase(self):
        result = self.bot.analyze_emotional_state("I want to end my life")
        self.assertTrue(result['crisis_indicators'])

    def test_detects_expanded_crisis_phrase(self):
        result = self.bot.analyze_emotional_state("honestly there is no reason to go on anymore")
        self.assertTrue(result['crisis_indicators'])

    def test_idiom_is_not_false_positive(self):
        result = self.bot.analyze_emotional_state("I need to cut myself some slack today, work was rough")
        self.assertFalse(result['crisis_indicators'])

    def test_non_crisis_message_has_no_indicators(self):
        result = self.bot.analyze_emotional_state("I had a pretty good day overall")
        self.assertFalse(result['crisis_indicators'])


if __name__ == '__main__':
    unittest.main()
