import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before import
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["markdown"] = MagicMock()
sys.modules["slugify"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

import os
# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the module under test
import generate_lesson

# Mock markdown.markdown specifically to return something predictable
# We need it to return HTML with an <h1> tag so our injection logic works
generate_lesson.markdown.markdown = MagicMock(return_value="<h1>Title</h1><p>Some content here.</p>")

from generate_lesson import convert_md_to_html

class TestReadingTime(unittest.TestCase):
    def test_convert_md_to_html_has_reading_time(self):
        # A simple text with few words -> 1 min read
        md_text = "word " * 50
        html = convert_md_to_html(md_text, "Title")

        # Check for CSS styles
        self.assertIn(".reading-time", html)

        # Check for injected HTML
        # The mocked markdown returns "<h1>Title</h1><p>..."
        # So we expect the reading time to be inserted after </h1>
        expected_fragment = '</h1>\n<p class="reading-time">⏱️ 1 min read</p>'
        self.assertIn(expected_fragment, html)

    def test_convert_md_to_html_reading_time_calculation(self):
        # 400 words -> 2 min read (assuming 200 wpm)
        md_text = "word " * 400

        html = convert_md_to_html(md_text, "Title")
        self.assertIn('2 min read', html)

    def test_convert_md_to_html_reading_time_long(self):
        # 1000 words -> 5 min read
        md_text = "word " * 1000

        html = convert_md_to_html(md_text, "Title")
        self.assertIn('5 min read', html)

if __name__ == '__main__':
    unittest.main()
