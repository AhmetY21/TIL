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

import generate_lesson

# Mock markdown.markdown specifically to return something predictable
# We want to simulate that markdown converts # Title to <h1>Title</h1>
# and the rest of the content follows.
def mock_markdown(text, extensions=None):
    html = ""
    lines = text.split('\n')
    for line in lines:
        if line.startswith('# '):
            html += f"<h1>{line[2:]}</h1>"
        elif line.strip():
            html += f"<p>{line}</p>"
    return html

generate_lesson.markdown.markdown = MagicMock(side_effect=mock_markdown)

from generate_lesson import convert_md_to_html

class TestReadingTime(unittest.TestCase):
    def test_reading_time_calculation(self):
        # 100 words -> 1 min (min)
        # We need a text with 100 words.
        text_100 = "word " * 100
        # 300 words -> 2 min (round(300/200) = 1.5 -> 2)
        text_300 = "word " * 300
        # 50 words -> 1 min (max(1, ...))
        text_50 = "word " * 50

        # We inject H1 so the regex replacement works
        md_100 = f"# Title\n{text_100}"
        md_300 = f"# Title\n{text_300}"
        md_50 = f"# Title\n{text_50}"

        # Case 1: 100 words -> 1 min
        html_100 = convert_md_to_html(md_100, "Title")
        self.assertIn('⏱️ 1 min read', html_100)

        # Case 2: 300 words -> 2 min
        html_300 = convert_md_to_html(md_300, "Title")
        self.assertIn('⏱️ 2 min read', html_300)

        # Case 3: 50 words -> 1 min
        html_50 = convert_md_to_html(md_50, "Title")
        self.assertIn('⏱️ 1 min read', html_50)

    def test_reading_time_css_injection(self):
        html = convert_md_to_html("# Title", "Title")
        self.assertIn('.read-time {', html)
        self.assertIn('.dark .read-time {', html)

    def test_reading_time_html_structure(self):
        html = convert_md_to_html("# Title\nContent", "Title")
        # Should be after </h1>
        # Check if </h1><div class="read-time"> pattern exists
        self.assertTrue('</h1><div class="read-time">' in html)

if __name__ == '__main__':
    unittest.main()
