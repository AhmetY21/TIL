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

# Mock markdown.markdown specifically to return something with <h1>
generate_lesson.markdown.markdown = MagicMock(return_value="<h1>Main Title</h1><p>Some content...</p>")

from generate_lesson import convert_md_to_html, CurriculumMeta

class TestReadingTime(unittest.TestCase):
    def test_reading_time_injection(self):
        # 400 words -> 2 min
        md_text = "word " * 400
        title = "Test Lesson"

        html = convert_md_to_html(md_text, title)

        # Check for reading time text
        self.assertIn("2 min read", html)

        # Check injection position: should be after </h1>
        self.assertIn("</h1><div class=\"reading-time\">", html)

        # Check for icon
        self.assertIn("⏱️", html)

    def test_reading_time_calculation_short(self):
        # 50 words -> 1 min
        md_text = "word " * 50
        html = convert_md_to_html(md_text, "Title")
        self.assertIn("1 min read", html)

    def test_reading_time_calculation_long(self):
        # 205 words -> 2 min (ceil)
        md_text = "word " * 205
        html = convert_md_to_html(md_text, "Title")
        self.assertIn("2 min read", html)

    def test_css_styles_present(self):
        html = convert_md_to_html("content", "Title")
        self.assertIn(".reading-time {", html)
        self.assertIn("color: #6b7280;", html)
        self.assertIn(".dark .reading-time {", html)
        self.assertIn("color: #94a3b8;", html)

if __name__ == '__main__':
    unittest.main()
