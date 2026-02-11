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
generate_lesson.markdown.markdown = MagicMock(return_value="<p>content</p>")

from generate_lesson import convert_md_to_html, update_index_page, CurriculumMeta

class TestDarkMode(unittest.TestCase):
    def test_convert_md_to_html_has_dark_mode(self):
        html = convert_md_to_html("# Title", "Title")
        # Check for dark mode class styles
        self.assertIn(".dark body {", html)
        self.assertIn("#0f172a", html)
        # Check for toggle button
        self.assertIn('id="theme-toggle"', html)
        # Check for init script
        self.assertIn("localStorage.theme", html)

    @patch('generate_lesson.safe_write')
    @patch('glob.glob')
    def test_update_index_page_has_dark_mode(self, mock_glob, mock_write):
        mock_glob.return_value = []

        meta = CurriculumMeta(
            subject="Test",
            slug="test",
            subtitle="Subtitle",
            prompt_domain="Domain",
            curriculum_file="test.json"
        )

        update_index_page(meta)

        # Check what was written
        self.assertTrue(mock_write.called, "safe_write should be called")
        args, _ = mock_write.call_args
        content = args[1]

        self.assertIn(".dark {", content)
        self.assertIn("#0f172a", content)
        self.assertIn('id="theme-toggle"', content)
        self.assertIn("localStorage.theme", content)

if __name__ == '__main__':
    unittest.main()
