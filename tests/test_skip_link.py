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
generate_lesson.markdown.markdown = MagicMock(return_value="<p>content</p>")

from generate_lesson import convert_md_to_html, update_index_page, CurriculumMeta

class TestSkipLink(unittest.TestCase):
    def test_convert_md_to_html_has_skip_link(self):
        html = convert_md_to_html("# Title", "Title")
        self.assertIn('href="#main-content"', html)
        self.assertIn('class="skip-link"', html)
        self.assertIn('id="main-content"', html)
        self.assertIn('.skip-link {', html)
        self.assertIn('position: absolute;', html)
        self.assertIn('top: -40px;', html)

    @patch('generate_lesson.safe_write')
    @patch('glob.glob')
    def test_update_index_page_has_skip_link(self, mock_glob, mock_write):
        mock_glob.return_value = []
        meta = CurriculumMeta(
            subject="Test",
            slug="test",
            subtitle="Subtitle",
            prompt_domain="Domain",
            curriculum_file="test.json"
        )
        update_index_page(meta)
        args, _ = mock_write.call_args
        html = args[1]
        self.assertIn('href="#main-content"', html)
        self.assertIn('class="skip-link"', html)
        self.assertIn('id="main-content"', html)
        self.assertIn('.skip-link {', html)
        self.assertIn('position: absolute;', html)
        self.assertIn('top: -40px;', html)

    def test_index_html_has_skip_link(self):
        index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
        with open(index_path, "r") as f:
            html = f.read()
        self.assertIn('href="#main-content"', html)
        self.assertIn('class="skip-link"', html)
        self.assertIn('id="main-content"', html)
        self.assertIn('.skip-link {', html)
        self.assertIn('position: absolute;', html)
        self.assertIn('top: -40px;', html)

if __name__ == '__main__':
    unittest.main()
