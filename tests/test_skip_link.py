import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Mock dependencies before import
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["markdown"] = MagicMock()
sys.modules["slugify"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the module under test
import generate_lesson

# Mock markdown.markdown specifically to return something predictable
generate_lesson.markdown.markdown = MagicMock(return_value="<p>content</p>")

from generate_lesson import convert_md_to_html, update_index_page, CurriculumMeta

class TestSkipLink(unittest.TestCase):
    def test_index_html_has_skip_link(self):
        """Test that the static index.html has the skip link."""
        index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
        if not os.path.exists(index_path):
            self.fail("index.html not found")

        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('class="skip-link"', html)
        self.assertIn('href="#main-content"', html)
        self.assertIn('id="main-content"', html)
        self.assertIn('.skip-link', html) # CSS check

    def test_convert_md_to_html_has_skip_link(self):
        """Test that generated lesson pages have the skip link."""
        html = convert_md_to_html("# Title", "Title")

        self.assertIn('class="skip-link"', html)
        self.assertIn('href="#main-content"', html)
        self.assertIn('id="main-content"', html)
        self.assertIn('.skip-link', html) # CSS check

    @patch('generate_lesson.safe_write')
    @patch('glob.glob')
    def test_update_index_page_has_skip_link(self, mock_glob, mock_write):
        """Test that generated index pages have the skip link."""
        mock_glob.return_value = []

        meta = CurriculumMeta(
            subject="Test",
            slug="test",
            subtitle="Subtitle",
            prompt_domain="Domain",
            curriculum_file="test.json"
        )

        update_index_page(meta)

        self.assertTrue(mock_write.called, "safe_write should be called")
        args, _ = mock_write.call_args
        content = args[1]

        self.assertIn('class="skip-link"', content)
        self.assertIn('href="#main-content"', content)
        self.assertIn('id="main-content"', content)
        self.assertIn('.skip-link', content) # CSS check

if __name__ == '__main__':
    unittest.main()
