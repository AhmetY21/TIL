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
generate_lesson.markdown.markdown = MagicMock(return_value="<p>Lesson Content</p>")

from generate_lesson import convert_md_to_html, update_index_page, CurriculumMeta

class TestSkipLink(unittest.TestCase):
    def test_convert_md_to_html_has_skip_link(self):
        html = convert_md_to_html("# Title", "Title")

        # Check for CSS styles
        self.assertIn(".skip-link", html)
        self.assertIn("top: -9999px", html)
        self.assertIn("top: 0", html)

        # Check for HTML elements
        self.assertIn('<a href="#main-content" class="skip-link">Skip to content</a>', html)
        self.assertIn('<main id="main-content">', html)
        self.assertIn('<p>Lesson Content</p>', html)
        self.assertIn('</main>', html)

    @patch('generate_lesson.safe_write')
    @patch('generate_lesson.glob.glob')
    def test_update_index_page_has_skip_link(self, mock_glob, mock_safe_write):
        # Setup
        mock_glob.return_value = [] # No existing lessons
        meta = CurriculumMeta(
            subject="Test Subject",
            slug="test-subject",
            subtitle="Test Subtitle",
            prompt_domain="Test Domain",
            curriculum_file="test.json"
        )

        # Action
        update_index_page(meta)

        # Assertion
        self.assertTrue(mock_safe_write.called)
        # Get the content passed to safe_write
        # safe_write(path, content)
        args, _ = mock_safe_write.call_args
        content = args[1]

        # Check for CSS styles
        self.assertIn(".skip-link", content)
        self.assertIn("top: -9999px", content)

        # Check for HTML elements
        self.assertIn('<a href="#main-content" class="skip-link">Skip to content</a>', content)
        self.assertIn('<header id="main-content">', content)

if __name__ == '__main__':
    unittest.main()
