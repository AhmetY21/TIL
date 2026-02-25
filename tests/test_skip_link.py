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

class TestSkipLink(unittest.TestCase):
    def test_convert_md_to_html_has_skip_link(self):
        html = convert_md_to_html("# Title", "Title")

        # Check for skip link
        self.assertIn('<a href="#main-content" class="skip-link">Skip to content</a>', html)

        # Check for main content wrapper with correct ID and tabindex
        self.assertIn('<main id="main-content" tabindex="-1">', html)
        self.assertIn('</main>', html)

        # Check for CSS
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

        # Check what was written
        self.assertTrue(mock_write.called, "safe_write should be called")
        args, _ = mock_write.call_args
        content = args[1]

        # Check for skip link
        self.assertIn('<a href="#main-content" class="skip-link">Skip to content</a>', content)

        # Check for main content wrapper with correct ID and tabindex
        self.assertIn('<main id="main-content" tabindex="-1">', content)
        self.assertIn('</main>', content)

        # Check for CSS
        self.assertIn('.skip-link {', content)

    def test_index_html_has_skip_link(self):
        # Read the actual index.html file
        index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for skip link
            self.assertIn('<a href="#main-content" class="skip-link">Skip to content</a>', content)

            # Check for target ID
            self.assertIn('id="main-content"', content)
            self.assertIn('tabindex="-1"', content)

            # Check for CSS
            self.assertIn('.skip-link {', content)
        else:
            self.fail("index.html not found")

if __name__ == '__main__':
    unittest.main()
