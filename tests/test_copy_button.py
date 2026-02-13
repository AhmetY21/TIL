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
generate_lesson.markdown.markdown = MagicMock(return_value="<pre><code>print('hello')</code></pre>")

from generate_lesson import convert_md_to_html

class TestCopyButton(unittest.TestCase):
    def test_convert_md_to_html_has_copy_button_logic(self):
        html = convert_md_to_html("# Title", "Title")

        # Check for CSS styles
        self.assertIn(".copy-button", html)
        self.assertIn("opacity: 0", html) # Ensure it's hidden by default
        self.assertIn("pre:hover .copy-button", html) # Ensure hover reveals it

        # Check for JS logic
        self.assertIn("document.querySelectorAll('pre')", html)
        self.assertIn("createElement('button')", html)
        self.assertIn("navigator.clipboard.writeText", html)
        self.assertIn("textContent = 'Copied!'", html)

if __name__ == '__main__':
    unittest.main()
