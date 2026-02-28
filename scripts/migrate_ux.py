import os
import glob
import re

def migrate_html_files():
    files = glob.glob('hubs/**/*.html', recursive=True) + glob.glob('topic/**/*.html', recursive=True)
    count = 0

    lesson_css_injection = """
    /* Accessibility: Focus styles */
    a:focus-visible, button:focus-visible {
      outline: 2px solid #2563eb;
      outline-offset: 2px;
      border-radius: 4px;
    }
    .theme-toggle:focus-visible {
      border-radius: 50%;
    }

    .dark .theme-toggle:hover {
"""

    lesson_dark_css_injection = """
    .dark a:focus-visible, .dark button:focus-visible {
      outline-color: #60a5fa;
    }

    .dark body {"""

    index_css_injection = """
    /* Accessibility: Focus styles */
    a:focus-visible, button:focus-visible {
      outline: 2px solid var(--primary);
      outline-offset: 2px;
      border-radius: 4px;
    }
    .theme-toggle:focus-visible {
      border-radius: 50%;
    }
    .lesson-card:focus-visible {
      border-radius: 14px;
    }
    .btn:focus-visible {
      border-radius: 10px;
    }

    .dark .theme-toggle:hover {"""

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'focus-visible' in content:
            continue

        is_index = 'index.html' in filepath

        if is_index:
            content = content.replace('.dark .theme-toggle:hover {', index_css_injection)
        else:
            content = content.replace('.dark .theme-toggle:hover {', lesson_css_injection)
            content = content.replace('.dark body {', lesson_dark_css_injection)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1

    print(f"Migrated {count} HTML files to add focus styles.")

if __name__ == "__main__":
    migrate_html_files()