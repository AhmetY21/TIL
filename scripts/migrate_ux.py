#!/usr/bin/env python3
"""
Adds skip-link for keyboard navigation and focus-visible styling to existing HTML files.
"""
import os
import glob
import re

CSS_BLOCK = """
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: #2563eb;
      color: white;
      padding: 8px 16px;
      z-index: 100;
      transition: top 0.2s;
      text-decoration: none;
      font-weight: 600;
    }
    .skip-link:focus {
      top: 0;
    }
    *:focus-visible {
      outline: 2px solid #2563eb;
      outline-offset: 2px;
    }
    #main-content:focus {
      outline: none;
    }
"""

HUB_CSS_BLOCK = """
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: var(--primary);
      color: white;
      padding: 8px 16px;
      z-index: 100;
      transition: top 0.2s;
      text-decoration: none;
      font-weight: 600;
    }
    .skip-link:focus {
      top: 0;
    }
    *:focus-visible {
      outline: 2px solid var(--primary);
      outline-offset: 2px;
    }
    #main-content:focus {
      outline: none;
    }
"""

DARK_CSS_BLOCK = """
    .dark .skip-link {
      background: #60a5fa;
      color: #0f172a;
    }
    .dark *:focus-visible {
      outline-color: #60a5fa;
    }
"""

HUB_DARK_CSS_BLOCK = """
    .dark .skip-link {
      color: #0f172a;
    }
"""

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already migrated
    if 'class="skip-link"' in content:
        return

    is_hub = ("hubs/" in filepath or "index.html" == filepath)
    css_to_use = HUB_CSS_BLOCK if is_hub else CSS_BLOCK
    dark_css_to_use = HUB_DARK_CSS_BLOCK if is_hub else DARK_CSS_BLOCK

    # Inject CSS
    if '.theme-toggle {' in content:
        content = content.replace('.theme-toggle {', f'{css_to_use}\n    .theme-toggle {{')
    elif '</style>' in content:
        content = content.replace('</style>', f'{css_to_use}\n</style>')

    # Inject Dark CSS
    if '.dark body {' in content:
        content = content.replace('.dark body {', f'{dark_css_to_use}\n    .dark body {{')
    elif '.dark .lesson-card {' in content:
        content = content.replace('.dark .lesson-card {', f'{dark_css_to_use}\n    .dark .lesson-card {{')
    elif '.dark .hub-card {' in content:
        content = content.replace('.dark .hub-card {', f'{dark_css_to_use}\n    .dark .hub-card {{')

    # Inject skip-link after body
    if '<body>' in content:
        content = content.replace('<body>', '<body>\n  <a href="#main-content" class="skip-link">Skip to main content</a>')

    # Wrap main content
    if filepath == 'index.html':
        if '<div class="hub-grid">' in content:
            content = content.replace('<div class="hub-grid">', '<main id="main-content" tabindex="-1" class="hub-grid">')
            content = content.replace('  </div>\n\n  <footer>', '  </main>\n\n  <footer>')
    elif is_hub:
        if '<div class="container">' in content:
            content = content.replace('<div class="container">', '<main id="main-content" tabindex="-1" class="container">')
            # find matching closing div for container
            content = content.replace('  </div>\n  <script>', '  </main>\n  <script>')
    else:
        # Lesson page
        # It's wrapped after the header
        header_end = content.find('  </div>\n', content.find('<div class="page-header">')) + 9
        if header_end != 8: # If found
            content = content[:header_end] + '  <main id="main-content" tabindex="-1">\n' + content[header_end:]
        else:
            # Maybe it doesn't have page-header
            content = content.replace('<h1>', '  <main id="main-content" tabindex="-1">\n<h1>', 1)

        # Add closing main tag before script
        script_idx = content.rfind('  <script>')
        if script_idx != -1:
            content = content[:script_idx] + '  </main>\n' + content[script_idx:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)
    if os.path.exists("index.html"):
        files.append("index.html")

    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()