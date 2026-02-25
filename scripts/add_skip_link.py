#!/usr/bin/env python3
"""
Adds 'Skip to content' link for accessibility to existing HTML files.
"""
import os
import glob

CSS_BLOCK = """
    /* Skip to Content Link */
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: #2563eb;
      color: white;
      padding: 8px;
      z-index: 100;
      transition: top 0.2s;
      font-weight: bold;
      text-decoration: none;
      border-radius: 0 0 4px 0;
    }
    .skip-link:focus {
      top: 0;
    }
    .dark .skip-link {
      background: #60a5fa;
      color: #0f172a;
    }
"""

LINK_HTML = """  <a href="#main-content" class="skip-link">Skip to content</a>"""
ANCHOR_HTML = """<div id="main-content" tabindex="-1"></div>"""

def process_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already has skip link
    if 'class="skip-link"' in content:
        print(f"Skipping {filepath} (already has skip link)")
        return

    # 1. Inject CSS
    if "</style>" in content:
        content = content.replace("</style>", f"{CSS_BLOCK}\n  </style>")
    else:
        print(f"Warning: No <style> tag found in {filepath}")

    # 2. Inject Link after <body>
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n{LINK_HTML}")
    else:
        print(f"Warning: No <body> tag found in {filepath}")

    # 3. Inject Anchor Target
    # Priority based on file type / structure
    injected_anchor = False

    # For index.html -> before .hub-grid
    if '<div class="hub-grid">' in content:
        content = content.replace('<div class="hub-grid">', f'{ANCHOR_HTML}\n  <div class="hub-grid">')
        injected_anchor = True

    # For Hub pages (hubs/*.html) -> before <header>
    # They usually have <div class="container"> ... <header>
    elif '<header>' in content and "hubs/" in filepath:
        content = content.replace('<header>', f'{ANCHOR_HTML}\n    <header>')
        injected_anchor = True

    # For Lesson pages (topic/*.html) -> before <h1>
    # They usually have <div class="page-header"> ... <h1>
    elif '<h1' in content:
        # We replace the first occurrence of <h1
        # This is a bit risky if there are h1s elsewhere, but usually the first h1 is the main title
        # We need to be careful not to break the tag
        content = content.replace('<h1', f'{ANCHOR_HTML}\n<h1', 1)
        injected_anchor = True

    if not injected_anchor:
        print(f"Warning: Could not determine where to inject anchor in {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filepath}")

def main():
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)
    if os.path.exists("index.html"):
        files.append("index.html")

    print(f"Found {len(files)} HTML files to process.")
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()
