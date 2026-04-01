#!/usr/bin/env python3
"""
Adds 'Skip to content' link, CSS, and main content wrapper to existing HTML files.
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
      font-weight: bold;
    }
    .skip-link:focus {
      top: 0;
    }
"""

SKIP_LINK_HTML = '<a href="#main-content" class="skip-link">Skip to content</a>'

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if 'class="skip-link"' in content:
        print(f"Skipping {filepath} (already migrated)")
        return

    # 1. Inject CSS
    if "</style>" in content:
        content = content.replace("</style>", f"{CSS_BLOCK}\n  </style>")
    else:
        print(f"Warning: No </style> tag in {filepath}")

    # 2. Inject Skip Link
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n  {SKIP_LINK_HTML}")
    else:
        print(f"Warning: No <body> tag in {filepath}")

    # 3. Add Main Content Wrapper
    migrated_structure = False

    # Check for Hub Page pattern
    if '<div class="container">' in content and '<header>' in content:
        # Hub Page
        if '</header>' in content:
            content = content.replace('</header>', '</header>\n    <main id="main-content" tabindex="-1">')
            # Close main before the div that closes container. The container usually closes before <script>.
            # Regex to find </div> followed by optional whitespace and <script>
            # Use rsub logic (replace last occurrence) to be safe, or assume unique pattern.
            # In hub pages, </div><script> is unique to the footer area.
            content = re.sub(r'(</div>\s*<script)', r'</main>\n  \1', content, count=1)
            migrated_structure = True
            print(f"Migrated structure for Hub Page: {filepath}")

    # Check for Lesson Page pattern (page-header)
    if not migrated_structure and '<div class="page-header">' in content:
        # Lesson Page with Header
        pattern = re.compile(r'(<div class="page-header">.*?</div>)', re.DOTALL)
        if pattern.search(content):
            content = pattern.sub(r'\1\n  <main id="main-content" tabindex="-1">', content, count=1)

            # Close main before the LAST script tag (which is in body)
            last_script_idx = content.rfind('<script>')
            if last_script_idx != -1:
                content = content[:last_script_idx] + '</main>\n  ' + content[last_script_idx:]
            else:
                # Fallback: Close before body end
                content = content.replace('</body>', '</main>\n</body>')

            migrated_structure = True
            print(f"Migrated structure for Lesson Page (Header): {filepath}")

    # Check for Lesson Page pattern (Fallback - just theme toggle button)
    if not migrated_structure and 'id="theme-toggle"' in content:
        # Lesson Page without Header (just button)
        pattern = re.compile(r'(<button id="theme-toggle".*?>.*?</button>)', re.DOTALL)
        if pattern.search(content):
            content = pattern.sub(r'\1\n  <main id="main-content" tabindex="-1">', content, count=1)

            # Close main before the LAST script tag
            last_script_idx = content.rfind('<script>')
            if last_script_idx != -1:
                content = content[:last_script_idx] + '</main>\n  ' + content[last_script_idx:]
            else:
                 content = content.replace('</body>', '</main>\n</body>')

            migrated_structure = True
            print(f"Migrated structure for Lesson Page (Fallback): {filepath}")

    if not migrated_structure:
        print(f"Warning: Could not determine structure for {filepath}, added skip link but NO main wrapper.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # Find all HTML files in hubs/ and topic/
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)

    count = 0
    for f in files:
        try:
            migrate_file(f)
            count += 1
        except Exception as e:
            print(f"Error migrating {f}: {e}")
    print(f"Processed {count} files.")

if __name__ == "__main__":
    main()
