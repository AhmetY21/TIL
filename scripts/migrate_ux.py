#!/usr/bin/env python3
"""
Adds Skip Link and Main Landmark to existing HTML files.
"""
import os
import glob
import re

SKIP_LINK_CSS = """
    .skip-link {
      position: absolute;
      top: -9999px;
      left: 0;
      background: #ffffff;
      padding: 8px;
      z-index: 1000;
      text-decoration: none;
      color: #2563eb;
      border: 1px solid #e5e7eb;
      border-radius: 0 0 6px 0;
    }
    .skip-link:focus {
      top: 0;
    }
"""

SKIP_LINK_HTML = """  <a href="#main-content" class="skip-link">Skip to main content</a>"""

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already migrated
    if 'class="skip-link"' in content:
        print(f"Skipping {filepath} (already migrated)")
        return

    # 1. Inject CSS before </style>
    if "</style>" in content and ".skip-link" not in content:
        content = content.replace("</style>", f"{SKIP_LINK_CSS}\n  </style>")

    # 2. Inject Skip Link after <body>
    if "<body>" in content and 'class="skip-link"' not in content:
        content = content.replace("<body>", f"<body>\n{SKIP_LINK_HTML}")

    # 3. Add <main> wrapper
    # Strategy depends on file type

    # Hub pages (in hubs/)
    if "hubs/" in filepath:
        # Wrap content: <main> <header>...</header> ... </main> inside .container
        if "<header>" in content and '<main id="main-content"' not in content:
             content = content.replace("<header>", '<main id="main-content" tabindex="-1">\n    <header>')

             # Close main before the last closing div (which closes .container)
             # Use regex to find the last div before script
             content = re.sub(r"</div>\s*<script>", "</main>\n  </div>\n  <script>", content)


    # Lesson pages (in topic/)
    elif "topic/" in filepath:
        if '<div class="page-header">' in content and '<main id="main-content"' not in content:
            # Use regex to find the closing div of page-header
            # Since page-header doesn't contain nested divs (usually), we can try finding the first </div> after page-header start.

            content = re.sub(r'(<div class="page-header">[\s\S]*?</div>)', r'\1\n<main id="main-content" tabindex="-1">', content, count=1)

            # Close main before <script>
            # Use rfind to find the LAST <script> tag to avoid targeting the one in <head>
            last_script_idx = content.rfind("<script>")
            if last_script_idx != -1:
                content = content[:last_script_idx] + '</main>\n  ' + content[last_script_idx:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Migrated {filepath}")

def main():
    # Find all HTML files in hubs/ and topic/
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)

    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()
