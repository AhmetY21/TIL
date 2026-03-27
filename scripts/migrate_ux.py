#!/usr/bin/env python3
"""
Migrates existing HTML files to include UX improvements (Skip Link, Main Landmark).
"""
import glob
import re
import os

SKIP_LINK_CSS = """
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: #2563eb;
      color: white;
      padding: 8px;
      z-index: 100;
    }
    .skip-link:focus {
      top: 0;
    }
"""

SKIP_LINK_HTML = '<a href="#main-content" class="skip-link">Skip to content</a>'

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "skip-link" in content:
        # Check if we need to upgrade to tabindex="-1"
        if 'id="main-content"' in content and 'tabindex="-1"' not in content:
             content = content.replace('id="main-content"', 'id="main-content" tabindex="-1"')
             with open(filepath, "w", encoding="utf-8") as f:
                 f.write(content)
             print(f"Upgraded {filepath} with tabindex")
        else:
             print(f"Skipping {filepath} (already migrated)")
        return

    # 1. Inject CSS
    # Add before </style>
    if "</style>" in content:
        content = content.replace("</style>", f"{SKIP_LINK_CSS}\n  </style>")

    # 2. Inject Skip Link
    # Add after <body> (or <body ...>)
    # Regex to find body tag ending
    content = re.sub(r"(<body[^>]*>)", f"\\1\n  {SKIP_LINK_HTML}", content, count=1)

    # 3. Wrap Content in <main>
    if "hubs/" in filepath:
        # Hub pages: <div class="container"> -> <main class="container" id="main-content" tabindex="-1">
        content = content.replace('<div class="container">', '<main class="container" id="main-content" tabindex="-1">')
        # Find closing div before script
        content = re.sub(r"</div>(\s*<script>)", r"</main>\1", content, count=1)

    elif "topic/" in filepath:
        # Lesson pages: Wrap <h1 ...> ... <script>
        # Replace <h1 with <main id="main-content" tabindex="-1">\n<h1
        content = re.sub(r"(<h1)", f'<main id="main-content" tabindex="-1">\n\\1', content, count=1)

        # Replace <script> with </main>\n<script> (at the end)
        script_start_marker = "  <script>\n    const btn ="
        if script_start_marker in content:
            content = content.replace(script_start_marker, f"</main>\n{script_start_marker}")
        else:
             # Fallback: look for <script> near </body>
             idx = content.rfind("<script>")
             if idx != -1:
                 content = content[:idx] + "</main>\n" + content[idx:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Migrated {filepath}")

def main():
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)
    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()
