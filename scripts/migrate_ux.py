#!/usr/bin/env python3
"""
Adds 'Skip to content' link, CSS, and content wrapper to existing HTML files.
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
      padding: 8px;
      z-index: 100;
      transition: top 0.2s;
      text-decoration: none;
      font-weight: bold;
      border-radius: 0 0 4px 0;
    }
    .skip-link:focus {
      top: 0;
    }
"""

SKIP_LINK_HTML = '<a href="#main-content" class="skip-link">Skip to content</a>'

CONTENT_WRAPPER_HTML = '<div id="main-content" tabindex="-1"></div>'

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    is_modified = False

    # 1. Inject CSS
    if ".skip-link {" not in content:
        if "</style>" in content:
            content = content.replace("</style>", f"{CSS_BLOCK}\n  </style>")
            print(f"[{filepath}] Injected CSS")
            is_modified = True

    # 2. Inject Skip Link Anchor
    if 'class="skip-link"' not in content:
        if "<body>" in content:
            content = content.replace("<body>", f"<body>\n{SKIP_LINK_HTML}")
            print(f"[{filepath}] Injected Skip Link Anchor")
            is_modified = True

    # 3. Inject Content Wrapper (id="main-content")
    # Avoid double injection
    if 'id="main-content"' in content:
        print(f"[{filepath}] Skipping content wrapper (already present)")
    else:
        # Determine if Lesson or Hub
        is_hub = "hubs/" in filepath or "index.html" in filepath

        wrapper_inserted = False

        if is_hub:
            # Hub Logic: Insert after back-link
            # Pattern: <a href="..." class="back-link">...</a>
            if 'class="back-link"' in content:
                match = re.search(r'(<a [^>]*class="back-link"[^>]*>.*?</a>)', content, re.DOTALL)
                if match:
                    full_tag = match.group(1)
                    replacement = f"{full_tag}\n{CONTENT_WRAPPER_HTML}"
                    content = content.replace(full_tag, replacement)
                    wrapper_inserted = True
                    print(f"[{filepath}] Injected Content Wrapper (Hub)")
        else:
            # Lesson Logic: Insert after .page-header
            # Pattern: <div class="page-header">...</div>
            match = re.search(r'(<div class="page-header">.*?</div>)', content, re.DOTALL)
            if match:
                full_block = match.group(1)
                replacement = f"{full_block}\n{CONTENT_WRAPPER_HTML}"
                content = content.replace(full_block, replacement)
                wrapper_inserted = True
                print(f"[{filepath}] Injected Content Wrapper (Lesson)")

        if wrapper_inserted:
            is_modified = True
        else:
            print(f"[{filepath}] WARNING: Could not find insertion point for content wrapper")

    if is_modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

def main():
    # Find all HTML files in hubs/ and topic/
    # Exclude index.html because we manually updated it
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)

    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()
