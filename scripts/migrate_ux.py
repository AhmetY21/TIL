#!/usr/bin/env python3
"""
Adds skip link and wraps content in <main> for existing lesson HTML files.
"""
import os
import glob
import re

SKIP_LINK_CSS = """
    /* Skip Link */
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: #0f172a;
      color: white;
      padding: 8px 16px;
      z-index: 100;
      transition: top 0.2s;
      font-weight: 600;
      border-bottom-right-radius: 6px;
    }
    .skip-link:focus {
      top: 0;
    }
"""

SKIP_LINK_HTML = '<a href="#main-content" class="skip-link">Skip to content</a>'

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already migrated
    if 'class="skip-link"' in content:
        print(f"Skipping {filepath} (already migrated)")
        return

    # 1. Inject CSS
    if "</style>" in content:
        # Avoid duplicating CSS if it exists (though check above should catch it)
        if ".skip-link {" not in content:
            content = content.replace("</style>", f"{SKIP_LINK_CSS}\n  </style>")
    else:
        print(f"Warning: No </style> tag in {filepath}")

    # 2. Inject Skip Link
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n  {SKIP_LINK_HTML}")
    else:
        print(f"Warning: No <body> tag in {filepath}")

    # 3. Wrap content in <main>
    # Logic: Find <div class="page-header">...</div> and insert <main id="main-content"> after it.
    # Find last <script> and insert </main> before it.

    header_pattern = re.compile(r'(<div class="page-header">.*?</div>)', re.DOTALL)
    match = header_pattern.search(content)

    if match:
        # Insert <main> after the header div
        end_pos = match.end()
        content = content[:end_pos] + '\n  <main id="main-content">' + content[end_pos:]

        # Find the script tag that handles the theme toggle
        # It's usually the last script tag or near the end of body
        # We want to wrap the content before the scripts at the bottom

        # Find the last occurrence of <script>
        script_idx = content.rfind("<script>")
        if script_idx != -1:
            content = content[:script_idx] + '  </main>\n' + content[script_idx:]
        else:
             # If no script at bottom, inject before </body>
             if "</body>" in content:
                 content = content.replace("</body>", "  </main>\n</body>")
             else:
                 print(f"Warning: No </body> tag in {filepath}")
    else:
        print(f"Warning: No page-header found in {filepath}. Skipping main wrapper.")
        # Fallback: maybe wrap everything in body? better not to break things.

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Migrated {filepath}")

def main():
    files = glob.glob("topic/**/*.html", recursive=True)
    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()
