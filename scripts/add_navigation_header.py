#!/usr/bin/env python3
"""
Adds navigation header to existing lesson HTML files.
"""
import os
import glob
import re
import json

# CSS for the new header
# We explicitly set position: static to override any previous absolute positioning
HEADER_CSS = """
    .page-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e5e7eb;
    }
    .back-link {
      font-weight: 600;
      text-decoration: none;
    }
    .back-link:hover {
      text-decoration: underline;
    }

    .theme-toggle {
      position: static;
      background: none;
      border: none;
      cursor: pointer;
      font-size: 1.5rem;
      padding: 8px;
      border-radius: 50%;
      transition: background 0.2s;
    }
    .theme-toggle:hover {
      background: rgba(0,0,0,0.05);
    }
    .dark .theme-toggle:hover {
      background: rgba(255,255,255,0.1);
    }
    .dark .page-header { border-bottom-color: #334155; }
"""

def get_subject_name(slug: str) -> str:
    # Try to find curriculum file
    curriculum_files = glob.glob("curriculums/*.json")
    for f in curriculum_files:
        try:
            with open(f, "r", encoding="utf-8") as cf:
                data = json.load(cf)
                meta = data.get("meta", {})
                if meta.get("slug") == slug:
                    return meta.get("subject", slug.title())
        except Exception:
            continue
    return slug.title().replace("-", " ")

def migrate_file(filepath: str):
    # Only process files in topic/
    if "topic/" not in filepath:
        return

    # Extract slug: topic/{slug}/...
    parts = filepath.split(os.sep)
    try:
        # handle case where path might start with ./topic or similar
        normalized_parts = [p for p in parts if p not in ('.', '..')]
        topic_idx = normalized_parts.index("topic")
        slug = normalized_parts[topic_idx + 1]
    except (ValueError, IndexError):
        print(f"Skipping {filepath} (cannot extract slug)")
        return

    subject_name = get_subject_name(slug)
    # Assuming standard depth: topic/{slug}/week/day/lesson/file.html -> 5 levels up
    back_link_url = f"../../../../../hubs/{slug}-index.html"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already has header
    if '<div class="page-header">' in content:
        print(f"Skipping {filepath} (already has header)")
        return

    # 1. Inject CSS
    if "</style>" in content:
        content = content.replace("</style>", f"{HEADER_CSS}\n  </style>")
    else:
        print(f"Warning: No </style> tag in {filepath}")

    # 2. Prepare Header HTML
    header_html = f"""
  <div class="page-header">
    <a href="{back_link_url}" class="back-link">← Back to {subject_name}</a>
    <button id="theme-toggle" class="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>
  </div>
"""

    # 3. Remove existing button
    # It might be: <button id="theme-toggle" ...>🌙</button>
    # We use regex to be safe about attributes order
    content = re.sub(
        r'<button\s+id="theme-toggle"[^>]*>.*?</button>',
        "",
        content,
        flags=re.DOTALL
    )

    # 4. Inject Header after <body>
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n{header_html}")
    else:
        print(f"Warning: No <body> tag in {filepath}")

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
