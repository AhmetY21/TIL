#!/usr/bin/env python3
"""
Adds Dark Mode toggle button, CSS, and JS to existing HTML files.
"""
import os
import glob
import re

CSS_BLOCK = """
    .theme-toggle {
      position: absolute;
      top: 20px;
      right: 20px;
      background: none;
      border: none;
      font-size: 1.5rem;
      cursor: pointer;
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

    .dark body {
      background: #0f172a;
      color: #e2e8f0;
    }
    .dark h1, .dark h2, .dark h3 { color: #f1f5f9; }
    .dark a { color: #60a5fa; }
    .dark code { background-color: #1e293b; color: #e2e8f0; }
    .dark pre {
      border: 1px solid #334155;
    }
    .dark blockquote {
      border-left-color: #334155;
      color: #94a3b8;
    }
    .dark th, .dark td { border-color: #334155; }
    .dark th { background: #1e293b; }
    .dark hr { border-top-color: #334155; }
"""

# Hub pages have slightly different structure/CSS
HUB_CSS_BLOCK = """
    .theme-toggle {
      position: absolute;
      top: 20px;
      right: 20px;
      background: none;
      border: none;
      font-size: 1.5rem;
      cursor: pointer;
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

    .dark {
      --primary: #60a5fa;
      --primary-hover: #93c5fd;
      --bg: #0f172a;
      --text: #e2e8f0;
      --secondary: #94a3b8;
    }
    .dark .lesson-card {
      background: #1e293b;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .dark h1, .dark .week-title, .dark .lesson-name {
      color: #f1f5f9;
    }
    .dark .week-title {
      border-bottom-color: #334155;
    }
"""

INIT_SCRIPT = """
  <script>
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  </script>
"""

BUTTON_HTML = """  <button id="theme-toggle" class="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>"""

TOGGLE_SCRIPT = """
  <script>
    const btn = document.getElementById('theme-toggle');
    const html = document.documentElement;
    
    function updateIcon() {
      btn.textContent = html.classList.contains('dark') ? '☀️' : '🌙';
    }
    
    // Set initial icon
    updateIcon();

    btn.addEventListener('click', () => {
      if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.theme = 'light';
      } else {
        html.classList.add('dark');
        localStorage.theme = 'dark';
      }
      updateIcon();
    });
  </script>
"""

def migrate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already migrated
    if 'id="theme-toggle"' in content:
        print(f"Skipping {filepath} (already migrated)")
        return

    # Determine if this is a hub page or lesson page
    # Hub pages have ".week-title" or "lesson-card" usually, or are in hubs/
    is_hub = "hubs/" in filepath or "index.html" in filepath
    css_to_use = HUB_CSS_BLOCK if is_hub else CSS_BLOCK

    # Remove old media query if present
    content = re.sub(r"@media \(prefers-color-scheme: dark\) \{[\s\S]*?\}\s*</style>", "</style>", content)

    # Inject Init Script before <style>
    if "<style>" in content:
        content = content.replace("<style>", f"{INIT_SCRIPT}\n  <style>")
    elif "</head>" in content:
        content = content.replace("</head>", f"{INIT_SCRIPT}\n</head>")

    # Inject CSS before </style>
    if "</style>" in content:
        content = content.replace("</style>", f"{css_to_use}\n  </style>")

    # Inject Button after <body>
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n{BUTTON_HTML}")

    # Inject Toggle Script before </body>
    if "</body>" in content:
        content = content.replace("</body>", f"{TOGGLE_SCRIPT}\n</body>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Migrated {filepath}")

def main():
    # Find all HTML files in hubs/ and topic/
    files = glob.glob("hubs/**/*.html", recursive=True) + glob.glob("topic/**/*.html", recursive=True)
    
    # Also index.html if needed (though already done manually)
    if os.path.exists("index.html"):
        files.append("index.html")
    
    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"Error migrating {f}: {e}")

if __name__ == "__main__":
    main()
