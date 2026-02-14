"""
Generic Daily Lesson Generator (Gemini) -> Markdown + HTML + Index

Fully curriculum-driven: reads a `meta` block from the curriculum JSON
to determine subject, output paths, prompts, and index pages.

Usage:
    python generate_lesson.py --curriculum curriculum_nlp.json
    python generate_lesson.py --curriculum curriculum_causal.json
"""

import os
import re
import glob
import json
import argparse
import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import google.generativeai as genai
from slugify import slugify
from dotenv import load_dotenv
import markdown

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini model
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Lesson cadence (every N hours)
LESSON_EVERY_HOURS = 4  # 4 => 6 lessons/day

# Markdown extensions for HTML conversion
MD_EXTENSIONS = [
    "fenced_code",
    "tables",
    "toc",
    "codehilite",
    "admonition",
]


@dataclass(frozen=True)
class CurriculumMeta:
    """Metadata extracted from the curriculum JSON's 'meta' block."""
    subject: str
    slug: str
    subtitle: str
    prompt_domain: str
    curriculum_file: str

    @property
    def topic_base_dir(self) -> str:
        return f"topic/{self.slug}"

    @property
    def state_file(self) -> str:
        return os.path.join(self.topic_base_dir, "state.json")

    @property
    def index_file(self) -> str:
        return f"hubs/{self.slug}-index.html"


def load_curriculum_meta(curriculum_file: str) -> CurriculumMeta:
    """Load and validate the meta block from a curriculum JSON file."""
    if not os.path.exists(curriculum_file):
        raise FileNotFoundError(f"Curriculum file not found: {curriculum_file}")

    with open(curriculum_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta")
    if not meta:
        raise ValueError(
            f"Curriculum file '{curriculum_file}' is missing a 'meta' block. "
            f"Expected: {{ \"meta\": {{ \"subject\": ..., \"slug\": ..., \"subtitle\": ..., \"prompt_domain\": ... }} }}"
        )

    required_fields = ["subject", "slug", "subtitle", "prompt_domain"]
    for field in required_fields:
        if field not in meta:
            raise ValueError(f"meta block in '{curriculum_file}' is missing required field: '{field}'")

    return CurriculumMeta(
        subject=meta["subject"],
        slug=meta["slug"],
        subtitle=meta["subtitle"],
        prompt_domain=meta["prompt_domain"],
        curriculum_file=curriculum_file,
    )


# -----------------------------
# Helpers
# -----------------------------
def unwrap_outer_markdown_fence(text: str) -> str:
    """
    If the *entire* response is wrapped like:
      ```markdown
      ...
      ```
    unwrap it. Preserve internal code fences.
    """
    t = text.strip()
    m = re.match(r"^```(?:markdown)?\s*\n([\s\S]*?)\n```$", t)
    return m.group(1).strip() if m else t


def safe_write(path: str, content: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class LessonInfo:
    week_num: int
    date_str: str  # YYYY-MM-DD
    lesson_num: int
    now: datetime.datetime


def get_lesson_info(now: Optional[datetime.datetime] = None) -> LessonInfo:
    now = now or datetime.datetime.now()
    week_num = now.isocalendar()[1]
    date_str = now.strftime("%Y-%m-%d")
    lesson_num = (now.hour // LESSON_EVERY_HOURS) + 1
    return LessonInfo(week_num=week_num, date_str=date_str, lesson_num=lesson_num, now=now)


def get_existing_topic_slugs(meta: CurriculumMeta) -> set:
    """Scans topic_base_dir for existing .md files and returns filename slugs."""
    slugs = set()
    files = glob.glob(os.path.join(meta.topic_base_dir, "**/*.md"), recursive=True)
    for f in files:
        slug = os.path.basename(f).replace(".md", "")
        slugs.add(slug)
    return slugs


def read_curriculum_topics(meta: CurriculumMeta) -> List[str]:
    if not os.path.exists(meta.curriculum_file):
        raise FileNotFoundError(f"{meta.curriculum_file} not found.")
    data = safe_read_json(meta.curriculum_file)
    raw_topics = data.get("topics", [])

    if not isinstance(raw_topics, list):
        raise ValueError(f"{meta.curriculum_file} must contain: {{ 'topics': [...] }}")

    # Extract names if they are dicts, or use strings directly
    topics = []
    for t in raw_topics:
        if isinstance(t, str):
            topics.append(t)
        elif isinstance(t, dict) and "name" in t:
            topics.append(t["name"])

    if not topics:
        raise ValueError(f"No valid topics found in {meta.curriculum_file}")

    return topics


def load_state(meta: CurriculumMeta) -> Dict[str, Any]:
    if os.path.exists(meta.state_file):
        try:
            return safe_read_json(meta.state_file)
        except Exception:
            return {}
    return {}


def save_state(meta: CurriculumMeta, state: Dict[str, Any]) -> None:
    safe_write_json(meta.state_file, state)


def get_next_topic(meta: CurriculumMeta) -> Optional[str]:
    """
    Chooses the next topic. Priority:
    1) state file next_index (deterministic)
    2) filesystem slug scan fallback
    """
    topics = read_curriculum_topics(meta)

    # 1) State-driven progression
    state = load_state(meta)
    next_index = state.get("next_index")
    if isinstance(next_index, int) and 0 <= next_index < len(topics):
        topic = topics[next_index]
        print(f"[{meta.slug}] Next topic (state): [{next_index}] {topic}")
        return topic

    # 2) Fallback: slug-scan
    existing_slugs = get_existing_topic_slugs(meta)
    for idx, topic in enumerate(topics):
        slug = slugify(topic)
        if slug not in existing_slugs:
            save_state(meta, {"next_index": idx, "total_topics": len(topics)})
            print(f"[{meta.slug}] Next topic (scan->state): [{idx}] {topic}")
            return topic

    print(f"[{meta.slug}] All topics in the curriculum have been covered!")
    return None


def advance_state_after_success(meta: CurriculumMeta, topic_name: str) -> None:
    """Increment next_index if state exists; otherwise create it from the topic match."""
    topics = read_curriculum_topics(meta)
    state = load_state(meta)

    if "next_index" in state and isinstance(state["next_index"], int):
        state["next_index"] = min(state["next_index"] + 1, len(topics))
        state["last_topic"] = topic_name
        save_state(meta, state)
        return

    try:
        idx = topics.index(topic_name)
        save_state(meta, {"next_index": min(idx + 1, len(topics)), "last_topic": topic_name, "total_topics": len(topics)})
    except ValueError:
        pass


def build_prompt(meta: CurriculumMeta, topic_name: str) -> str:
    return f"""Teach me about the following topic in {meta.prompt_domain}: "{topic_name}".

Please strictly follow this format and headings:

# Topic: {topic_name}

## 1) Formal definition (what is it, and how can we use it?)
## 2) Application scenario
## 3) Python method (if possible)
- Put code inside a fenced code block using ```python
## 4) Follow-up question

Ensure the response is formatted in valid Markdown.
"""


def generate_content(meta: CurriculumMeta) -> Tuple[Optional[str], Optional[str]]:
    topic_name = get_next_topic(meta)
    if not topic_name:
        return None, None

    prompt = build_prompt(meta, topic_name)

    if not API_KEY:
        print("WARNING: GEMINI_API_KEY not found. Using MOCK response for testing.")
        mock = (
            f"# Topic: {topic_name} (Mock)\n\n"
            "## 1) Formal definition (what is it, and how can we use it?)\n"
            f"Mock definition for **{topic_name}**...\n\n"
            "## 2) Application scenario\n"
            "Mock scenario...\n\n"
            "## 3) Python method (if possible)\n"
            "```python\n"
            f'print("Hello {meta.prompt_domain}")\n'
            "```\n\n"
            "## 4) Follow-up question\n"
            f"What is one limitation of {topic_name} in real-world {meta.prompt_domain} pipelines?\n"
        )
        return mock, topic_name

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(prompt)
    content = getattr(response, "text", None)
    if not content:
        print("ERROR: Model returned empty content.")
        return None, topic_name

    return content, topic_name


def add_front_matter(md: str, *, title: str, date_str: str, week_num: int, lesson_num: int, slug: str) -> str:
    fm = (
        "---\n"
        f'title: "{title}"\n'
        f'date: "{date_str}"\n'
        f"week: {week_num}\n"
        f"lesson: {lesson_num}\n"
        f'slug: "{slug}"\n'
        "---\n\n"
    )
    if md.lstrip().startswith("---\n"):
        return md
    return fm + md


def convert_md_to_html(md_text: str, title: str, meta: Optional[CurriculumMeta] = None) -> str:
    html_body = markdown.markdown(md_text, extensions=MD_EXTENSIONS)

    header_html = ""
    if meta:
        # Assuming fixed directory structure: topic/{slug}/week_N/day_Y/lesson_N/slug.html
        # Relative path to root is ../../../../../ (5 levels up)
        # Hub is at hubs/{slug}-index.html
        hub_link = f"../../../../../hubs/{meta.slug}-index.html"
        header_html = f"""
  <div class="page-header">
    <a href="{hub_link}" class="back-link">← Back to {meta.subject}</a>
    <button id="theme-toggle" class="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>
  </div>
"""
    else:
        # Fallback if no meta provided
        header_html = '<button id="theme-toggle" class="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <script>
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
      document.documentElement.classList.add('dark');
    }} else {{
      document.documentElement.classList.remove('dark');
    }}
  </script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      max-width: 900px;
      margin: 0 auto;
      padding: 24px;
      color: #111827;
      background: #ffffff;
    }}
    h1, h2, h3 {{ color: #111827; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background-color: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    pre {{ background-color: #0b1020; color: #e5e7eb; padding: 16px; border-radius: 10px; overflow-x: auto; }}
    pre code {{ background: transparent; padding: 0; }}
    blockquote {{ border-left: 4px solid #e5e7eb; margin: 0; padding-left: 16px; color: #4b5563; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
    hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }}

    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .back-link {{
      font-weight: 600;
      text-decoration: none;
    }}
    .back-link:hover {{
      text-decoration: underline;
    }}

    .theme-toggle {{
      background: none;
      border: none;
      cursor: pointer;
      font-size: 1.5rem;
      padding: 8px;
      border-radius: 50%;
      transition: background 0.2s;
    }}
    .theme-toggle:hover {{
      background: rgba(0,0,0,0.05);
    }}
    .dark .theme-toggle:hover {{
      background: rgba(255,255,255,0.1);
    }}

    .dark body {{
      background: #0f172a;
      color: #e2e8f0;
    }}
    .dark h1, .dark h2, .dark h3 {{ color: #f1f5f9; }}
    .dark a {{ color: #60a5fa; }}
    .dark .page-header {{ border-bottom-color: #334155; }}
    .dark code {{ background-color: #1e293b; color: #e2e8f0; }}
    .dark pre {{
      border: 1px solid #334155;
    }}
    .dark blockquote {{
      border-left-color: #334155;
      color: #94a3b8;
    }}
    .dark th, .dark td {{ border-color: #334155; }}
    .dark th {{ background: #1e293b; }}
    .dark hr {{ border-top-color: #334155; }}

    /* Copy Button */
    pre {{ position: relative; }}
    .copy-button {{
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 4px 8px;
      font-size: 0.8rem;
      color: #94a3b8;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s, background 0.2s;
    }}
    pre:hover .copy-button, .copy-button:focus {{
      opacity: 1;
    }}
    .copy-button:hover {{
      background: rgba(255, 255, 255, 0.2);
      color: #e2e8f0;
    }}
  </style>
</head>
<body>
{header_html}
{html_body}
  <script>
    const btn = document.getElementById('theme-toggle');
    const html = document.documentElement;
    
    function updateIcon() {{
      btn.textContent = html.classList.contains('dark') ? '☀️' : '🌙';
    }}
    
    // Set initial icon
    updateIcon();

    btn.addEventListener('click', () => {{
      if (html.classList.contains('dark')) {{
        html.classList.remove('dark');
        localStorage.theme = 'light';
      }} else {{
        html.classList.add('dark');
        localStorage.theme = 'dark';
      }}
      updateIcon();
    }});

    // Copy Code Functionality
    document.querySelectorAll('pre').forEach(pre => {{
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');

      pre.appendChild(button);

      button.addEventListener('click', async () => {{
        const code = pre.querySelector('code');
        if (!code) return;

        try {{
          await navigator.clipboard.writeText(code.innerText);
          button.textContent = 'Copied!';
          setTimeout(() => {{
            button.textContent = 'Copy';
          }}, 2000);
        }} catch (err) {{
          console.error('Failed to copy:', err);
          button.textContent = 'Error';
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def extract_topic_title_fallback(md: str) -> Optional[str]:
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    for ln in lines[:30]:
        m1 = re.match(r"^#\s*Topic:\s*(.+)$", ln, re.IGNORECASE)
        if m1:
            return m1.group(1).replace("(Mock)", "").strip()
        m2 = re.match(r"^Topic:\s*(.+)$", ln, re.IGNORECASE)
        if m2:
            return m2.group(1).replace("(Mock)", "").strip()
    return None


def save_content(meta: CurriculumMeta, content: str, topic_name: Optional[str], info: LessonInfo) -> Optional[str]:
    if not content:
        return None

    raw_md = unwrap_outer_markdown_fence(content)

    title = (topic_name or extract_topic_title_fallback(raw_md) or f"Unknown {meta.subject} Topic").strip()
    title = title.replace("(Mock)", "").strip()

    slug = slugify(title)

    week_dir = f"week_{info.week_num}"
    day_dir = f"day_{info.date_str}"
    lesson_dir = f"lesson_{info.lesson_num}"
    output_dir = os.path.join(meta.topic_base_dir, week_dir, day_dir, lesson_dir)
    os.makedirs(output_dir, exist_ok=True)

    final_md = add_front_matter(
        raw_md,
        title=title,
        date_str=info.date_str,
        week_num=info.week_num,
        lesson_num=info.lesson_num,
        slug=slug,
    )

    filename_md = f"{slug}.md"
    filepath_md = os.path.join(output_dir, filename_md)
    safe_write(filepath_md, final_md)

    filename_html = f"{slug}.html"
    filepath_html = os.path.join(output_dir, filename_html)
    html_content = convert_md_to_html(raw_md, title, meta)
    safe_write(filepath_html, html_content)

    print(f"[{meta.slug}] Generated Markdown: {filepath_md}")
    print(f"[{meta.slug}] Generated HTML: {filepath_html}")
    return filepath_md


def parse_lesson_path(rel_path: str, meta: CurriculumMeta) -> Optional[Dict[str, Any]]:
    """
    Expected: topic/{slug}/week_6/day_2026-02-02/lesson_3/some-topic.html
    """
    parts = rel_path.split(os.sep)
    if len(parts) < 6:
        return None
    if parts[0] != "topic" or parts[1] != meta.slug:
        return None

    m_week = re.match(r"^week_(\d+)$", parts[2])
    m_day = re.match(r"^day_(\d{4}-\d{2}-\d{2})$", parts[3])
    m_lsn = re.match(r"^lesson_(\d+)$", parts[4])
    if not (m_week and m_day and m_lsn):
        return None

    week_num = int(m_week.group(1))
    day_str = m_day.group(1)
    lesson_num = int(m_lsn.group(1))

    name = os.path.basename(parts[5]).replace(".html", "").replace("-", " ").title()

    return {
        "week_num": week_num,
        "day_str": day_str,
        "lesson_num": lesson_num,
        "name": name,
    }


def update_index_page(meta: CurriculumMeta) -> None:
    lessons: List[Dict[str, Any]] = []
    files = glob.glob(os.path.join(meta.topic_base_dir, "**/*.html"), recursive=True)

    for abs_path in files:
        if os.path.basename(abs_path) == "index.html":
            continue

        rel_path = os.path.relpath(abs_path, ".")
        lesson_meta = parse_lesson_path(rel_path, meta)
        if not lesson_meta:
            continue

        lessons.append(
            {
                **lesson_meta,
                "html_path": f"../{rel_path}",
                "md_path": f"../{rel_path.replace('.html', '.md')}",
            }
        )

    # Newest first
    lessons.sort(key=lambda x: (x["day_str"], x["lesson_num"], x["week_num"]), reverse=True)

    lessons_html = ""
    current_week_label = None

    for lesson in lessons:
        week_label = f"Week {lesson['week_num']}"
        if week_label != current_week_label:
            if current_week_label is not None:
                lessons_html += "</div>"
            current_week_label = week_label
            lessons_html += f"<h2 class='week-title'>{week_label}</h2><div class='week-container'>"

        lessons_html += f"""
        <div class="lesson-card">
            <div class="lesson-date">{lesson["day_str"]} · Lesson {lesson["lesson_num"]}</div>
            <div class="lesson-name">{lesson["name"]}</div>
            <div class="lesson-links">
                <a href="{lesson["html_path"]}" class="btn btn-primary">View Lesson (HTML)</a>
            </div>
        </div>
        """

    if current_week_label is not None:
        lessons_html += "</div>"

    index_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta.subject} Learning Hub</title>
  <script>
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
      document.documentElement.classList.add('dark');
    }} else {{
      document.documentElement.classList.remove('dark');
    }}
  </script>
  <style>
    :root {{
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --secondary: #64748b;
      --bg: #f8fafc;
      --text: #1e293b;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      margin: 0;
      padding: 40px 20px;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 50px; }}
    h1 {{ font-size: 2.5rem; color: #0f172a; margin-bottom: 10px; }}
    .subtitle {{ color: var(--secondary); font-size: 1.1rem; }}
    .back-link {{ display: inline-block; margin-bottom: 20px; color: var(--primary); text-decoration: none; font-weight: 600; }}
    .back-link:hover {{ text-decoration: underline; }}

    .week-title {{
      margin-top: 40px;
      padding-bottom: 10px;
      border-bottom: 2px solid #e2e8f0;
      color: #334155;
    }}
    .week-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }}

    .lesson-card {{
      background: white;
      padding: 22px;
      border-radius: 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      transition: transform 0.18s, box-shadow 0.18s;
    }}
    .lesson-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 10px 18px -8px rgba(0,0,0,0.18);
    }}

    .lesson-date {{ font-size: 0.9rem; color: var(--secondary); font-weight: 600; }}
    .lesson-name {{ font-size: 1.2rem; font-weight: 800; margin: 10px 0 18px; color: #0f172a; }}

    .lesson-links {{ display: flex; gap: 10px; }}
    .btn {{
      padding: 10px 14px;
      border-radius: 10px;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 700;
      transition: background 0.18s;
      flex: 1;
      text-align: center;
    }}
    .btn-primary {{ background: var(--primary); color: white; }}
    .btn-primary:hover {{ background: var(--primary-hover); }}

    .theme-toggle {{
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
    }}
    .theme-toggle:hover {{
      background: rgba(0,0,0,0.05);
    }}
    .dark .theme-toggle:hover {{
      background: rgba(255,255,255,0.1);
    }}

    .dark {{
      --primary: #60a5fa;
      --primary-hover: #93c5fd;
      --bg: #0f172a;
      --text: #e2e8f0;
      --secondary: #94a3b8;
    }}
    .dark .lesson-card {{
      background: #1e293b;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }}
    .dark h1, .dark .week-title, .dark .lesson-name {{
      color: #f1f5f9;
    }}
    .dark .week-title {{
      border-bottom-color: #334155;
    }}
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>
  <div class="container">
    <a href="../index.html" class="back-link">← All Learning Hubs</a>
    <header>
      <h1>{meta.subject} Learning Hub</h1>
      <p class="subtitle">{meta.subtitle}</p>
    </header>

    {lessons_html if lessons else "<p style='text-align:center;'>No lessons generated yet.</p>"}
  </div>
  <script>
    const btn = document.getElementById('theme-toggle');
    const html = document.documentElement;
    
    function updateIcon() {{
      btn.textContent = html.classList.contains('dark') ? '☀️' : '🌙';
    }}
    
    // Set initial icon
    updateIcon();

    btn.addEventListener('click', () => {{
      if (html.classList.contains('dark')) {{
        html.classList.remove('dark');
        localStorage.theme = 'light';
      }} else {{
        html.classList.add('dark');
        localStorage.theme = 'dark';
      }}
      updateIcon();
    }});
  </script>
</body>
</html>
"""
    safe_write(meta.index_file, index_content)
    print(f"[{meta.slug}] Updated {meta.index_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic curriculum-based lesson generator")
    parser.add_argument(
        "--curriculum",
        required=True,
        help="Path to the curriculum JSON file (e.g. curriculum_nlp.json)",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        meta = load_curriculum_meta(args.curriculum)

        print(f"=== [{meta.subject}] Starting content generation ===")
        content, topic = generate_content(meta)

        if content:
            info = get_lesson_info()
            filepath = save_content(meta, content, topic, info)
            if filepath:
                advance_state_after_success(meta, topic_name=topic or "")
            print(f"[{meta.slug}] Generating/Updating Index Hub...")
            update_index_page(meta)
            print(f"[{meta.slug}] Done.")
        else:
            print(f"[{meta.slug}] No new content, checking index updates...")
            update_index_page(meta)
            print(f"[{meta.slug}] Done.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
