import os
import datetime
import glob
import json
import google.generativeai as genai
from slugify import slugify
from dotenv import load_dotenv
import markdown

load_dotenv()

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
TOPIC_BASE_DIR = "topic/nlp"
CURRICULUM_FILE = "curriculum.json"

def get_existing_topic_slugs():
    """Scans the directory for existing .md files and returns their slugs."""
    slugs = set()
    files = glob.glob(os.path.join(TOPIC_BASE_DIR, "**/*.md"), recursive=True)
    for f in files:
        # Filename without extension is the slug
        slug = os.path.basename(f).replace(".md", "")
        slugs.add(slug)
    return slugs

def get_next_topic():
    """Reads curriculum.json and returns the first topic that hasn't been generated."""
    if not os.path.exists(CURRICULUM_FILE):
        print(f"Error: {CURRICULUM_FILE} not found.")
        return None

    with open(CURRICULUM_FILE, 'r') as f:
        data = json.load(f)
        curriculum_topics = data.get("topics", [])

    existing_slugs = get_existing_topic_slugs()

    for topic in curriculum_topics:
        slug = slugify(topic)
        if slug not in existing_slugs:
            print(f"Next topic selected: {topic}")
            return topic
    
    print("All topics in the curriculum have been covered!")
    return None

def get_lesson_info():
    now = datetime.datetime.now()
    # ISO Week number
    week_num = now.isocalendar()[1]
    # Date string for folder
    date_str = now.strftime("%Y-%m-%d")
    # Lesson number (every 4 hours: 0, 4, 8, 12, 16, 20) -> 1, 2, 3, 4, 5, 6
    lesson_num = (now.hour // 4) + 1
    return week_num, date_str, lesson_num, now

def generate_content():
    topic_name = get_next_topic()
    if not topic_name:
        return None, None

    prompt = f"""
Teach me about the following topic in Natural Language Processing: "{topic_name}".

Please strictly follow this format:
Topic: {topic_name}

1- Provide formal definition, what is it and how can we use it?
2- Provide an application scenario
3- Provide a method to apply in python (if possible)
4- Provide a follow up question about that topic
5- Schedule a chatgpt chat to send notification (Simulated)

Ensure the response is formatted in valid Markdown.
"""

    if not API_KEY:
        print("WARNING: GEMINI_API_KEY not found. Using MOCK response for testing.")
        return f"""Topic: {topic_name} (Mock)

1- Formal Definition:
Mock definition for {topic_name}...

2- Application Scenario:
Mock scenario...

3- Python Method:
```python
# Mock code
print("Hello NLP")
```

4- Follow up:
Mock question?

5- Schedule:
Standard notification...
""", topic_name
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content(prompt)
    content = response.text
    return content, topic_name

def convert_md_to_html(content, title):
    """Converts markdown content to a full HTML page."""
    # Strip leading/trailing code blocks if Gemini added them
    clean_content = content.replace("```markdown", "").replace("```", "").strip()
    html_body = markdown.markdown(clean_content, extensions=['fenced_code', 'codehilite'])
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #eee; margin: 0; padding-left: 15px; color: #666; }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""
    return html_content

def save_content(content, topic_name, week_num, date_str, lesson_num):
    if not content:
        return None
        
    # Robust topic name (use provided one if extraction fails)
    if not topic_name:
        lines = content.strip().split('\n')
        topic_line = next((line for line in lines if "Topic:" in line), None)
        if topic_line:
            topic_name = topic_line.split("Topic:")[-1].replace("(Mock)", "").replace("*", "").replace("#", "").strip()
        else:
            topic_name = "Unknown NLP Topic"

    slug = slugify(topic_name)
    
    # Construct path
    # topic/nlp/week_WW/day_YYYY-MM-DD/lesson_N/
    week_dir = f"week_{week_num}"
    day_dir = f"day_{date_str}"
    lesson_dir = f"lesson_{lesson_num}"
    
    output_dir = os.path.join(TOPIC_BASE_DIR, week_dir, day_dir, lesson_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Strip potential markdown wrapping from Gemini
    clean_content = content.replace("```markdown", "").replace("```", "").strip()
    
    # Save Markdown
    filename_md = f"{slug}.md"
    filepath_md = os.path.join(output_dir, filename_md)
    with open(filepath_md, "w") as f:
        f.write(clean_content)
        
    # Save HTML
    filename_html = f"{slug}.html"
    filepath_html = os.path.join(output_dir, filename_html)
    html_content = convert_md_to_html(clean_content, topic_name)
    with open(filepath_html, "w") as f:
        f.write(html_content)
        
    print(f"Generated Markdown: {filepath_md}")
    print(f"Generated HTML: {filepath_html}")
    return filepath_md

def update_index_page():
    """Scans the topic directory and rebuilds the index.html page."""
    lessons = []
    files = glob.glob(os.path.join(TOPIC_BASE_DIR, "**/*.html"), recursive=True)
    
    for f in files:
        if os.path.basename(f) == "index.html":
            continue
        
        # Get relative path for the link
        rel_path = os.path.relpath(f, ".")
        
        # Extract metadata from path: topic/nlp/week_6/day_2026-02-02/lesson_6/filename.html
        parts = rel_path.split(os.sep)
        if len(parts) >= 5:
            week = parts[2].replace("week_", "Week ")
            day = parts[3].replace("day_", "")
            lesson_name = os.path.basename(f).replace(".html", "").replace("-", " ").title()
            
            lessons.append({
                "week": week,
                "day": day,
                "name": lesson_name,
                "html_path": rel_path,
                "md_path": rel_path.replace(".html", ".md")
            })

    # Sort lessons: newest first
    lessons.sort(key=lambda x: (x["week"], x["day"]), reverse=True)

    # Generate HTML content
    lessons_html = ""
    current_week = None
    
    for lesson in lessons:
        if lesson["week"] != current_week:
            if current_week is not None:
                lessons_html += "</div>"
            current_week = lesson["week"]
            lessons_html += f"<h2 class='week-title'>{current_week}</h2><div class='week-container'>"
            
        lessons_html += f"""
        <div class="lesson-card">
            <div class="lesson-date">{lesson["day"]}</div>
            <div class="lesson-name">{lesson["name"]}</div>
            <div class="lesson-links">
                <a href="{lesson["html_path"]}" class="btn btn-primary">View Lesson (HTML)</a>
                <a href="{lesson["md_path"]}" class="btn btn-secondary">Source (Markdown)</a>
            </div>
        </div>
        """
    
    if current_week is not None:
        lessons_html += "</div>"

    index_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Learning Hub</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --secondary: #64748b;
            --bg: #f8fafc;
            --text: #1e293b;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 50px; }}
        h1 {{ font-size: 2.5rem; color: #0f172a; margin-bottom: 10px; }}
        .subtitle {{ color: var(--secondary); font-size: 1.1rem; }}
        
        .week-title {{ 
            margin-top: 40px; 
            padding-bottom: 10px; 
            border-bottom: 2px solid #e2e8f0;
            color: #334155;
        }}
        .week-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .lesson-card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .lesson-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }}
        
        .lesson-date {{ font-size: 0.875rem; color: var(--secondary); font-weight: 500; }}
        .lesson-name {{ font-size: 1.25rem; font-weight: 700; margin: 8px 0 20px; color: #1e293b; }}
        
        .lesson-links {{ display: flex; gap: 10px; }}
        .btn {{
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 600;
            transition: background 0.2s;
            flex: 1;
            text-align: center;
        }}
        .btn-primary {{ background: var(--primary); color: white; }}
        .btn-primary:hover {{ background: var(--primary-hover); }}
        .btn-secondary {{ background: #f1f5f9; color: var(--secondary); }}
        .btn-secondary:hover {{ background: #e2e8f0; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NLP Learning Hub</h1>
            <p class="subtitle">A curriculum-based journey into Natural Language Processing</p>
        </header>
        
        {lessons_html if lessons else "<p style='text-align:center;'>No lessons generated yet.</p>"}
    </div>
</body>
</html>
"""
    with open("index.html", "w") as f:
        f.write(index_content)
    print("Updated index.html")

if __name__ == "__main__":
    try:
        print("Starting content generation...")
        content, topic = generate_content()
        if content:
            week, date_val, lesson, _ = get_lesson_info()
            filepath = save_content(content, topic, week, date_val, lesson)
            print("Generating/Updating Index Hub...")
            update_index_page()
            print("Done.")
        else:
            print("No new content, checking index updates...")
            update_index_page()
            print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
