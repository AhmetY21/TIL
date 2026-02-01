import os
import datetime
import glob
import google.generativeai as genai
from slugify import slugify
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
TOPIC_BASE_DIR = "topic/nlp"

def get_existing_topics():
    """Scans the directory for existing .md files and extracts topic names."""
    topics = []
    files = glob.glob(os.path.join(TOPIC_BASE_DIR, "**/*.md"), recursive=True)
    for f in files:
        # Simple extraction: filename without .md (slugified)
        topic_slug = os.path.basename(f).replace(".md", "")
        topics.append(topic_slug)
    return topics

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
    existing_topics = get_existing_topics()
    topics_context = ", ".join(existing_topics) if existing_topics else "None yet"
    
    prompt = f"""
Teach me one topic from natural language processing. 
IMPORTANT: Avoid these topics already covered: {topics_context}. 
Pick a unique, interesting, or more advanced sub-topic each time.

Please strictly follow this format:
Topic: <Topic Name>

1- Provide formal definition, what is it and how can we use it?
2- Provide an application scenario
3- Provide a method to apply in python (if possible)
4- Provide a follow up question about that topic
5- Schedule a chatgpt chat to send notification (Simulated)

Ensure the response is formatted in valid Markdown.
"""

    if not API_KEY:
        print("WARNING: GEMINI_API_KEY not found. Using MOCK response for testing.")
        return """Topic: Word Embeddings (Mock)

1- Formal Definition:
Word embeddings are vector representations of words...

2- Application Scenario:
Used in recommendation systems...

3- Python Method:
```python
import gensim
```

4- Follow up:
How does Word2Vec differ from GloVe?

5- Schedule:
Standard notification...
"""
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content(prompt)
    content = response.text
    return content

def save_content(content, week_num, date_str, lesson_num):
    # Extract topic for filename
    lines = content.strip().split('\n')
    topic_line = next((line for line in lines if line.strip().startswith("Topic:")), "Topic: Unknown NLP Topic")
    topic_name = topic_line.replace("Topic:", "").strip()
    slug = slugify(topic_name)
    
    # Construct path
    # topic/nlp/week_WW/day_YYYY-MM-DD/lesson_N/
    week_dir = f"week_{week_num}"
    day_dir = f"day_{date_str}"
    lesson_dir = f"lesson_{lesson_num}"
    
    output_dir = os.path.join(TOPIC_BASE_DIR, week_dir, day_dir, lesson_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{slug}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(content)
        
    print(f"Generated: {filepath}")
    return filepath

if __name__ == "__main__":
    try:
        print("Starting content generation...")
        content = generate_content()
        week, date_val, lesson, _ = get_lesson_info()
        filepath = save_content(content, week, date_val, lesson)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
