import os
import glob
import re
from slugify import slugify
from generate_lesson import convert_md_to_html, update_index_page, TOPIC_BASE_DIR

def extract_topic(content):
    """Robustly extracts the topic from markdown content."""
    # Look for ## Topic: or Topic: or **Topic:**
    match = re.search(r'(?:##\s*|Topic:\s*|\*\*Topic:\s*|#\s*)(?:Topic:\s*)?([^\n\*#]+)', content, re.IGNORECASE)
    if match:
        return match.group(1).replace("(Mock)", "").strip()
    return None

def migrate():
    print("Starting global migration...")
    md_files = glob.glob(os.path.join(TOPIC_BASE_DIR, "**/*.md"), recursive=True)

    for md_path in md_files:
        print(f"Processing: {md_path}")
        with open(md_path, 'r') as f:
            content = f.read()

        # Clean content (strip markdown wrapping if present)
        clean_content = content.replace("```markdown", "").replace("```", "").strip()

        topic_name = extract_topic(clean_content)
        if not topic_name:
            print(f"  Warning: Could not extract topic from {md_path}")
            # Try to get it from the folder name if possible or use filename
            topic_name = os.path.basename(md_path).replace(".md", "").replace("-", " ").title()

        slug = slugify(topic_name)
        new_md_name = f"{slug}.md"
        new_html_name = f"{slug}.html"

        current_dir = os.path.dirname(md_path)
        new_md_path = os.path.join(current_dir, new_md_name)
        new_html_path = os.path.join(current_dir, new_html_name)

        # Update content if it was wrapped in markdown blocks
        if clean_content != content:
            with open(md_path, 'w') as f:
                f.write(clean_content)

        # Rename if necessary
        if os.path.abspath(md_path) != os.path.abspath(new_md_path):
            print(f"  Renaming {os.path.basename(md_path)} -> {new_md_name}")
            if os.path.exists(new_md_path):
                 os.remove(md_path)
            else:
                 os.rename(md_path, new_md_path)
            actual_md_path = new_md_path
        else:
            actual_md_path = md_path

        # Generate/Update HTML
        print(f"  Generating HTML: {new_html_name}")
        html_content = convert_md_to_html(clean_content, topic_name)
        with open(new_html_path, 'w') as f:
            f.write(html_content)

    print("Refreshing index.html...")
    update_index_page()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
