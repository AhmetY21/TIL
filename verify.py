from playwright.sync_api import sync_playwright
import os

def test_skip_link(page, url, prefix):
    page.goto(url)

    # Press Tab to focus on the first element (which should be the skip-link)
    page.keyboard.press("Tab")

    # Capture a screenshot showing the focused skip link
    page.screenshot(path=f"{prefix}_focused.png")

    # Verify focus-visible style
    skip_link = page.locator(".skip-link")
    print(f"[{prefix}] skip-link visible? {skip_link.is_visible()}")

    # Verify pressing enter jumps to main-content
    page.keyboard.press("Enter")
    main_content = page.locator("#main-content")
    print(f"[{prefix}] main-content visible? {main_content.is_visible()}")
    print(f"[{prefix}] current focused element id: {page.evaluate('document.activeElement.id')}")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    # Test index.html
    abs_index_path = f"file://{os.path.abspath('index.html')}"
    test_skip_link(page, abs_index_path, "index")

    # Test hub
    abs_hub_path = f"file://{os.path.abspath('hubs/nlp-index.html')}"
    test_skip_link(page, abs_hub_path, "hub")

    # Test lesson
    abs_lesson_path = f"file://{os.path.abspath('topic/nlp/week_11/day_2026-03-10/lesson_3/positional-encodings-in-transformers.html')}"
    test_skip_link(page, abs_lesson_path, "lesson")

    browser.close()
