import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Get absolute path to index.html
        cwd = os.getcwd()
        index_path = f"file://{cwd}/index.html"

        print(f"Navigating to {index_path}")
        page.goto(index_path)

        # Initial screenshot (skip link should be hidden)
        page.screenshot(path="verification/before_tab.png")

        # Press Tab to focus the first element (should be skip link)
        page.keyboard.press("Tab")

        # Check if skip link is focused
        focused_element_class = page.evaluate("document.activeElement.className")
        focused_element_text = page.evaluate("document.activeElement.innerText")
        print(f"Focused element class: {focused_element_class}")
        print(f"Focused element text: {focused_element_text}")

        if "skip-link" in focused_element_class:
            print("SUCCESS: Skip link is focused.")
        else:
            print(f"FAILURE: focused element is {focused_element_class}")

        # Take screenshot of focused state
        page.screenshot(path="verification/after_tab.png")

        browser.close()

if __name__ == "__main__":
    run()
