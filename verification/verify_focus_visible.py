import sys
import os
from playwright.sync_api import sync_playwright

def test_focus_visible():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Get absolute path to index.html
        abs_path = os.path.abspath("index.html")
        page.goto(f"file://{abs_path}")

        # Focus the first hub card
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")

        # Take a screenshot
        page.screenshot(path="verification/focus_light.png")

        # Toggle dark mode
        page.click("#theme-toggle")

        # Focus again
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")

        # Take another screenshot
        page.screenshot(path="verification/focus_dark.png")

        browser.close()

if __name__ == "__main__":
    test_focus_visible()
