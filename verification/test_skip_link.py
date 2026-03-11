from playwright.sync_api import sync_playwright, expect
import os

def test_skip_link(page):
    # Get absolute path to index.html
    current_dir = os.path.abspath(os.getcwd())
    file_url = f"file://{current_dir}/index.html"

    page.goto(file_url)

    # Focus the skip link via keyboard
    page.keyboard.press('Tab')

    # Check if skip link is focused and visible
    skip_link = page.locator('.skip-link')
    expect(skip_link).to_be_focused()

    # Take screenshot of focus state
    page.screenshot(path="verification/skip_link_focus.png")

    # Press enter to skip to main content
    page.keyboard.press('Enter')

    # Check if main content has focus
    main_content = page.locator('#main-content')
    expect(main_content).to_be_focused()

    # Take screenshot of main content focus
    page.screenshot(path="verification/main_content_focus.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            test_skip_link(page)
            print("Playwright test completed successfully.")
        finally:
            browser.close()
