# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-14 - Verifying :focus-visible states with Playwright
**Learning:** When visually verifying focus states (like `:focus-visible`) using Playwright, simulating clicks on interactive elements after focusing the target element will shift the DOM focus and remove the outline. Instead, use `page.keyboard.press('Tab')` to simulate keyboard navigation, and avoid clicking to ensure the focus state is accurately preserved.
**Action:** Use simulated keyboard events (like `Tab`) and `getComputedStyle` evaluation without subsequent clicks to accurately capture and verify focus styles in automated visual tests.
