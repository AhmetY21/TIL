# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - [Global Focus Outline Accessibility]
**Learning:** For a11y focus enhancements, using `*:focus-visible` ensures universal keyboard support while preventing unwanted focus rings for mouse interactions. Additionally, when using Playwright to test dynamically styled outlines, standard hex values may get converted to RGB representation (e.g., `#2563eb` -> `rgb(37, 99, 235)`).
**Action:** When asserting CSS `outline` values evaluated via Playwright's `getComputedStyle` in Chromium, remember to expect their RGB equivalents. Ensure f-string Python templates always escape raw CSS curly braces with double brackets `{{ }}`.
