# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - [Global Focus Visible Styling]
**Learning:** When applying global `*:focus-visible` accessibility styles using a universal selector, it's crucial to explicitly provide a `.dark *:focus-visible` fallback if CSS variables aren't defined in all HTML contexts (like isolated dynamically generated f-string templates). Otherwise, focus rings might lose contrast or disappear entirely in dark mode.
**Action:** Always verify that focus indicator colors dynamically adapt to the active theme (e.g., using `--primary` vs `#60a5fa`) across all generated HTML outputs.
