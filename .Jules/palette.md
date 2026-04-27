# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - Focus Visible in Python f-strings
**Learning:** When adding `*:focus-visible` pseudo-class definitions inside Python f-strings (like in `generate_lesson.py`), it's crucial to double-escape the CSS block braces (`{{ ... }}`) and ensure variables like `--primary` match the defined CSS context. Static pages use `--primary`, whereas standard lesson pages don't define the CSS variables, requiring hardcoded hex values instead.
**Action:** When working with dynamic CSS inside HTML generation templates, carefully verify whether the parent context defines specific CSS variables or requires explicit hex color fallback, and verify curly brace syntax immediately.
