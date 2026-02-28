# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - Focus Styles Keyboard Navigation
**Learning:** Adding visible focus states (`:focus-visible`) is critical for keyboard accessibility. A simple `outline: 2px solid var(--primary); outline-offset: 2px;` goes a long way. Retrofitting requires a script that modifies both `generate_lesson.py` (for future templates) and existing static HTMLs. It is also important to test dummy test files are not accidentally committed when checking project context.
**Action:** Always implement `:focus-visible` for all interactive elements in custom UI projects. Remember to verify the correct `.dark` mode focus state colors when dealing with dark mode toggles.
