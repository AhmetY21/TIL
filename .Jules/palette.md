# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-14 - Adding Skip Links to Generated Templates
**Learning:** When adding CSS blocks inside Python f-strings, standard CSS syntax like `.class { ... }` will cause a `NameError` or `ValueError` because Python tries to interpret the content inside `{}` as expressions.
**Action:** Double all curly braces in CSS blocks within f-strings (e.g., `.class {{ property: value; }}`) to escape them.

## 2026-02-14 - Focus Management for Skip Links
**Learning:** Simply linking to an ID (e.g., `<a href="#main-content">`) scrolls the page but does not always move keyboard focus to the target container, especially if it's a non-interactive element like `<main>` or `<div>`.
**Action:** Always add `tabindex="-1"` to the target container of a skip link to ensure focus is programmatically transferable.
