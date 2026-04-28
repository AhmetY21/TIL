# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.
## 2026-02-14 - Skip Link Navigation
**Learning:** When adding skip links retroactively, wrap the target content body in `<main id="main-content" tabindex="-1">`. Blindly injecting an empty target div fails to skip navigation elements properly.
**Action:** When adding accessibility features like skip links retroactively, ensure the target is a structural wrapper and remember to hide its outline.
