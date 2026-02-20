# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-20 - Skip Link Implementation
**Learning:** When generating standalone HTML pages via Python f-strings, CSS variables (like `--primary`) defined in other templates (like `index.html`) are not available unless redefined in `:root`. Direct hex values or a shared CSS file are safer for standalone pages.
**Action:** When adding global styles to generated pages, either duplicate the `:root` block or use direct values to ensure consistency.
