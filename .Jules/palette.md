# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - Universal Keyboard Focus Visibility
**Learning:** Adding universal `*:focus-visible` outlines is a low-effort, high-impact accessibility win, but dynamically generated HTML files using Python f-strings require CSS braces to be explicitly escaped (`{{` and `}}`) to prevent runtime string-formatting exceptions. CSS variable usage also needs to be explicitly validated per-template context, as lightweight generators may lack central stylesheets.
**Action:** Always verify CSS string injection inside f-strings with local execution or unit tests, and rely on hardcoded fallback colors if custom properties (like `var(--primary)`) are not guaranteed to be defined in the template's minimal head scope.
