# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-15 - Adding Skip Navigation and Focus Indicators
**Learning:** When adding accessibility features like skip-to-content links, wrapping the target content in a `<main id="main-content" tabindex="-1">` is crucial to ensure the browser successfully programmatically shifts focus when the user navigates. Furthermore, adding `#main-content:focus { outline: none; }` prevents an ugly browser default focus ring from appearing around the entire main content block.
**Action:** Always combine the `href="#main-content"` skip link with a `<main id="main-content" tabindex="-1">` target container, and explicitly suppress its focus outline. Also ensure `*:focus-visible` styles are globally defined (including dark mode variants) so keyboard users have clear visual focus indicators on all interactive elements.