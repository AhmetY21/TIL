# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-03-10 - Retrofitting Skip Links
**Learning:** When adding accessibility features like skip links to legacy HTML files or Python generator templates, simply inserting an empty `<div id="main-content">` fails to skip actual navigation elements. Proper implementation requires wrapping the actual main content body in `<main id="main-content" tabindex="-1">` so programmatic focus lands exactly on the actionable content area. Additionally, focus outlines must explicitly support dark mode (`.dark *:focus-visible`).
**Action:** Always wrap the core UI in a focusable `<main>` element and test keyboard navigation across all themes when implementing a skip-link.
