# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-03-12 - [Accessibility: Skip to Content Links]
**Learning:** Adding skip links retroactively requires wrapping existing main content in `<main id="main-content" tabindex="-1">`. Injecting an empty target div directly below the skip-link defeats the purpose as it won't actually skip navigation components. It is crucial to set `tabindex="-1"` on the target container so programmatic focus behaves properly, but outline reset is needed (`#main-content:focus { outline: none; }`) so we don't accidentally draw focus outlines around the whole page body when clicked.
**Action:** Always wrap main layout regions with `<main>` tags rather than relying on `<div>`s, and make sure that skip targets explicitly set `tabindex="-1"` while resetting their focus outline. Focus styles (`:focus-visible`) should be applied globally for consistency.
