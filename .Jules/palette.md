# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-23 - Focus Visible Accessibility Enhancement
**Learning:** Adding `:focus-visible` globally (`*:focus-visible`) provides excellent keyboard navigation support while preventing unwanted focus rings during mouse interactions. When retrofitting this, it's crucial to implement it across all templates (like static shells and Python HTML generators) and ensure dark mode variants maintain contrast against darker backgrounds.
**Action:** When implementing accessibility enhancements, always check both the static template files and the dynamic generators to ensure consistent behavior across the entire app. Use the `:focus-visible` pseudo-class for focus indicators to balance a11y and aesthetics.
