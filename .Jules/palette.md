# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-03-01 - [Adding :focus-visible Styles]
**Learning:** Global CSS variables defined in `:root` (e.g. `var(--primary)`) are essential for maintaining consistent focus indicator colors across different components and dark/light modes. Explicitly defining `:focus-visible` with `outline` and `outline-offset` is critical for keyboard accessibility.
**Action:** Ensure all interactive elements have a defined `:focus-visible` state using a solid outline with an offset to provide clear visual feedback for keyboard users.
