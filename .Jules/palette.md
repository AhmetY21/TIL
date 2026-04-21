# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - Focus Visible A11y
**Learning:** Using `*:focus-visible` provides global keyboard a11y coverage across all interactive elements, improving keyboard navigation visibility while preventing unwanted focus rings during mouse interactions.
**Action:** Default to `*:focus-visible` over `:focus` for uniform accessibility. Ensure these implementations are mirrored consistently across static shells (like `index.html`) and dynamic HTML generators.
