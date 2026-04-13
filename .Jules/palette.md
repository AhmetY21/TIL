# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2024-05-18 - Implement Universal Focus-Visible Pattern
**Learning:** To support keyboard accessibility effectively, we learned to prioritize the `:focus-visible` pseudo-class over `:focus`. This prevents unwanted focus rings for mouse users while still maintaining clear, visible focus rings for keyboard navigators. Using the universal selector (`*:focus-visible`) applies the pattern broadly without adding complex classes.
**Action:** When implementing new UI elements or migrating standard controls, consistently apply a `2px solid` outline using `--primary` (or appropriate hex value), an `outline-offset` of `2px`, and include matching `.dark` mode implementations.
