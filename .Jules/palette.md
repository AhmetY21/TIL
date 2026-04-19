# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2024-06-25 - Implement Keyboard Accessibility and Skip Links
**Learning:** Keyboard accessibility requires a universal `:focus-visible` to ensure users navigating via Tab clearly see their focus state without showing focus rings during mouse clicks. Additionally, skip links must target a specific container (e.g., `#main-content`) with `tabindex="-1"` so it can programmatically receive focus, and the targeted container must have `outline: none` to prevent a large focus ring around the whole page body when the skip link is activated.
**Action:** Always implement `*:focus-visible` over `:focus` for interactive elements, and ensure retrofitted skip links target a proper `<main id="main-content" tabindex="-1">` wrapper instead of just an empty anchor div, removing the outline on focus. Ensure all these styles include corresponding `.dark` mode implementations.
