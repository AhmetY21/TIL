# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2024-04-30 - Focus Visible vs Focus
**Learning:** For interactive elements like theme toggles, applying a standard `:focus` style draws a distracting ring when users click it with a mouse. Using the `:focus-visible` pseudo-class universally (`*:focus-visible`) provides essential visual feedback for keyboard navigators without annoying mouse/touch users.
**Action:** When adding global accessibility focus styles, prefer `*:focus-visible` over `*:focus` and ensure it has an explicit `.dark` mode color override to maintain contrast.
