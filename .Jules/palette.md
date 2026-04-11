# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2024-04-11 - Adding robust universal focus-visible styles
**Learning:** Adding keyboard accessibility focus states (`:focus-visible`) globally is best achieved using a universal selector (`*:focus-visible`) rather than targeting specific elements. However, in dynamically generated Python templating systems where CSS custom variables might not be defined for all template contexts, specific fallback hardcoded color variables with explicit `.dark` mode declarations (`.dark *:focus-visible`) must be added to ensure the focus outline remains highly visible across all color modes and generated artifacts.
**Action:** Always add universal `:focus-visible` styles with a solid 2px outline and 2px outline-offset globally to interactive elements, ensuring the color provides good contrast, and verify whether a dedicated `.dark` mode override is necessary depending on whether the stylesheet context natively supports CSS variables.
