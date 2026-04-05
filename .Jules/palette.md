# Palette's Journal

## 2026-02-13 - Retrofitting Navigation Headers
**Learning:** When injecting navigation components into existing HTML files, CSS conflicts (like absolute positioning on buttons) must be explicitly overridden (e.g., `position: static`) rather than just appending new styles, as the old styles might still apply.
**Action:** When migrating UI components, check for and neutralize conflicting legacy CSS properties in the migration script.

## 2026-02-13 - [Adding Copy Button to Code Blocks]
**Learning:** When injecting JavaScript into Python f-string templates, remember to escape curly braces (`{` -> `{{`, `}` -> `}}`) to avoid syntax errors and runtime crashes. This is especially tricky when mixing CSS and JS in the same f-string block.
**Action:** Always verify f-string template injections with a unit test that parses the output or checks for successful execution.

## 2026-02-13 - [Global focus-visible outline for Keyboard Accessibility]
**Learning:** To maintain accessibility standards (a11y) and ensure uniform coverage across all interactive elements, apply `:focus-visible` globally using the universal selector (e.g., `*:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }`) to support and visually indicate keyboard navigation. This provides a consistent focus indicator without relying on custom implementations per element, though it requires specific `.dark` mode handling if CSS variables aren't used.
**Action:** Always include global `:focus-visible` styles with appropriate outline and outline-offset in the base CSS to guarantee a focus ring for keyboard users, and ensure there's a `.dark` variant for visibility across themes.
