## 2024-03-24 - Initial Setup
**Learning:** Checking memory constraints and UX capabilities.
**Action:** Always verify before proceeding.

## 2025-02-14 - Global vs Local CSS Contexts in Generated HTML
**Learning:** When injecting styles into dynamically generated files (like markdown to HTML conversions), CSS custom properties (variables like `--primary`) might not be defined if they are only declared in the root `index.html`. Using them blindly leads to broken focus styles.
**Action:** Always verify if a template defines its own CSS variables before using them; fallback to hardcoded hex values (and explicit `.dark` variants) when styling independent generated artifacts like lessons.

## 2025-02-14 - Global vs Local CSS Contexts in Generated HTML
**Learning:** When injecting styles into dynamically generated files (like markdown to HTML conversions), CSS custom properties (variables like `--primary`) might not be defined if they are only declared in the root `index.html`. Using them blindly leads to broken focus styles.
**Action:** Always verify if a template defines its own CSS variables before using them; fallback to hardcoded hex values (and explicit `.dark` variants) when styling independent generated artifacts like lessons.
