---
title: "Regular Expressions for Text Processing"
date: "2026-03-03"
week: 10
lesson: 2
slug: "regular-expressions-for-text-processing"
---

# Topic: Regular Expressions for Text Processing

## 1) Formal definition (what is it, and how can we use it?)

Regular expressions (often shortened to "regex" or "regexp") are a powerful way to describe patterns in text.  A regular expression is a sequence of characters that define a search pattern. They are essentially a domain-specific language for pattern matching.  We can use them for:

*   **Searching:** Finding occurrences of specific patterns within a text.
*   **Validation:** Checking if a string conforms to a specific format (e.g., email address, phone number).
*   **Substitution:** Replacing matched patterns with different text.
*   **Extraction:**  Pulling out specific pieces of information from a text based on a pattern (e.g., extracting dates, phone numbers).
*   **Tokenization:** Splitting text into meaningful units based on defined patterns.

Regular expressions utilize metacharacters (special characters with specific meanings) to represent different kinds of patterns. Common metacharacters include:

*   `.`: Matches any single character (except newline).
*   `^`: Matches the beginning of a string.
*   `$`: Matches the end of a string.
*   `*`: Matches zero or more occurrences of the preceding character or group.
*   `+`: Matches one or more occurrences of the preceding character or group.
*   `?`: Matches zero or one occurrences of the preceding character or group.
*   `[]`: Defines a character class (matches any character within the brackets).  For example, `[abc]` matches 'a', 'b', or 'c'.
*   `[^]`: Defines a negated character class (matches any character *not* within the brackets). For example, `[^abc]` matches any character except 'a', 'b', or 'c'.
*   `\d`: Matches a digit (0-9).
*   `\D`: Matches a non-digit character.
*   `\w`: Matches a word character (alphanumeric and underscore).
*   `\W`: Matches a non-word character.
*   `\s`: Matches whitespace (space, tab, newline).
*   `\S`: Matches non-whitespace.
*   `()`: Creates a capturing group (allows you to extract the matched text).
*   `|`:  Represents an "or" condition (e.g., `a|b` matches 'a' or 'b').
*   `{n}`: Matches exactly n occurrences of the preceding character or group.
*   `{n,m}`: Matches between n and m occurrences (inclusive) of the preceding character or group.
*   `\`: Escapes special characters (e.g., `\.` matches a literal period).

## 2) Application scenario

**Scenario: Extracting email addresses from a text document.**

Imagine you have a large text document containing various information, and you need to extract all the email addresses. A regular expression can be used to identify and extract these email addresses efficiently.  A common regex for email addresses (while not perfect for *all* possible valid addresses, it's a good starting point) is:

`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`

This pattern breaks down as follows:

*   `[a-zA-Z0-9._%+-]+`:  Matches one or more alphanumeric characters, periods, underscores, percent signs, plus signs, or hyphens (the part before the `@`).
*   `@`: Matches the "@" symbol.
*   `[a-zA-Z0-9.-]+`: Matches one or more alphanumeric characters, periods, or hyphens (the domain name part).
*   `\.`: Matches a literal period.
*   `[a-zA-Z]{2,}`: Matches two or more alphabetic characters (the top-level domain, like "com", "org", "net").

Using this regex, you can search the text document and extract all strings that match this email pattern.

## 3) Python method (if possible)

Python's `re` module provides regular expression operations.

```python
import re

text = "Contact us at support@example.com or sales@another-example.net for assistance.  My personal email is test.user123@sub.domain.co.uk."

# Regex pattern for email addresses
pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  #raw string to avoid escaping backslashes

# Find all matches
emails = re.findall(pattern, text)

# Print the extracted email addresses
print(emails)

# Using search to find the FIRST occurance
match = re.search(pattern, text)
if match:
  print("First Email Found (using search):", match.group(0)) # group(0) contains the entire match
else:
  print("No Email found")

# Using sub to replace all emails with "REDACTED"
redacted_text = re.sub(pattern, "REDACTED", text)
print("Redacted text: ", redacted_text)

```

Key functions used:

*   `re.findall(pattern, string)`:  Returns a list of all non-overlapping matches of the pattern in the string.
*   `re.search(pattern, string)`:  Searches the string for the first occurrence of the pattern and returns a match object if found; otherwise, it returns `None`.  The `match.group(0)` method retrieves the entire matched string.
*   `re.sub(pattern, replacement, string)`: Replaces all occurrences of the pattern in the string with the replacement string.

Note the `r` prefix before the regex string. This denotes a "raw string" and prevents Python from interpreting backslashes as escape sequences, which is crucial for regex patterns containing backslashes (like `\d`, `\w`, etc.).  Without the raw string prefix, you would need to double-escape the backslashes (e.g., `\\d`).

## 4) Follow-up question

How would you modify the regex from the application scenario to handle email addresses that have a country-specific top-level domain (e.g., `.ca`, `.fr`, `.de`) while still allowing for the common `.com`, `.net`, and `.org` domains, AND ensuring the top-level domain is between 2 and 4 characters long?