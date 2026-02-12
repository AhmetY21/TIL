---
title: "Regular Expressions for Text Processing"
date: "2026-02-12"
week: 7
lesson: 3
slug: "regular-expressions-for-text-processing"
---

# Topic: Regular Expressions for Text Processing

## 1) Formal definition (what is it, and how can we use it?)

A regular expression (regex or regexp) is a sequence of characters that define a *search pattern*. It's a powerful tool for pattern matching within text. Think of it as a miniature programming language designed specifically for describing text patterns.

Formally, regular expressions are a way to represent regular languages, which are a class of formal languages that can be recognized by a finite state machine. However, in practice, modern regular expression engines often implement features that go beyond strict regular languages (like backreferences).

We can use regular expressions for:

*   **Searching:** Finding occurrences of a specific pattern within a larger text.  For example, finding all email addresses in a document.
*   **Replacing:**  Replacing text that matches a pattern with a different string. For example, standardizing phone number formats.
*   **Validating:**  Checking if a piece of text conforms to a specific format. For example, validating that an input string is a valid date.
*   **Splitting:**  Dividing a string into multiple parts based on a pattern. For example, splitting a sentence into individual words.
*   **Extracting:**  Pulling out specific parts of a string that match a pattern. For example, extracting area codes from phone numbers.
## 2) Application scenario

Imagine you are working with a large dataset of customer reviews and you want to analyze the sentiment towards a specific product, say "SmartWatch X".

You could use regular expressions to:

1.  **Identify relevant reviews:** Search for reviews that contain the phrase "SmartWatch X" or variations like "Smart Watch X" or "SmartWatchX".  A regex like `Smart\s*Watch\s*X` (explained below) could handle these variations.

2.  **Extract product features mentioned:**  Look for patterns near the product name that mention specific features, like "battery life," "screen resolution," or "user interface."  For example, you might use a regex to find phrases like "SmartWatch X has great battery life" or "The screen resolution of SmartWatch X is disappointing."

3.  **Clean up the text data:**  Remove unwanted characters like extra spaces, HTML tags, or special symbols that might interfere with sentiment analysis. You could use regex for removing all HTML tags using a pattern like `<[^>]+>`.

In short, regular expressions help preprocess and extract structured information from unstructured text data (the reviews), making it suitable for further analysis.

## 3) Python method (if possible)

Python's `re` module provides regular expression operations.

```python
import re

text = "This is a sample text with SmartWatch X and Smart Watch X. Also contains email address example@domain.com and phone number 123-456-7890."

# 1. Searching for "SmartWatch X" (handling variations in whitespace)
pattern = r"Smart\s*Watch\s*X" #r"" denotes a raw string, preventing unintended escape sequences. \s matches any whitespace character, * matches zero or more occurences
matches = re.findall(pattern, text)
print(f"Matches for 'SmartWatch X': {matches}")

# 2. Extracting email addresses
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  #\b matches a word boundary.  [A-Za-z0-9._%+-]+ matches one or more alphanumeric characters, dots, underscores, %, +, or -.  @[A-Za-z0-9.-]+ matches @ followed by one or more alphanumeric characters, dots, or -.  \.[A-Z|a-z]{2,} matches a dot followed by two or more alphabetic characters.
email_matches = re.findall(email_pattern, text)
print(f"Email addresses found: {email_matches}")

# 3. Replacing phone number format
phone_pattern = r"(\d{3})-(\d{3})-(\d{4})" # Capture groups for area code, exchange, and line number
replaced_text = re.sub(phone_pattern, r"(\1) \2-\3", text) # \1, \2, and \3 are backreferences to the captured groups.
print(f"Text with reformatted phone number: {replaced_text}")

# 4. Splitting the string into sentences
sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s" #complex pattern that uses negative lookbehinds and lookaheads to split based on sentence end marks
sentences = re.split(sentence_pattern, text)
print(f"Sentences : {sentences}")
```

**Explanation of some regex elements:**

*   `\s`: Matches any whitespace character (space, tab, newline).
*   `*`: Matches zero or more occurrences of the preceding character or group.
*   `.`: Matches any character (except newline).
*   `[]`: Defines a character class, matching any character within the brackets.
*   `\d`: Matches a digit (0-9).
*   `\w`: Matches a word character (alphanumeric and underscore).
*   `+`: Matches one or more occurrences of the preceding character or group.
*   `\b`: Matches a word boundary.
*   `()`: Creates a capturing group, allowing you to extract or refer to the matched text.
*   `|`: acts as an "OR"

## 4) Follow-up question

How can regular expressions be used to detect and filter out spam emails based on common patterns in subject lines or email content? For example, what regex patterns might indicate a phishing email?