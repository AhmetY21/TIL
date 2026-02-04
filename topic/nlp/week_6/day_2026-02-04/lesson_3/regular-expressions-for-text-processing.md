Topic: Regular Expressions for Text Processing

1- Provide formal definition, what is it and how can we use it?

A regular expression (regex or regexp) is a sequence of characters that define a search pattern. Essentially, it's a powerful tool used to match, locate, and manage text based on defined patterns. Regular expressions are composed of literal characters and special characters (metacharacters) that have specific meanings.

*   **Formal Definition:** A regular expression is a formal language consisting of constants and operators that denote sets of strings and operations over these sets. More formally, they can be defined recursively as follows:
    *   ε (the empty string) is a regular expression.
    *   If 'a' is a symbol (a single character), then 'a' is a regular expression.
    *   If R and S are regular expressions, then:
        *   R|S (alternation): R or S
        *   RS (concatenation): R followed by S
        *   R* (Kleene star): Zero or more occurrences of R
    are also regular expressions.

*   **How can we use it?**  Regular expressions are crucial for:
    *   **Pattern Matching:**  Finding specific text sequences within a larger body of text. For example, finding all email addresses or phone numbers in a document.
    *   **Text Extraction:** Isolating and extracting specific parts of text that match a pattern.  For example, extracting all dates in a particular format.
    *   **Text Validation:** Verifying that a string conforms to a defined format. For example, ensuring a password meets certain complexity requirements.
    *   **Text Substitution:** Replacing text that matches a pattern with a different string. For example, replacing all instances of "color" with "colour".
    *   **Data Cleaning:** Cleaning and standardizing textual data by removing unwanted characters or formatting.
    *   **Tokenization:**  Breaking down text into smaller units, such as words or sentences, based on defined patterns.

2- Provide an application scenario

**Application Scenario: Sentiment Analysis of Customer Reviews**

Imagine you're analyzing customer reviews for a product on an e-commerce website. You want to automatically identify reviews that contain certain keywords associated with positive or negative sentiment.

For example:

*   Positive Keywords: "excellent", "amazing", "fantastic", "love", "recommend"
*   Negative Keywords: "terrible", "awful", "bad", "disappointed", "problem"

You can use regular expressions to:

1.  **Detect the presence of these keywords in each review.** This provides a preliminary indication of sentiment.
2.  **Extract surrounding phrases:** For instance, if a review contains "absolutely amazing", you can use regex to extract "absolutely amazing" and analyze the context to strengthen sentiment determination.
3.  **Identify negated phrases:** If a review contains "not good" or "not happy", a regex can help identify these negations and reverse the sentiment polarity of "good" or "happy" in those contexts.
4.  **Handle variations in spelling and grammar:** Regex can accommodate slight variations in keyword spellings (e.g., "reccomend" instead of "recommend") or handle plural forms.

Without regular expressions, achieving this level of nuanced text analysis would be significantly more difficult and require writing far more complex procedural code.  Regex allows you to express the search patterns concisely and efficiently.

3- Provide a method to apply in python

Python's `re` module provides comprehensive support for regular expressions.  Here's a demonstration focusing on finding email addresses in a text:

python
import re

text = "Contact us at support@example.com or sales@another-example.org for assistance. Also, try info@yet-another.com."

# Regular expression for matching email addresses
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Find all matches in the text
emails = re.findall(email_pattern, text)

# Print the extracted email addresses
print(emails)

# Example of using re.search to find the first email
first_email_match = re.search(email_pattern, text)
if first_email_match:
    print("First email found:", first_email_match.group(0)) #Access the matched string using group(0)

#Example of replacing the emails with [REDACTED]

redacted_text = re.sub(email_pattern, "[REDACTED]", text)
print(redacted_text)



**Explanation:**

*   `import re`: Imports the regular expression module.
*   `email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"`: Defines the regular expression pattern for an email address.  The `r` prefix indicates a raw string, which prevents backslashes from being interpreted as escape sequences. Let's break down the pattern:
    *   `[a-zA-Z0-9._%+-]+`: Matches one or more alphanumeric characters, periods, underscores, percent signs, plus signs, or hyphens (the part before the @ symbol).
    *   `@`: Matches the "@" symbol.
    *   `[a-zA-Z0-9.-]+`: Matches one or more alphanumeric characters, periods, or hyphens (the domain name).
    *   `\.`: Matches a literal period (escaped with a backslash).
    *   `[a-zA-Z]{2,}`: Matches two or more alphabetic characters (the top-level domain, e.g., "com", "org", "net").
*   `re.findall(email_pattern, text)`: Finds all occurrences of the `email_pattern` in the `text` and returns them as a list.
*   `re.search(email_pattern, text)`: Returns a match object if found in the string. Useful if you only want to find the first occurence.
*   `re.sub(email_pattern, "[REDACTED]", text)`: Replaces all matches of the `email_pattern` with "[REDACTED]" in the text.

4- Provide a follow up question about that topic

**Follow-up Question:**

How can regular expressions be used to handle different date formats (e.g., "MM/DD/YYYY", "YYYY-MM-DD", "DD-MMM-YYYY") within the same text corpus, and how would you normalize these different formats into a consistent format using Python's `re` module and the `datetime` library?