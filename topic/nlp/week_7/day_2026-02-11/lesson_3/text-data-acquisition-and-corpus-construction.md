---
title: "Text Data Acquisition and Corpus Construction"
date: "2026-02-11"
week: 7
lesson: 3
slug: "text-data-acquisition-and-corpus-construction"
---

# Topic: Text Data Acquisition and Corpus Construction

## 1) Formal definition (what is it, and how can we use it?)

Text data acquisition and corpus construction refers to the process of gathering and organizing text data into a structured collection (a corpus) that can be used for various natural language processing (NLP) tasks.

*   **Text Data Acquisition:** This involves identifying sources of textual data, such as websites, books, news articles, social media feeds, and databases, and then collecting that data through methods like web scraping, API calls, manual downloads, or using existing datasets.  Crucially, this step includes considerations of legality (copyright, terms of service), ethics (privacy, bias), and data quality (noise, inconsistency).

*   **Corpus Construction:** Once text data is acquired, it needs to be organized and structured into a corpus.  This involves cleaning the data (removing irrelevant content, handling encoding issues), potentially annotating the data (part-of-speech tagging, named entity recognition, sentiment analysis), and storing it in a format that is suitable for NLP algorithms (e.g., plain text files, XML, databases).  The specific structure of the corpus will depend on the intended application. A corpus can be as simple as a collection of text files, or it can be a highly structured database with rich metadata and annotations.

**How can we use it?**

A well-constructed corpus is essential for:

*   **Training NLP models:** Machine learning models learn patterns from data. A corpus provides the training data for tasks like text classification, machine translation, and language modeling.
*   **Evaluating NLP models:**  A separate corpus (test set) is needed to evaluate the performance of trained models.
*   **Linguistic analysis:** Corpora allow linguists to study language usage patterns, frequency of words and phrases, and changes in language over time.
*   **Information retrieval:**  Corpora can be used to build search engines and other information retrieval systems.
*   **Developing language resources:** Corpora serve as a basis for creating lexicons, grammars, and other language resources.
*   **Building question answering system:** The data is used to train the model to answer questions given the source text.

## 2) Application scenario

**Scenario:**  A company wants to build a sentiment analysis model to analyze customer reviews of their products on Amazon.

**Text Data Acquisition:**

*   **Source:**  Amazon product review pages.
*   **Method:** Web scraping using Python libraries like `BeautifulSoup` and `requests` to extract the review text and rating scores.  Consider using Amazon's Product Advertising API (if available and permissible) for a more structured approach.

**Corpus Construction:**

1.  **Cleaning:**  Remove HTML tags, irrelevant characters, and duplicate reviews.  Handle potential encoding issues (e.g., UTF-8).
2.  **Annotation:**  The rating score (e.g., 1-5 stars) can be used as a label for sentiment (e.g., 1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive).  Manual annotation might be needed to refine labels if the star ratings are not reliable.
3.  **Structuring:**  Store the reviews and their sentiment labels in a CSV file or a database.  Consider adding metadata such as product ID, reviewer ID, and date of the review.
4.  **Splitting:** Divide the corpus into training, validation, and testing sets for model development and evaluation.

## 3) Python method (if possible)

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_amazon_reviews(url, num_pages=1):
    """
    Scrapes Amazon product reviews from a given URL.

    Args:
        url (str): The URL of the Amazon product review page.
        num_pages (int): The number of pages to scrape.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped reviews and ratings.
    """
    reviews = []
    ratings = []

    for page in range(1, num_pages + 1):
        page_url = f"{url}&pageNumber={page}"
        response = requests.get(page_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')
        review_elements = soup.find_all('span', class_='a-size-base review-text')
        rating_elements = soup.find_all('i', class_='a-icon-star') # Find all rating elements

        # Only proceed if both reviews and ratings are present
        if review_elements and rating_elements:
            # Extract review text
            for review_element in review_elements:
                reviews.append(review_element.text.strip())

            # Extract ratings: Take the 1st element for each page (typically the rating title), then
            # slice it from the 2nd element onwards for each subsequent review
            for i in range(1, len(rating_elements)):
                rating_text = rating_elements[i].text.strip()
                rating = rating_text.split(' ')[0]  # Extract the rating number from the text e.g., '4.0 out of 5 stars' -> '4.0'
                ratings.append(float(rating))

    df = pd.DataFrame({'review': reviews, 'rating': ratings})
    return df

# Example usage:  Replace with a real Amazon product review URL.  This is a placeholder.
amazon_url = "https://www.amazon.com/product-reviews/B0XXXXXXXXXX/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
try:
    review_df = scrape_amazon_reviews(amazon_url, num_pages=2) # Scrape 2 pages
    print(review_df.head())
except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

**Explanation:**

1.  **`scrape_amazon_reviews(url, num_pages)`:**  This function takes the Amazon product review URL and the number of pages to scrape as input.
2.  **`requests.get(page_url)`:**  Uses the `requests` library to fetch the HTML content of the review page.
3.  **`BeautifulSoup(response.content, 'html.parser')`:**  Parses the HTML content using `BeautifulSoup`.
4.  **`soup.find_all(...)`:**  Finds all the review text elements (spans with class `a-size-base review-text`) and the rating elements (i elements with class `a-icon-star`) on the page.  **Important:**  Amazon's HTML structure can change, so these selectors might need to be updated.
5.  **`review_element.text.strip()`:** Extracts the review text from each element and removes leading/trailing whitespace.
6.  The function iterates through the found elements and extracts the text and ratings.  It includes error handling.
7.  **`pd.DataFrame({'review': reviews, 'rating': ratings})`:**  Creates a Pandas DataFrame to store the scraped data.
8.  **Error Handling:** The `try...except` block catches potential `requests` exceptions (e.g., network errors, invalid URLs) and other exceptions during parsing.

**Important Considerations:**

*   **Amazon's Terms of Service:**  Web scraping can violate Amazon's terms of service.  Use caution and adhere to their guidelines. Consider using the Amazon Product Advertising API if available.
*   **Rate Limiting:**  Amazon might implement rate limiting to prevent excessive scraping. Implement delays between requests to avoid being blocked.
*   **HTML Structure Changes:**  Amazon's website structure can change frequently, which could break the scraper.  Monitor the scraper regularly and update the selectors if needed.
*   **Robustness:**  This is a basic example.  A production-quality scraper would need to handle various error conditions and edge cases.
*   **Legal and Ethical Considerations:**  Always be mindful of privacy and legal regulations when collecting and using data.  Obtain consent where necessary.
*   **API Alternatives:** Always check if the platform offers an API before resorting to scraping, as APIs are designed for programmatic data access and are generally more reliable and compliant with the platform's rules.

## 4) Follow-up question

How do you handle imbalanced datasets in corpus construction, where one class (e.g., positive sentiment reviews) is significantly more prevalent than others (e.g., negative sentiment reviews), and how does this imbalance affect model training and evaluation?