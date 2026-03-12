---
title: "GPT-3 and Few-shot Learning"
date: "2026-03-12"
week: 11
lesson: 1
slug: "gpt-3-and-few-shot-learning"
---

# Topic: GPT-3 and Few-shot Learning

## 1) Formal definition (what is it, and how can we use it?)

**GPT-3** (Generative Pre-trained Transformer 3) is a powerful autoregressive language model developed by OpenAI. It's designed to generate human-quality text, translate languages, write different kinds of creative content, and answer your questions in an informative way. It's a transformer model, meaning it relies on the attention mechanism to weigh the importance of different parts of the input sequence. Crucially, GPT-3 was pre-trained on a massive dataset of text from the internet, giving it a broad understanding of language and the world.

**Few-shot learning** is a machine learning approach where a model is trained to generalize to new tasks from only a small number of training examples. This contrasts with traditional supervised learning which often requires large datasets. Few-shot learning is particularly useful when:

*   Data is scarce and expensive to obtain.
*   The task is new or highly specialized, making it difficult to gather enough training data.
*   Rapid adaptation to new tasks is required.

**GPT-3 and Few-shot Learning Combined:** GPT-3's strength lies in its ability to perform well in few-shot learning scenarios. Because of its massive pre-training, it has already learned a vast amount of knowledge about language and common tasks.  Therefore, it can be given just a few examples of a new task (the "few shots") and then, based on these examples, perform well on that task without any further explicit training. This is accomplished by including the examples in the prompt. The input prompt includes both examples of the desired input/output relationship AND the new input for which the model needs to generate the output. This is referred to as *in-context learning* because the training happens directly within the query itself.

There are different settings of in-context learning:

*   **Zero-shot learning:** The model is given *no* examples, just a task description.
*   **One-shot learning:** The model is given *one* example.
*   **Few-shot learning:** The model is given a *small number* of examples (typically between 2 and 10).

We can use GPT-3 and few-shot learning for various tasks such as text summarization, translation, question answering, code generation, and creative writing. The user provides a few examples of the desired input-output format, and GPT-3 attempts to generalize from these examples to produce relevant and accurate output.

## 2) Application scenario

Let's consider a scenario where we want to use GPT-3 to translate English sentences into French. We have a limited number of English-French sentence pairs.

**Traditional Approach (High Resource):**  We would typically need hundreds or thousands of English-French sentence pairs to train a dedicated translation model. This would require significant data collection and training efforts.

**Few-shot Approach with GPT-3:**  Instead, we can leverage GPT-3's pre-existing knowledge and provide a few example sentence pairs directly in the input prompt:

```
English: The cat sat on the mat. French: Le chat était assis sur le tapis.
English: I like to eat apples. French: J'aime manger des pommes.
English: The sun is shining brightly. French: Le soleil brille fort.
English: I am going to the store. French:
```

In this example, the first three lines provide the "few shots" – the training examples. The last line presents the input we want translated.  GPT-3, by recognizing the pattern from the provided examples, should be able to generate a reasonable French translation for "I am going to the store."

## 3) Python method (if possible)

To use GPT-3, you'll need an API key from OpenAI and the `openai` Python library.

```python
import openai

# Replace with your actual OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def translate_english_to_french(english_sentence, examples=None):
    """
    Translates an English sentence to French using GPT-3 with few-shot learning.

    Args:
        english_sentence: The English sentence to translate.
        examples: A list of example dictionaries, each with 'english' and 'french' keys.
                 Example: [{'english': 'Hello', 'french': 'Bonjour'}, ...]

    Returns:
        The translated French sentence, or None if an error occurs.
    """

    prompt = ""
    if examples:
        for example in examples:
            prompt += f"English: {example['english']} French: {example['french']}\n"
    prompt += f"English: {english_sentence} French:"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or another suitable engine
            prompt=prompt,
            max_tokens=60,             # Adjust as needed
            n=1,                       # Number of completions to generate
            stop=["\nEnglish:"],        # Stop generating when encountering "English:" to prevent repeating
            temperature=0.3,              # Lower temperature for more deterministic output, higher for more creative
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example Usage
examples = [
    {"english": "The dog is barking.", "french": "Le chien aboie."},
    {"english": "She is reading a book.", "french": "Elle lit un livre."},
]

english_text = "He is drinking coffee."
french_translation = translate_english_to_french(english_text, examples=examples)

if french_translation:
    print(f"English: {english_text}")
    print(f"French: {french_translation}")
else:
    print("Translation failed.")

```

**Explanation:**

1.  **Import `openai`:** Imports the OpenAI library.
2.  **Set API Key:**  Sets the `openai.api_key` using your secret API key obtained from OpenAI. *Do not share your API key!*
3.  **`translate_english_to_french()` function:**
    *   Takes the English sentence to translate and a list of example dictionaries as input.
    *   Constructs the prompt by concatenating the examples and the sentence to be translated.
    *   Calls `openai.Completion.create()` to generate a completion using the "text-davinci-003" engine (you can experiment with other engines).
        *   `prompt`: The input prompt containing the examples and the sentence.
        *   `max_tokens`: Limits the length of the generated French sentence.
        *   `n`:  Specifies the number of completions to generate (set to 1 here).
        *   `stop`: This parameter is crucial.  It tells the model to stop generating text when it encounters "\nEnglish:". This helps to prevent the model from continuing the example pattern beyond the translation and starting to generate new, unintended "English: French:" pairs.
        *   `temperature`: Controls the randomness of the output. Lower values (e.g., 0.3) result in more predictable and consistent translations.  Higher values (closer to 1) can lead to more creative, but potentially less accurate, results.
    *   Returns the generated French translation.
    *   Includes error handling to catch potential exceptions.
4.  **Example Usage:**
    *   Defines a list of example English-French sentence pairs.
    *   Calls `translate_english_to_french()` to translate a sample sentence.
    *   Prints the original English sentence and the generated French translation.

**Important Considerations:**

*   **API Key:**  Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.
*   **Engine Selection:** Experiment with different OpenAI engines to see which one performs best for your task. `"text-davinci-003"` is a generally good choice, but newer engines might offer better performance.
*   **Prompt Engineering:** The quality of the prompt is crucial.  Experiment with different examples and phrasing to optimize the model's performance.
*   **Token Limits:**  Be mindful of token limits.  GPT-3 has a maximum context window size. If your prompt and desired output exceed this limit, you'll need to shorten the prompt or use a different approach.
*   **Cost:** Using the OpenAI API incurs costs based on the number of tokens processed. Be aware of pricing and monitor your usage.

## 4) Follow-up question

How can we automatically determine the optimal number and quality of examples to include in the prompt for few-shot learning with GPT-3 to achieve the best performance for a given task, while minimizing API costs?