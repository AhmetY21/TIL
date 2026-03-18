---
title: "Prompt Engineering Best Practices"
date: "2026-03-18"
week: 12
lesson: 4
slug: "prompt-engineering-best-practices"
---

# Topic: Prompt Engineering Best Practices

## 1) Formal definition (what is it, and how can we use it?)

Prompt engineering is the art and science of designing effective prompts for large language models (LLMs) like GPT-3, LaMDA, and others, to elicit desired and high-quality responses. A prompt is the input given to the LLM that guides its generation of text. Instead of simply asking a question, prompt engineering involves carefully crafting the prompt's phrasing, format, and context to steer the model toward the specific outcome you want.

We can use prompt engineering to:

*   **Improve accuracy:** By providing context and clear instructions, we can reduce the likelihood of the LLM generating incorrect or nonsensical answers.
*   **Control output style:** We can influence the tone, format, and length of the response (e.g., "Write a short poem...", "Explain this concept like I'm five...", "Summarize this article in three bullet points...").
*   **Extract specific information:** We can design prompts that guide the model to extract relevant data from a body of text or answer precise questions.
*   **Generate creative content:** We can use prompts to spark the LLM's creativity and generate stories, poems, code, or other forms of content.
*   **Mitigate biases:** While not a perfect solution, careful prompt design can help minimize the impact of biases present in the model's training data.

Ultimately, good prompt engineering results in more predictable, useful, and desirable outputs from LLMs.
## 2) Application scenario

Let's say you're building a customer service chatbot that uses an LLM to answer frequently asked questions. A naive approach might be to simply feed the user's question directly to the model. However, this might lead to inconsistent or inaccurate responses.

Instead, using prompt engineering, you can create a more structured prompt like this:

"You are a customer service chatbot for [Company Name]. Your goal is to answer customer questions accurately and helpfully. Please use the following knowledge base to answer the question below. If the knowledge base does not contain an answer, respond with 'I'm sorry, I don't have the information to answer that question.'

**Knowledge Base:**
[...Insert company FAQs and information here...]

**Question:** [User's question]"

This prompt provides the LLM with:

*   **Context:** Telling it its role (customer service chatbot) and goals.
*   **Instructions:** How to behave (accurately and helpfully).
*   **Information Source:** Providing a knowledge base for grounding its responses.
*   **Fallback Strategy:** Defining how to respond when information is unavailable.

By using this engineered prompt, the chatbot is much more likely to provide relevant, accurate, and helpful responses compared to simply feeding the raw user question to the model. This approach significantly improves the user experience.
## 3) Python method (if possible)
```python
import openai

def generate_response(prompt):
  """
  Generates a response from the OpenAI API given a prompt.

  Args:
    prompt: The prompt to send to the API.

  Returns:
    The response from the API as a string, or None if there's an error.
  """
  try:
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose an appropriate engine
        prompt=prompt,
        max_tokens=150,          # Adjust as needed
        n=1,                     # Number of responses to generate
        stop=None,               # Specify stop sequences, if any
        temperature=0.7,          # Adjust for creativity (0-1)
    )
    return response.choices[0].text.strip()
  except Exception as e:
    print(f"Error generating response: {e}")
    return None

# Example usage with prompt engineering:
company_name = "Acme Corp"
knowledge_base = """
Q: What are your hours?
A: We are open Monday-Friday, 9am-5pm.
Q: Do you offer refunds?
A: Yes, we offer full refunds within 30 days of purchase.
"""

user_question = "What are your hours?"

prompt = f"""
You are a customer service chatbot for {company_name}. Your goal is to answer customer questions accurately and helpfully. Please use the following knowledge base to answer the question below. If the knowledge base does not contain an answer, respond with 'I'm sorry, I don't have the information to answer that question.'

Knowledge Base:
{knowledge_base}

Question: {user_question}
"""

openai.api_key = "YOUR_OPENAI_API_KEY" # Replace with your actual API key

response = generate_response(prompt)

if response:
  print(response)
else:
  print("Failed to get a response.")
```

**Explanation:**

1.  **`generate_response(prompt)` function:**  This function takes a prompt string as input and uses the `openai` library to send it to the OpenAI API.  It handles potential errors and returns the generated text.
2.  **`engine`:**  Specifies the OpenAI model to use. Replace `"text-davinci-003"` with a more suitable model if needed, considering cost and performance trade-offs.
3.  **`max_tokens`:** Limits the length of the generated response.  Adjust based on the expected complexity of the answer.
4.  **`temperature`:**  Controls the randomness of the output. A lower value (e.g., 0.2) makes the response more deterministic, while a higher value (e.g., 0.9) introduces more creativity.
5.  **Prompt construction:**  The code demonstrates how to build the prompt string dynamically, inserting relevant information like the company name, knowledge base, and user question.  This allows for customization and flexibility.
6.  **API key:** Remember to replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key. You can obtain this from the OpenAI website.
7.  **Error Handling:** Includes a `try-except` block to gracefully handle potential errors during the API call.

**Important:** You will need to install the OpenAI Python library: `pip install openai`

## 4) Follow-up question

How can I systematically evaluate and compare the effectiveness of different prompt engineering techniques for a given task (e.g., using metrics)?  What metrics are most relevant for different types of LLM outputs (e.g., factual accuracy, fluency, coherence)?