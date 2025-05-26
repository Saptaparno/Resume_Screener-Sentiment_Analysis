# 🧠 HR Assistant: Resume Screener & Comparator

This beginner-friendly Python app uses **LangChain**, **OpenAI**, and the **Unstructured** library to:

- 📝 Evaluate candidate fit for a job.
- 💬 Analyze the tone of a cover letter.
- 🧠 Compare the current candidate to the previous one using memory.
- ✅ Recommend which candidate to hire.

---

## 📦 Features

- Extracts text from **PDF**, **DOCX**, and other file formats.
- Uses **LLM Chains** for reasoning and decision-making.
- Remembers previous candidate data using `ConversationBufferMemory`.
- Makes clear recommendations based on analysis.

---

## 🛠 Requirements

Install the required packages:

```bash
pip install langchain openai unstructured python-dotenv
pip install "unstructured[local-inference-docs]"
