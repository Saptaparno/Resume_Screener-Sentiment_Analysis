import os
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
def extract_text(file_path):
    elements = partition(file_path)
    if not elements:
        raise ValueError(f"No text found in file {file_path}")
    return "\n".join(el.text for el in elements if el.text)
screening_template = """
You are an expert HR assistant.

Given the Job Description and Resume below, evaluate the candidate's fit.

Job Description:
{job_desc}

Resume:
{resume}

Give a score out of 100 and a short explanation.
"""
sentiment_template = """
Analyze the tone and sentiment of the following cover letter:

{cover_letter}

Is it positive, neutral, or negative? Explain briefly.
"""
comparison_template = """
You are a hiring expert.

Here is the previous candidate evaluation:
{previous_candidate}

Here is the current candidate evaluation:
{current_candidate}

Who is the better fit and why? Make a clear recommendation.
"""
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
screening_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(screening_template))
sentiment_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(sentiment_template))
comparison_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(comparison_template), memory=memory)
def screen_candidate(jd_path, resume_path, cover_letter_path):
    job_desc = extract_text(jd_path)
    resume = extract_text(resume_path)
    cover_letter = extract_text(cover_letter_path)
    fit_result = screening_chain.run(job_desc=job_desc, resume=resume)
    sentiment_result = sentiment_chain.run(cover_letter=cover_letter)
    current_eval = f"Fit:\n{fit_result}\nSentiment:\n{sentiment_result}"
    print("\nCurrent Candidate Evaluation:\n", current_eval)
    if memory.chat_memory.messages:
        previous_eval = memory.chat_memory.messages[-1].content
        comparison = comparison_chain.run(
            previous_candidate=previous_eval,
            current_candidate=current_eval
        )
        print("\nComparison Result:\n", comparison)
    else:
        print("\nNo previous candidate to compare.")
    memory.save_context({"input": "Candidate Evaluation"}, {"output": current_eval})
if __name__ == "__main__":
    print("Enter file paths for job description, resume, and cover letter:")
    jd_path = input("Job Description path: ").strip()
    resume_path = input("Resume path: ").strip()
    cl_path = input("Cover Letter path: ").strip()
    screen_candidate(jd_path, resume_path, cl_path)
