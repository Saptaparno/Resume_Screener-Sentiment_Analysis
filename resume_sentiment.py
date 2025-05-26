import os
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
def extract_text(file_path):
    elements = partition(file_path)
    return "\n".join(el.text for el in elements if el.text)
candidate_texts = {}
def upload_candidate(candidate_name):
    print(f"Uploading for {candidate_name}")
    jd_path = input("Enter Job Description file path: ").strip()
    resume_path = input("Enter Resume file path: ").strip()
    cover_letter_path = input("Enter Cover Letter file path: ").strip()
    candidate_texts[candidate_name] = {
        "job_desc": extract_text(jd_path),
        "resume": extract_text(resume_path),
        "cover_letter": extract_text(cover_letter_path)
    }
    print(f"Uploaded documents for {candidate_name}.\n")
screening_template = """
You are an expert HR assistant.
Evaluate the candidate's fit based on the Job Description and Resume.
Job Description:
{job_desc}
Resume:
{resume}
Give a score out of 100 and a short explanation.
"""

sentiment_template = """
Analyze the sentiment of the following cover letter:
{cover_letter}
Is it positive, neutral, or negative? Explain briefly.
"""

comparison_template = """
You are a hiring expert.
Compare the two candidate evaluations.
Previous Candidate:
{previous_candidate}
Current Candidate:
{current_candidate}
Who is the better fit and why?
"""
screening_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(screening_template))
sentiment_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(sentiment_template))
comparison_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(comparison_template))
def screening_tool(inputs):
    name = inputs.get("candidate_name")
    data = candidate_texts.get(name)
    if not data:
        return f"No data for candidate {name}."
    return screening_chain.run(job_desc=data["job_desc"], resume=data["resume"])
def sentiment_tool(inputs):
    name = inputs.get("candidate_name")
    data = candidate_texts.get(name)
    if not data:
        return f"No data for candidate {name}."
    return sentiment_chain.run(cover_letter=data["cover_letter"])
def comparison_tool(inputs):
    name1 = inputs.get("candidate1")
    name2 = inputs.get("candidate2")
    data1 = candidate_texts.get(name1)
    data2 = candidate_texts.get(name2)
    if not data1 or not data2:
        return f"Missing data for one or both candidates."
    eval1 = screening_chain.run(job_desc=data1["job_desc"], resume=data1["resume"]) + "\n" + sentiment_chain.run(cover_letter=data1["cover_letter"])
    eval2 = screening_chain.run(job_desc=data2["job_desc"], resume=data2["resume"]) + "\n" + sentiment_chain.run(cover_letter=data2["cover_letter"])
    return comparison_chain.run(previous_candidate=eval1, current_candidate=eval2)
tools = [
    Tool(name="Candidate Screening", func=screening_tool, description="Evaluate a candidate based on JD and Resume"),
    Tool(name="Cover Letter Sentiment", func=sentiment_tool, description="Check sentiment of a candidate's cover letter"),
    Tool(name="Candidate Comparison", func=comparison_tool, description="Compare two candidates")
]
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
def main():
    while True:
        action = input("Type 'upload' to add a candidate, 'ask' to ask a question, or 'exit' to quit: ").strip().lower()
        if action == "upload":
            name = input("Enter candidate name or ID: ").strip()
            upload_candidate(name)
        elif action == "ask":
            question = input("Ask a question (e.g., 'Evaluate candidate John'): ")
            response = agent.run(question)
            print("\nAgent Response:\n", response)
        elif action == "exit":
            print("Goodbye!")
            break
        else:
            print("Invalid command. Try again.")
if __name__ == "__main__":
    main()
