from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from enhancer import reframe_question
import logging
import os


load_dotenv()

#loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING) 


API_KEY = os.environ["GROQ_API_KEY"]

agent1 = ChatGroq(
    model= "llama-3.3-70b-versatile",
    api_key= API_KEY
)

agent2 = ChatGroq(
    model= "llama-3.3-70b-versatile",
    api_key= API_KEY
)


agenda = [
"Greetings", 
"Ask for the introduction : ",
"Ask questions on Projects / Experience ",
"Ask the contribution in the project ",
"Ask the questions about the tech stack and why not some similar tech stack used" ,
"Give them an approach for solving the problem. And ask them why they didn’t used this approach." ,
"Why are they interested and applying for this role and organization?",
"What do you think about this tech?" ,
"Deep questions about the responses" ,
"Exemplary Questions on the field of study / Experience" ,
"Generate hypothetical conditions and ask the responsibilities" ,
"How will you contribute to the society."
]

prompt1 = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an interviewer. The interviewee will approach you, you have to generate conversations and questions based on 
    the question parameter {q_par}. You will be receiving the role {role} of the person. Also the previous questions param you
    asked the person {preq} and the previous responses {presp}.
    The current response of the user to you {resp}.
    Return only the question and be concise.
    """)
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", """You are an interviewer. Another interviewer will ask the interviewee a question, 
                    you have to generate conversations and questions of very depth and specific to the field.
                    The generated question should be dependent on the role {role} of the person. 
                    The question asked by the other interviewer {question} and the response of the interviewee {response} on the 
                    following questioning parameter {param}.
                    Return only the question and be concise.
""")
])

def conversation_agent1():
    i = 0
    presp = []
    resp = input("Initiate the conversation : ")
    while True: 
        if i == 0: 
            role = input("Enter the role you are applying for : ")
            agent1_response = agent1.invoke(prompt1.format(q_par = agenda[i], role = role, preq = "", presp = "", resp = resp ))
            agent3_response = reframe_question(agent1_response.content)
            print(agent3_response)
            presp.append(resp)
            i += 1
            
        resp = input("Enter your response : ")
        agent1_response = agent1.invoke(prompt1.format(q_par = agenda[i], role = role, preq = agenda[i-1], presp = presp[i-1], resp = resp ))
        agent3_response = reframe_question(agent1_response.content)
        print(agent3_response)
        presp.append(resp)
        conversation_agent2(role,agent1_response,resp,agenda[i])
        i += 1

def conversation_agent2(role, ques, resp, para):
    agent2_response = agent2.invoke(prompt2.format(role = role, question = ques, response = resp, param = para))
    agent3_response = reframe_question(agent2_response.content)
    print(agent3_response)
    
    
conversation_agent1()
