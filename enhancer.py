import os
import warnings
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

warnings.filterwarnings("ignore")

def reframe_question(question: str) -> str:
    """
    Takes a question as a parameter and reframes it 
    using the contents of 'resume- Deepesh Jha.pdf' that
    has been embedded into a vectorstore.
    """
    load_dotenv()
    OPEN_AI_API_KEY = os.environ["OPENAI_API_KEY"]
    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    pdf_path = "resume- Deepesh Jha.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    embedding_model = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    system_template = (
        "You are an interviewer. You are sitting in an interview; "
        "another interviewer will pass you a question. You must use the "
        "user's resume (embedded in your memory as {context}) to enhance/rewrite "
        "the question. Then return only the final question and be concise , keep it short."
    )
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = (
        "Here is the question:\n{question}\n\n"
        "Please rewrite/enhance it using the context above."
    )
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": chat_prompt,
            "document_variable_name": "context"
        }
    )
    reframed_question = qa_chain.run(question)
    return reframed_question


# if __name__ == "__main__":
