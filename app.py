import os
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from zipfile import ZipFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import openai, langchain, pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message
from langchain.chains import RetrievalQA

# ========================= Global settings
# Basic page settings
st.set_page_config(page_title="Q/A with your file")
st.header("LangChain Chatbot - Document Processing and Retrieval")

# Global variables and load embedding model
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Check if any of the specified environment variables is empty
if any(os.getenv(variable, "") == "" 
    for variable in ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_ENVIRONMENT", "OPENAI_API_KEY"]
    ):
    st.error("Please add API Keys in .env file first")
    st.stop()

# Models and VectorDB
with st.spinner('Getting things ready for you...'):
    pinecone.init(apikey = PINECONE_API_KEY,
              envirnoment = PINECONE_ENVIRONMENT)

    # FIXME: When ever I specify the model it give the error that it is applicable in version openai<1.0
    # details mentioned in the document
    # llm = OpenAI(model_name="gpt-3.5-turbo-16k",
    #                 temperature = 0, 
    #                 openai_api_key = OPENAI_API_KEY)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
# ========================= Functions
def save_uploaded_file(uploaded_file, temp_file_path):
    """
    Save the content of the uploaded file to a temporary file.

    Parameters:
    - uploaded_file: The uploaded file object.
    - temp_file_path: The path to the temporary file to save.

    Returns:
    - file name/path
    """
    # This condition is for the case of ZIP file 
    if isinstance(uploaded_file, str):
        # If the uploaded_file is a string, it's the path to an existing file
        return uploaded_file
    
    # Read file content
    file_content = uploaded_file.read()
    # Save the content to a temporary file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    return temp_file_path

def load_pdf(uploaded_file):
    """
    This function will take a pdf file as a parameter and return content of that file as document
    """
    temp_file_path = save_uploaded_file(uploaded_file, "temp_file.pdf")
    # Use PyMuPDFLoader to load the temporary file
    loader = PyMuPDFLoader(temp_file_path)
    data = loader.load()
    # Clean up: Remove the temporary file
    os.remove(temp_file_path)
    return data

def load_docx(uploaded_file):
    """
    This function will take a word file(.docx) as a parameter and return content of that file
    """
    temp_file_path = save_uploaded_file(uploaded_file, "temp_file.docx")
    # Use UnstructuredWordDocumentLoader to load the temporary file
    loader = UnstructuredWordDocumentLoader(temp_file_path)

    data = loader.load()
    # Clean up: Remove the temporary file
    os.remove(temp_file_path)

    return data

def load_zip(uploaded_file):
    """
    This function will take a ZIP file and return content of the files present in it
    """
    with ZipFile(uploaded_file, 'r') as zip_file:
        # Extract all files from the ZIP archive
        extracted_files = [zip_file.extract(name) for name in zip_file.namelist()]

    # Initialize an empty list to store the combined documents
    combined_text = []
    # Loop through extracted files and load text from supported types (PDF and DOCX)
    for extracted_file in extracted_files:
        if extracted_file.lower().endswith(".pdf"):
            combined_text += load_pdf(extracted_file)
        elif extracted_file.lower().endswith(".docx"):
            combined_text += load_docx(extracted_file)

    return combined_text

def get_text_chunks(file_text):
    # spilit ito chuncks
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500,
    length_function=len
    )
    docs = text_splitter.split_documents(file_text)
    return docs

def  vector_store_pinecone(document_chunks):
    # Use huggingface embedding model declare on the top
    try:
        vectore_db = Pinecone.from_texts([page.page_content for page in document_chunks],
                                        embeddings,
                                        index_name = PINECONE_INDEX_NAME
                                        )
    except Exception as e:
        st.write("Something went wrong! please try again")
    return vectore_db

def get_qa_chain(retriever):
    qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI() , chain_type="stuff", retriever=retriever
            )
    return qa_chain
    
def handel_user_query(user_question):
    with st.spinner('Generating response...'):
        result = st.session_state.conversation({"query": user_question})
        response = result['result']
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{response}")

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))

def main():
    # Session variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "process_complete" not in st.session_state:
        st.session_state.process_complete = None

    # Take file from the user
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["pdf", "docx", "zip"])
    process_button = st.sidebar.button("Process")
    document_uploaded = False

    # File processing
    if process_button and uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = load_pdf(uploaded_file)
            document_uploaded = True
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = load_docx(uploaded_file)
            document_uploaded = True
        elif uploaded_file.type == "application/x-zip-compressed":
            text = load_zip(uploaded_file)
            document_uploaded = True
        else:
            st.error("Unsupported file type")
            st.stop()
    else:
        st.sidebar.info("Please! Upload a file first and then press process button")
    
    # Check if user upload the document successfully
    if document_uploaded:
        st.info("Creating chunks...")
        text_chunks = get_text_chunks(text)
        
        st.info("Making Vector DB ready...")
        try:
            vector_db = vector_store_pinecone(text_chunks)
            retriever = vector_db.as_retriever(search_type="mmr")

            # create qa chain
            st.session_state.conversation = get_qa_chain(retriever) 
            st.session_state.process_complete = True
        except Exception as e:
            st.error(e)

    if st.session_state.process_complete:
        st.info("Now you can query from your uploaded document")
        user_question = st.chat_input("Ask Question from your document")
        if user_question:
            handel_user_query(user_question)

if __name__ == '__main__':
    main()