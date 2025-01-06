import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document

st.set_page_config(page_title="Chat with PDF", layout="wide", page_icon="💬")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("bookshelf_background.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Courier New', Courier, monospace;
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.9);
    font-family: 'Courier New', Courier, monospace;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    st.success("💡 Reply: " + response["output_text"])

def summarize_text(raw_text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = """
    Summarize the following text as concisely as possible, while retaining the main ideas:\n\n
    {context}\n\n
    Summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    document = Document(page_content=raw_text)
    response = chain(
        {"input_documents": [document], "context": raw_text}
    )
    return response["output_text"]

def main():
    st.title("💬 Chat with Your PDF")
    st.markdown("---")

    st.sidebar.markdown("### 📂 Upload Your PDFs")
    pdf_docs = st.sidebar.file_uploader(
        "Drag & Drop or Select PDF Files",
        accept_multiple_files=True,
        type=["pdf"],
    )
    st.sidebar.markdown("---")
    
    user_question = st.text_input("🔎 Ask a Question from the PDF Files")
    if user_question:
        placeholder=st.empty()
        placeholder.write("Querying...")
        user_input(user_question)
        placeholder.empty()

    if st.sidebar.button("📝 Summarize PDF"):
        if pdf_docs:
            st.write("Summarizing your files...")
            raw_text = get_pdf_text(pdf_docs)
            summary = summarize_text(raw_text)
            st.info("📃 Summary: " + summary)
        else:
            st.error("🚨 Please upload PDF files before summarizing.")

    if st.sidebar.button("⚙ Submit & Process"):
        if pdf_docs:
            st.write("Processing your files...")
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ Files processed successfully!")
        else:
            st.error("🚨 Please upload PDF files before processing.")

    st.markdown("---")
    st.markdown(
        "👨‍💻 Developed by Guna Sekhar | [GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/) "
    )

if __name__ == "__main__":
    main()
