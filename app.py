import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # import the text splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

#Save the Variables
if "docs" not in st.session_state:
    st.session_state["docs"] = ""

#Save the data into database
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

#API Key
if "apikey" not in st.session_state:
    st.session_state["apikey"] = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#preprocess text
def preprocess_text(docs):
    # Split the text into sentences
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 200) # chunk_size is the size of each chunk, chunk_overlap is the overlap between chunks
    all_split= text_splitter.split_documents(docs) # split the text into chunks

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state["apikey"])

    # Create the FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(documents=all_split, embedding=embeddings)
    # Convert vectorstore to a retriever with the correct search_type argument
    st.session_state["retriever"] = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # "k" is the number of similar documents to retrieve

with st.sidebar:
    st.session_state["apikey"] = st.text_input("GoogleGenAI", key="chatbot_api_key", type="password")

    wikipedia_search = st.text_input("Search Wikipedia Knowledge")
    search_button = st.button("Search")

    if wikipedia_search or search_button:
        with st.spinner("On Process"):
            st.session_state["docs"] = WikipediaLoader(query=wikipedia_search,  load_max_docs=2, lang='id').load()
        st.success("Data has been loaded", icon="✅") 
        st.subheader("Summary", divider=True)
        st.write(st.session_state["docs"][0].metadata['summary'])
    
        with st.spinner("Processing Data"):
            preprocess_text(st.session_state["docs"])
            st.success("Retriever & Vector Store Created", icon="✅")

   
st.title("💬 Wikichat: Chatbot empowered by Wikipedia")
st.divider()

# Function Formating Answer
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-pro",  google_api_key=st.session_state["apikey"])

    prompt_template = PromptTemplate.from_template("""
        Answer the question as detailed as possible from the provided context, make sure to provide all the details\n
        Ulangi pertanyaan kepada jawaban
        Context : {context}\n
        Pertanyaan : {question}
    """)

    rag_chain = (
        {"context": st.session_state["retriever"] | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    msg = rag_chain.invoke(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

  