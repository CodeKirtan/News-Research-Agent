import os
import streamlit as st
import re
import time
import uuid
import threading
from collections import defaultdict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone

# Load environment variables (for local testing)
load_dotenv()

# Streamlit Page Config 
st.set_page_config(
    page_title="News Research Agent",
    page_icon="📉",
    layout="wide",
)

# Session Isolation with Timestamp for 24-hour cleanup
if "namespace" not in st.session_state:
    st.session_state.namespace = f"{int(time.time())}_{uuid.uuid4()}"

# Secrets Management
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME"))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GROQ_API_KEY, HF_TOKEN, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("Please set GROQ_API_KEY, HUGGINGFACEHUB_API_TOKEN, PINECONE_API_KEY, and PINECONE_INDEX_NAME in your secrets or .env file.")
    st.stop()

# Set env vars for Langchain Pinecone client
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Styling
main_bg = "https://i.ibb.co/ccRBvfz7/Screenshot-2025-08-14-at-12-14-51-PM.png"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{main_bg}");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }}
    .stTextInput>div>div>input {{
        background-color: rgba(0,0,0,0.5);
        color: #ffffff;
    }}
    .stButton>button {{
        background-color: #333333;
        color: #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<h1 style="
    background: -webkit-linear-gradient(#ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align:center;
">
News Research Agent
</h1>
""", unsafe_allow_html=True)

# Caching LLM and Embeddings to improve latency
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=500
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

llm = get_llm()
embeddings = get_embeddings()

# Junk Filter Function
def is_junk_chunk(chunk_text):
    text_lower = chunk_text.lower()
    if len(chunk_text.split()) < 30: return True
    boilerplate_keywords = [
        "login", "sign-up", "subscribe", "advertisement", "cookie",
        "follow us", "download app", "price alerts", "fixed deposits",
        "credit score", "watchlist", "logout"
    ]
    if any(keyword in text_lower for keyword in boilerplate_keywords): return True
    non_alpha_ratio = sum(1 for c in chunk_text if not c.isalpha() and not c.isspace()) / len(chunk_text)
    if non_alpha_ratio > 0.4: return True
    return False

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

# Sidebar 
st.sidebar.title("News Articles URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url.strip():
        if is_valid_url(url.strip()):
            urls.append(url.strip())
        else:
            st.sidebar.error(f"Invalid URL format: {url}")

process_url_clicked = st.sidebar.button("Process URLs")
summarize_clicked = st.sidebar.button("Summarize Articles")

# URL Processing & Pinecone Upload
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
        st.stop()

    with st.spinner("Loading, processing, and uploading data to Pinecone..."):
        try:
            # Use WebBaseLoader to bypass SSL and NLTK model download issues
            loader = WebBaseLoader(web_paths=urls)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            raw_chunks = text_splitter.split_documents(data)

            docs = [chunk for chunk in raw_chunks if not is_junk_chunk(chunk.page_content)]
            if not docs:
                st.error("❌ No readable content was found in the provided URLs. This usually means the website blocked the scraper, or it only contains videos/images.")
                st.stop()
            st.info(f"✅ Filtered out {len(raw_chunks) - len(docs)} junk chunks. {len(docs)} clean chunks remain.")

            # Store in Pinecone
            
            # Initialize Pinecone client to check stats and cleanup
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            
            # --- 24-HOUR AUTOMATIC CLEANUP (BACKGROUND THREAD) ---
            # We run this in a background thread so it NEVER slows down the user's experience!
            def background_cleanup(idx):
                try:
                    stats = idx.describe_index_stats()
                    all_namespaces = stats.get('namespaces', {})
                    current_time = int(time.time())
                    for ns in all_namespaces.keys():
                        try:
                            ns_time = int(ns.split('_')[0])
                            if current_time - ns_time > 86400: # 86400 seconds = 24 hours
                                idx.delete(delete_all=True, namespace=ns)
                        except ValueError:
                            pass 
                except Exception as e:
                    print(f"Cleanup failed: {e}")
            
            threading.Thread(target=background_cleanup, args=(index,), daemon=True).start()
            # -----------------------------------------------------
            
            # Get initial vector count
            initial_stats = index.describe_index_stats()
            initial_count = initial_stats.get('total_vector_count', 0)
            
            # Upload documents
            PineconeVectorStore.from_documents(
                docs, 
                embeddings, 
                index_name=PINECONE_INDEX_NAME,
                namespace=st.session_state.namespace
            )
            
            # Poll until Pinecone finishes indexing (solving Eventual Consistency)
            expected_count = initial_count + len(docs)
            with st.spinner("Waiting for Pinecone cloud indexing to complete (this ensures instant answers)..."):
                while True:
                    time.sleep(3)
                    current_stats = index.describe_index_stats()
                    current_count = current_stats.get('total_vector_count', 0)
                    if current_count >= expected_count:
                        break

            st.success("✅ Data successfully processed and fully indexed in Pinecone!")
            st.session_state["data_processed"] = True
            st.session_state["raw_docs"] = docs
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# Initialize session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Summarize Feature
if summarize_clicked:
    if "raw_docs" not in st.session_state or not st.session_state.raw_docs:
        st.warning("Please process some URLs first before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            try:
                # Group chunks by their source URL to ensure every article is represented
                source_to_docs = defaultdict(list)
                for doc in st.session_state.raw_docs:
                    source = doc.metadata.get("source", "Unknown")
                    source_to_docs[source].append(doc)
                
                # Take an equal number of chunks from each source (max 15 chunks total)
                max_total_chunks = 15
                chunks_per_source = max(1, max_total_chunks // len(source_to_docs))
                
                selected_docs = []
                for source, docs in source_to_docs.items():
                    selected_docs.extend(docs[:chunks_per_source])
                
                # Format context explicitly with the source URL
                context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in selected_docs])
                
                prompt = f"Please provide a comprehensive summary of the following articles. Give an overview of the key points for EACH article.\n\nContext:\n{context}"
                response = llm.invoke(prompt)
                st.session_state.messages.append({"role": "assistant", "content": f"**Summary:**\n{response.content}"})
                st.rerun()
            except Exception as e:
                st.error(f"Error summarizing: {e}")

# Query Section 
if query := st.chat_input("Ask a question about the articles:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vectorstore = PineconeVectorStore(
                    index_name=PINECONE_INDEX_NAME, 
                    embedding=embeddings,
                    namespace=st.session_state.namespace
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                # Define a strict prompt template to constrain the LLM
                strict_prompt_template = """You are a specialized News Research Assistant. 
You must ONLY answer the user's question based on the following context retrieved from the provided news articles.
If the answer is not explicitly contained in the context, you must respond with: "I don't know based on the provided articles." Do not attempt to guess or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""
                STRICT_PROMPT = PromptTemplate(
                    template=strict_prompt_template, 
                    input_variables=["context", "question"]
                )
                
                # We use ConversationalRetrievalChain for memory context
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=st.session_state.memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": STRICT_PROMPT}
                )
                
                result = qa_chain({"question": query})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])
                
                # Formatting the answer
                st.markdown(answer)
                
                if source_docs:
                    with st.expander("View Sources"):
                        for idx, doc in enumerate(source_docs, start=1):
                            st.markdown(f"**Source {idx}:** {doc.metadata.get('source', 'Unknown')}")
                            st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
