"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
COURSE_PATH = "documents/course_corsica.md"

def ingest_docs():
    """Convert markdown to embeddings and store them."""
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    with open(COURSE_PATH) as f:
        markdown_text = f.read()
        docs = markdown_splitter.create_documents([markdown_text])
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(markdown_text)
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
