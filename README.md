# ✍🏻 ParollaGPT 

Learn Corsican language using Large Language Model through a discussion with a personalized assistant.

## Philosophy

We want to propose a 1 to 1 tutoring dialogue that will be based on large corpus of all kinds of books about the given language, how to conjugate verbs and more.
We are planning to add more languages, but we started with Corsican language since it's the repository author country. We are targetting languages that have bad coverage in LLM datasets.


## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
1. Run `ingest.sh` to ingest LangChain docs data into the vectorstore (only needs to be done once).
   1. You can use other [Document Loaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html) to load your own data into the vectorstore.
1. Run the app: `make start`
   1. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`. You can find more documentation [here](https://langchain.readthedocs.io/en/latest/tracing.html).
1. Open [localhost:9000](http://localhost:9000) in your browser.

## 🚀 Important Links

Deployed version (to be updated soon):


## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site
2. Load html with LangChain's [ReadTheDocs Loader](https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/readthedocs_documentation.html)
3. Split documents with LangChain's [TextSplitter](https://langchain.readthedocs.io/en/latest/reference/modules/text_splitter.html)
4. Create a vectorstore of embeddings, using LangChain's [vectorstore wrapper](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html) (with OpenAI's embeddings and FAISS vectorstore).

Question-Answering has the following steps, all handled by [ChatVectorDBChain](https://langchain.readthedocs.io/en/latest/modules/indexes/chain_examples/chat_vector_db.html):

1. Given the chat history and new student input, determine student missing knowledge and propose an exercise to fill the knowledge gap.
2. Given that exercise proposition, look up relevant documents from the vectorstore.
3. Pass the student input and relevant documents to a LLM to generate a final answer.
