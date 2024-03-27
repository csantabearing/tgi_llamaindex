from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from fastapi import FastAPI
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
import os.path

#Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
#Settings.llm = Ollama(model="mistral", request_timeout=30.0)
# check if storage already exists
PERSIST_DIR = "storage"
if not os.path.exists(f'{PERSIST_DIR}/docstore.json'):
    # load the documents and create the index
    documents = SimpleDirectoryReader("rag_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

app = FastAPI()


@app.get("/query")
async def root(question:str):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response