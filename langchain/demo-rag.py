import asyncio
from langchain_core.documents import Document
from torch import embedding


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

print(documents)


from langchain_community.document_loaders import PyPDFLoader
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "example_data", "nke-10k-2023.pdf")
loader = PyPDFLoader(file_path)

docs = loader.load()



from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


from langchain_openai import OpenAIEmbeddings
from config import ALI_BL_API_KEY, ALI_BL_BASE_URL
from typing import cast
from pydantic import SecretStr

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-v4", 
#     api_key=cast(SecretStr, ALI_BL_API_KEY),
#     base_url=ALI_BL_BASE_URL)

from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("ALI_BL_API_KEY"))

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])


from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

print("==============================================================")
#vector_store.asimilarity_search("When was Nike incorporated?")
result = asyncio.run(vector_store.asimilarity_search("When was Nike incorporated?"))
print(result[0])


# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)


embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
