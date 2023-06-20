import os
import getpass
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import ( RecursiveCharacterTextSplitter, CharacterTextSplitter)
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


embeddings = OpenAIEmbeddings()

dataset_path = "hub://rgala" + "/data"

db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False
)

# What was the restaurant the group was talking about called?
query = input("Enter query:")

# The Hungry Lobster
ans = qa({"query": query})

print(ans)