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

with open("b99_episode.txt") as f:
	episode = f.read()

text_splitter = CharacterTextSplitter(chunk_size =1000, chunk_overlap=0)
pages = text_splitter.split_text(episode)

text_splitter = CharacterTextSplitter(chunk_size =1000, chunk_overlap=100)
texts = text_splitter.create_documents(pages)

#print(texts)
print("Created Texts!")


db = DeepLake.from_documents(texts, embeddings, dataset_path = dataset_path, overwrite=True)