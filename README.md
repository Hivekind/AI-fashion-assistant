## Description
We'll build an **AI fashion assistant**, using a fashion shop dataset from HuggingFace for indexing, and set up a RAG chain to process user queries and generate responses.

### Retrieval-Augmented Generation (RAG)
RAG a technique that enhances the knowledge of language models by integrating additional data. A RAG application has two main components:

#### 1. Indexing:
Ingest and index data from a specific source, typically done offline.

#### 2. Retrieval and Generation:
During runtime, process the user's query, retrieve relevant data, and generate a response.

The data used in this project is the Fashion Shop Dataset from HuggingFace. The data consists of questions and answers related to fashion. The data is stored in MongoDB Atlas and the embeddings are generated using LangChain. The user query is passed to LangChain to generate an embedding and then the most similar data is retrieved from MongoDB Atlas.

- Project: Shopping Assistant
- Dataset: [Fashion Shop Dataset](https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant) from HuggingFace


## Prerequisites
* [MongoDB Atlas Subscription](https://cloud.mongodb.com/) (Free Tier is fine)
* Open AI [API key](https://platform.openai.com/account/api-keys)
* LangChain [API key](https://docs.smith.langchain.com/)


## Quick Start Steps
- Setup env var for OpenAI API key and MongoDB connection string
```zsh
export OPENAI_API_KEY=<your-api-key>
export MONGODB_CONN_STRING=<your-conn-string>
```

- Setup env var for LangChain API key and tracing, required for: `hub.pull(...)` in the script
```zsh
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
```

- Create a new Python environment
```zsh
python3 -m venv env
```

- Activate the new Python environment
```zsh
source env/bin/activate
```

- Install the requirements
```zsh
pip3 install -r requirements.txt
```

### 1. Indexing:

Embedding is generated from the combination of `question` + `answer` fields. This is based on the assumption that we want to match user query against both the question and answer fields in the database. The embedding is stored in MongoDB and index is created for vector search.

#### Load, Transform, Embed and Store

Run the below script:

```zsh
python3 ingest.py
```

### 2. Retrieval and Generation:

Run the below script, by passing your prompt as an argument.

```zsh
python3 query.py  -q "What is the store policy for returns?"
```


## Google Colab Notebook
You can also run the code in a Google Colab notebook. The notebook is available [here](RAG_fashion_langchain_openai_mongodb.ipynb).



## Resources
* [MongoDB Atlas](https://cloud.mongodb.com/)
* [Open AI API key](https://platform.openai.com/account/api-keys)
* [LangChain Doc](https://python.langchain.com)
* [LangChain API key](https://docs.smith.langchain.com/)
* [MongoDB Atlas module](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas)  
* [Open AI module](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/openai)