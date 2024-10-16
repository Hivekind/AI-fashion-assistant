from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
import warnings
import argparse
import os
import params


mongodb_conn_string = os.getenv("MONGODB_CONN_STRING")

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Fashion Shop Assistant')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

query = args.question

print("\nYour question:")
print("-------------")
print(query)

# Connect to MongoDB Atlas
client = MongoClient(mongodb_conn_string)
db = client[params.db_name]
collection = db[params.collection_name]

ai_model = params.ai_model
vector_dimension = params.vector_dimension
index_name = params.index_name

# openAI embedding model
embeddings = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)

# Initialize MongoDBAtlasVectorSearch with correct keys
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,  # Your embedding model
    text_key="text",  # Field in MongoDB for the text you want to retrieve
    embedding_key="embedding",  # Field in MongoDB for the stored embeddings
    index_name=index_name,  # Name of Vector Index in MongoDB Atlas
    relevance_score_fn="cosine"  # Use cosine similarity
)

print(f"User question: {query}\n")


################################
# get relevant docs from MongoDB
################################


def get_similarity_search():
  # Perform the similarity search
  similar_docs = vectorStore.similarity_search(query=query, include_scores=True)

  print("\nQuery Response:")
  print("---------------")

  # Access the closest matching document
  if similar_docs:
      # Iterate through each document and print its content
      for i, doc in enumerate(similar_docs):
          print(f"Doc {i+1}: {doc.page_content}\n\n")
          print(doc.metadata["score"], end="\n\n\n")

      closest_match = similar_docs[0]
      # print("Closest Match:", closest_match)
  else:
      print("No matching document found.")



###################
# Set up RAG chain
###################

llm = ChatOpenAI(model="gpt-4o-mini")

search_kwargs = {
    "include_scores": True,
}

retriever = vectorStore.as_retriever(search_kwargs=search_kwargs)

# prompt = hub.pull("rlm/rag-prompt")


def get_prompt():
    # Define the template as a string
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 
    Context: {context} 
    Answer:
    """

    # Create a PromptTemplate object with input variables
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Wrap it in a HumanMessagePromptTemplate
    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    return chat_prompt

prompt = get_prompt()



def format_docs(docs):
    print("\nRetriver:")
    print("---------------")

    for i, doc in enumerate(docs):
        print(f"Doc {i+1}: {doc.page_content}\n\n")
        print(doc.metadata, end="\n\n\n")

    return "\n\n".join(
      [f"{doc.page_content}" for doc in docs]
    )


# def format_docs(docs):
#     print("\nRetriever - prepare context for prompt:")
#     print("--------------------------------------")

#     # Check if docs is empty
#     if len(docs) <= 0:
#         print("no matches found")
#         return ""

#     # If the first document has a score greater than 0.8, return its content
#     if docs[0].metadata.get("score", 0) > 0.8:
#         text = docs[0].page_content
#     else:
#         # Otherwise, concatenate all results
#         text = "\n\n".join([doc.page_content for doc in docs])

#     print(text)
#     return text




rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke(query)

print("\nRAG Chain Response:")
print("-------------------")
print(response)

# Close the MongoDB connection
client.close()

