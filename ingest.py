from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from datasets import load_dataset
import pandas as pd
import tiktoken
import os
import params

mongodb_conn_string = os.getenv("MONGODB_CONN_STRING")

# Connect to MongoDB Atlas
client = MongoClient(mongodb_conn_string)
db = client[params.db_name]
collection = db[params.collection_name]
ai_model = params.ai_model
vector_dimension = params.vector_dimension


# Load dataset
dataset = load_dataset("Quangnguyen711/Fashion_Shop_Consultant", split="train")

# Convert dataset to Panda DataFrame
df = pd.DataFrame(dataset)

# Only keep records where the Question and Answer fields are not null
df = df[df["Question"].notna() & df["Answer"].notna()]

# Combine Question and Answer fields into a single text field
# axis=1: This means the function is applied row-wise
df["text"] = df.apply(lambda row: f"[Question]{row['Question']}[Answer]{row['Answer']}", axis=1)

# Convert the combined text column to a list of strings
texts = df["text"].tolist()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)

# Initialize the tokenizer for the specific model
tokenizer = tiktoken.encoding_for_model(ai_model)

# Initialize a variable to keep track of the total tokens used
total_tokens_used = 0

# Define a reasonable batch size
batch_size = 50

# Process the dataset in batches
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]

    # Calculate total tokens used for the current batch
    batch_tokens_used = sum(len(tokenizer.encode(text)) for text in batch_texts)
    total_tokens_used += batch_tokens_used

    # Generate embeddings for the current batch
    embeddings_list = embeddings.embed_documents(batch_texts)

    # Prepare documents with embeddings to insert into MongoDB
    documents = []
    for j, (index, row) in enumerate(df.iloc[i:i + batch_size].iterrows()):
        document = {
            "text" : row["text"],
            "embedding": embeddings_list[j]
        }

        # print(f"Item i: {i}, j: {j}\n")
        # print (document)

        documents.append(document)

    # Insert the batch of documents into MongoDB
    collection.insert_many(documents)

    # Print total tokens used in the current batch
    print(f"Processed and inserted batch {i // batch_size + 1}, tokens used : {batch_tokens_used}")

print(f"Embeddings generated and stored in MongoDB! Total tokens used: {total_tokens_used}")

# Close the MongoDB connection
client.close()
