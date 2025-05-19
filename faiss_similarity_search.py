import os 
import numpy as np 
import pandas as pd  
from langchain_huggingface import HuggingFaceEmbeddings
import faiss 
import pickle 

pandas_code_store_df = pd.read_excel("./pandas_code_store.xlsx", sheet_name="Sheet1")
pandas_length_stored_df = len(pandas_code_store_df) 
comparative_faiss_index_path = "questions.index" 
comparative_faiss_metadata_path = "metadata.pkl" 

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_existing_comparative_pairs_to_vectorstore():
    if os.path.exists(comparative_faiss_index_path):
        try: 
            os.remove(comparative_faiss_index_path)
            print("Old Comparative questions FAISS Index deleted successfully.")
        except PermissionError:
            print("Error: Comparative questions FAISS index file is in use. Close other processes using it.")
    if os.path.exists(comparative_faiss_metadata_path):
        try:
            os.remove(comparative_faiss_metadata_path)
            print("Old Comparative questions FAISS metadata deleted successfully.")

        except PermissionError:
            print("Error: Comparative questions FAISS metadata file is in use. Close other processes using it.") 
    print("Creating new comparative FAISS index.")
    
    #preparing data for Comparative Faiss store
    questions = []
    pandas_code = []
    question_type = []
    embeddings = [] 

    for idx,row in pandas_code_store_df.iterrows():
        question = row["Question"]
        code = row["Code"]
                
        questions.append(question)
        pandas_code.append(code)
        embedding = embed_model.embed_query(question)
        embeddings.append(embedding) 
    
     # Convert to numpy array for FAISS
    embeddings_array = np.array(embeddings).astype("float32")

    # Create and save FAISS index
    dimension = embeddings_array.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity
    # Normalize vectors to ensure inner product is equivalent to cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)

    # Save the index
    faiss.write_index(index, comparative_faiss_index_path) 
    
    metadata = {"questions": questions, "pandas_code": pandas_code, "question_type": question_type}
    
    with open(comparative_faiss_metadata_path, "wb") as f: 
        pickle.dump(metadata, f) 
    
    print("Data successfully loaded into the new comparative FAISS index.")


load_existing_comparative_pairs_to_vectorstore()

def search_similar_questions(query, similarity_threshold=0.8):
    # Load the FAISS index and metadata
    index = faiss.read_index(comparative_faiss_index_path)

    with open(comparative_faiss_metadata_path, "rb") as f:
        metadata = pickle.load(f)

    questions = metadata["questions"]
    pandas_code = metadata["pandas_code"] 

    # First check for exact match
    normalized_query = query.lower().strip()
    for i, question in enumerate(questions):
        if question.lower().strip() == normalized_query:
            return [
                {
                    "question": question,
                    "pandas_code": pandas_code[i],
                    "similarity": 1.0,
                    "exact_match": True,
                }
            ]

    # If no exact match, proceed with similarity search
    # Generate embedding for the query
    query_embedding = np.array([embed_model.embed_query(query)]).astype("float32")

    # Normalize the query vector for cosine similarity calculation
    faiss.normalize_L2(query_embedding)

    # Search in FAISS
    k = pandas_length_stored_df  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    # Convert distances to similarities (FAISS returns distances, we need similarity)
    # For normalized vectors with inner product, the range is [-1, 1], so we adjust it
    similarities = distances[
        0
    ]  # distances are already similarities for normalized vectors with inner product

    # Filter by similarity threshold and create results
    filtered_results = []
    for i, similarity in enumerate(similarities):
        if similarity >= similarity_threshold:
            idx = indices[0][i]
            filtered_results.append(
                {
                    "question": questions[idx],
                    "pandas_code": pandas_code[idx],
                    "similarity": float(similarity),
                    "exact_match": False,
                }
            )

    filtered_results.sort(key=lambda x: x["similarity"], reverse=True)

    return filtered_results

def get_few_shot_examples(user_query, num_examples=2):
    similar_questions = search_similar_questions(user_query, similarity_threshold=0.8)

    if similar_questions and similar_questions[0].get("exact_match", False):
        # If exact match, just return that one
        exact_match = similar_questions[0]
        return (
            f"Using this exact or similar Code:\nCode: {exact_match['pandas_code']}",
            True,
        )

    # Filter and format only if question_type is 'YoY' or 'Seq'
    few_shot_examples = []
    for i, item in enumerate(similar_questions):
        if item.get("question_type") in ("YoY", "Seq"):
            example = (
                f"\n**Example {len(few_shot_examples) + 1}**:\n"
                f"Question: {item['question']}\n"
                f"Python/Pandas Code: \n{item['pandas_code']}\n"
            )
            few_shot_examples.append(example)
            if len(few_shot_examples) == num_examples:
                break

    return ("\n".join(few_shot_examples), False) if few_shot_examples else ("", False) 