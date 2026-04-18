import pandas as pd
from tqdm import tqdm

def retrieve(retriever, df, hierarchy_pipeline=None):
    questions = df["question"].tolist()
    retrieved_docs = []
    
    for i, question in enumerate(questions):
        d_results = retriever.retrieve(question)
        
        # Detect query's geo level
        query_geo = hierarchy_pipeline(question) if hierarchy_pipeline else {}
        query_place = query_geo.get("detected_place", "United States")
        query_level = infer_level(query_place)
        
        retrieved_row = {
            "question": question,
            "query_geo_scope": query_place,          
            "query_geo_level": query_level,         
            "retrieved_passages": [doc.text for doc in d_results],
            "retrieved_geo_scopes": [doc.metadata.get("geo_scope", "United States") for doc in d_results], 
            "retrieved_geo_levels": [doc.metadata.get("geo_level", 0) for doc in d_results],               
            "expected_passage": df.iloc[i]["passage"],
            "expected_answer": df.iloc[i]["answer"],
        }
        
        retrieved_docs.append(retrieved_row)
    
    return pd.DataFrame(retrieved_docs)

def generate(query_engine, df):
    responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        prompt = row["prompt"]
        d_response = query_engine.query(prompt)
        row_dict = row.to_dict()
        row_dict["generated_response"] = str(d_response)
        
        responses.append(row_dict)
    
    results_df = pd.DataFrame(responses)
    
    return results_df