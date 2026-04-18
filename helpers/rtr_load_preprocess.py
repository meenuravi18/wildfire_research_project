import json
import pandas as pd

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    flattened_data = []
    for entry in data:
        for qa_pair in entry["data"]:
            flattened_data.append({
                "fileType": entry["fileType"],
                "source": entry["source"],
                "date": entry["date"],
                "geo_scope": entry.get("geo_scope", "United States"), 
                "geo_level": entry.get("geo_level", 0),                
                "passage": qa_pair["passage"],
                "answer": qa_pair["answer"],
            })
    
    df_dataset = pd.DataFrame(flattened_data)
    
    return df_dataset


def preprocess_data(df):
    df = df.dropna(subset=["question", "passage"])
    df = df.drop_duplicates(subset=["question", "passage"], keep="first").reset_index(drop=True)
    
    grouped_data = df.groupby("question").agg({
        "passage": list,
        "answer": lambda x: x.iloc[0],
        "fileType": list,
        "source": list,
        "date": list,
        "geo_scope": list,   
        "geo_level": list,   
    }).reset_index()
    
    print("Data is clean, grouped by question.")
    return grouped_data