from kss import split_sentences
import json
from tqdm import tqdm

train_file_path = "train_summary.json"
train_summary_file_path = "train_summary_broadcast.json"
test_file_path = "test_summary.json"

with open(train_file_path, "r", encoding="utf-8") as f:
    train_files = json.load(f)
    train_error_idx = []
    idx = 0
    for file in tqdm(train_files):
        text = file["original"]
        try:
            text_splited = split_sentences(text)
            file["original_splited"] = text_splited
        except:
            train_error_idx.append(idx)
        idx += 1            
with open("train_summary_splited.json", "w", encoding="utf-8") as f:
    json.dump(train_files, f)
print(f"train_error_idx : {train_error_idx}")
    
with open(test_file_path, "r", encoding="utf-8") as f:
    test_files = json.load(f)
    test_error_idx = []
    idx = 0
    for file in tqdm(test_files):
        text = file["original"]
        try:
            text_splited = split_sentences(text)
            file["original_splited"] = text_spliteds
        except:
            test_error_idx.append(idx)
        idx += 1
with open("test_summary_splited.json", "w", encoding="utf-8") as f:
    json.dump(test_files, f)
print(f"test_error_idx : {test_error_idx}")