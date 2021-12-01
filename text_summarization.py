import torch
import json
import numpy as np
import random
import torch.nn as nn

from kss import split_sentences
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed
from sklearn.model_selection import train_test_split

file_train = "train_summary_splited.json"
file_train_broad = "train_summary_broadcast.json"
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
seed = 2021

def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
seed_everything(seed)

with open(file_train, "r", encoding="utf-8") as f:
    train_file1 = json.load(f)
# with open(file_train_broad, "r", encoding="utf-8") as f:
#     train_file2 = json.load(f)
# train_file = train_file1 + train_file2
train_file = train_file1
train_file = [file for file in train_file if "original_splited" in file.keys()]

train_set, val_set = train_test_split(train_file, test_size=0.1, random_state=seed)
max_labels_num = 60

class SummaryDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.dataset = file
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]["original"]
        y = self.dataset[idx]["summary"]

        #나뉘어진 문장
        x_splited = self.dataset[idx]["original_splited"]

        #각 문장마다 [CLS], [SEP] 붙이기
        x_splited_with_special = [f"[CLS] {x_splited[i]} [SEP]" if i != 0 else f"{x_splited[i]} [SEP]" for i in range(len(x_splited)-1)]
        x_splited_with_special.append(f"[CLS] {x_splited[len(x_splited)-1]}") # last sentence
        
        x_splited += ["" for _ in range(60-len(x_splited))]
        
        # 각 문장에 [CLS], [SEP] 붙인것들 join 하기
        x_with_special = " ".join(x_splited_with_special)

        # 문장들의 정답 라벨 구하기 -> [1, 0, 0, 1] 같은 형식
        labels = [1 if sentence in y else 0 for sentence in x_splited]
        labels += [-1 for _ in range(60-len(labels))]

        # 나뉘어진 문장을 tokenizer에 넣은 후 CLS토큰의 index찾아서 담아주기 → 여기 예시에서는 CLS가 4개 -> 위치 index도 담아야함
        # res = self.tokenizer(x_with_special, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        res = self.tokenizer(x_with_special, padding="max_length", truncation=True, return_tensors="pt")
        cls_idx = [i for i in range(res["input_ids"][0].size()[0]) if res["input_ids"][0][i] == self.tokenizer.convert_tokens_to_ids("[CLS]")]
        cls_idx += [-1 for _ in range(60-len(cls_idx))]
        
        #segment embedding 구하기 -> 홀수번째 문장은 0, 짝수번째 문장은 1
        seg_embed = []
        cls_cnt = 0
        for ids in res["input_ids"][0]:
            if ids == self.tokenizer.convert_tokens_to_ids("[CLS]"):
                cls_cnt += 1
            if cls_cnt % 2 == 0:
                seg_embed.append(1)
            else:
                seg_embed.append(0)
        # print(f"token_type_ids: {res['token_type_ids'].size()}")
        # print(f"input_ids : {res['input_ids'].size()}")
        # print(f"attention : {res['attention_mask'].size()}")
        # dict형태로 return
        res_dict = {"original_splited" : x_splited, 
                    "input_ids": res["input_ids"],
                    "token_type_ids": torch.tensor(seg_embed).unsqueeze(0),
                    # "token_type_ids": res["token_type_ids"],
                    "attention_mask": res["attention_mask"],
                    "cls_idx": torch.tensor(cls_idx).unsqueeze(0),
                    "labels": torch.tensor(labels).unsqueeze(0)}
        # return res_dict["input_ids"], res_dict["attention_mask"], res_dict["token_type_ids"], res_dict["original_splited"], res_dict["cls_idx"], res_dict["labels"]
        return res_dict

train_dataset = SummaryDataset(train_set, tokenizer)
val_dataset = SummaryDataset(val_set, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
class SummaryModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # self.config = AutoConfig.from_pretrained(model_name)
        # self.config.max_position_embeddings = 1026

        self.encoder = AutoModel.from_pretrained(model_name)
        # self.pos_emb = PositionalEncoding(dropout, self.config.hidden_size, max_len=self.condig.max_position_embeddings)

    def forward(self, input_dict):
        print(input_dict["token_type_ids"].size())
        output = self.encoder(input_ids=input_dict["input_ids"].squeeze(), 
                            attention_mask=input_dict["attention_mask"].squeeze(),
                            token_type_ids=input_dict["token_type_ids"].squeeze())
        print(output)

model = SummaryModel(model_name)
for batch in train_dataloader:
    model(batch)
    break
# model(train_dataset[1])