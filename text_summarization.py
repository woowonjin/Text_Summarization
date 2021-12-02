import torch
import json
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kss import split_sentences
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed
from sklearn.model_selection import train_test_split
from ignite.metrics import RougeL

file_train = "train_summary_splited.json"
file_train_broad = "train_summary_broadcast.json"
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
seed = 2021
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
        y = self.dataset[idx]["summary"]
        
        #나뉘어진 문장
        x_splited = self.dataset[idx]["original_splited"][:]

        #각 문장마다 [CLS], [SEP] 붙이기
        x_splited_with_special = [f"[CLS] {x_splited[i]} [SEP]" if i != 0 else f"{x_splited[i]} [SEP]" for i in range(len(x_splited)-1)]
        x_splited_with_special.append(f"[CLS] {x_splited[len(x_splited)-1]}") # last sentence
        
        
        # 각 문장에 [CLS], [SEP] 붙인것들 join 하기
        x_with_special = " ".join(x_splited_with_special)

        # 문장들의 정답 라벨 구하기 -> [1, 0, 0, 1] 같은 형식
        labels = [1 if sentence in y else 0 for sentence in x_splited]
        labels += [0 for _ in range(60-len(labels))]

        # 나뉘어진 문장을 tokenizer에 넣은 후 CLS토큰의 index찾아서 담아주기 → 여기 예시에서는 CLS가 4개 -> 위치 index도 담아야함
        # res = self.tokenizer(x_with_special, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        res = self.tokenizer(x_with_special, padding="max_length", truncation=True, return_tensors="pt")
        cls_idx = [i for i in range(res["input_ids"][0].size(0)) if res["input_ids"][0][i] == self.tokenizer.convert_tokens_to_ids("[CLS]")]
        cls_idx += [-1 for _ in range(60-len(cls_idx))]
        
        x_splited += ["" for _ in range(60-len(x_splited))]
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
        # print(f"x_splited : {x_splited}")
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

def collate_fn(samples):
    original_splited = [sample["original_splited"] for sample in samples]
    input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
    token_type_ids = torch.stack([sample["token_type_ids"] for sample in samples], dim=0)
    attention_mask = torch.stack([sample["attention_mask"] for sample in samples], dim=0)
    cls_idx = torch.stack([sample["cls_idx"] for sample in samples], dim=0)
    labels = torch.stack([sample["labels"] for sample in samples], dim=0)
    res_dict = {"original_splited" : original_splited,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "cls_idx": cls_idx,
                    "labels": labels}
    return res_dict

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

class SummaryModel(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        # self.config = AutoConfig.from_pretrained(model_name)
        # self.config.max_position_embeddings = 1026

        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        # self.pos_emb = PositionalEncoding(dropout, self.config.hidden_size, max_len=self.condig.max_position_embeddings)

    def forward(self, input_dict):
        # print(tokenizer.decode(input_dict["input_ids"]))
        # print(input_dict["cls_idx"].size())
        # print(input_dict["labels"].size())
        if input_dict["input_ids"].size(0) == 1:
            output = self.encoder(input_ids=input_dict["input_ids"].to(self.device), 
                                attention_mask=input_dict["attention_mask"].to(self.device),
                                token_type_ids=input_dict["token_type_ids"].to(self.device))
        else:
            output = self.encoder(input_ids=input_dict["input_ids"].squeeze().to(self.device), 
                            attention_mask=input_dict["attention_mask"].squeeze().to(self.device),
                            token_type_ids=input_dict["token_type_ids"].squeeze().to(self.device))
        embed_vectors = output[0]
        #embed_vectors : [batch, 512, 768], cls_idx : [batch, 60]
        cls_idx = input_dict["cls_idx"] if input_dict["cls_idx"].size(0)==1 else input_dict["cls_idx"].squeeze()
        labels = input_dict["labels"] if input_dict["labels"].size(0)==1 else input_dict["labels"].squeeze()
        all_probs = []
        for idx, batch in enumerate(embed_vectors):
            class_idx = cls_idx[idx, :].tolist()
            temp = []
            for i in class_idx:
                if i == -1:
                    break
                temp.append(batch[i,:])
            cls_stack = torch.stack(temp, dim=0)
            logits = self.classifier(cls_stack)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs)
        return all_probs

class CustomBCE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, labels, probs:list):
        all_loss = 0
        for prob, label_all in zip(probs, labels):
            prob = prob.to(device)
            label = label_all[0, 0:prob.size(0)].to(device)
            loss = F.binary_cross_entropy(prob, label.float())
            all_loss += loss
        return all_loss/len(probs)

model = SummaryModel(model_name, device).to(device)
criterion = CustomBCE()
optimizer = torch.optim.Adam(model.parameters())
epochs = 30

def get_top_sentence(probs, sentence_list):
    pred_sentences = []
    for prob, sentence in zip(probs, sentence_list):
        k = 3 if prob.size(0) >= 3 else prob.size(0)
        top_k = torch.topk(prob, k)
        top_vals = top_k.values
        top_indices = top_k.indices
        threshold_index = torch.where(top_vals > 0.5)
        candidate = torch.sort(top_indices[threshold_index]).values
        temp_sentence = []
        for idx in candidate:
            temp_sentence.append(sentence[idx])
        pred_sentences.append(" ".join(temp_sentence))
    return pred_sentences

def get_ground_truth(labels, sentence_list):
    ans = []
    for label, sentence in zip(labels, sentence_list):
        index = torch.where(label==1)[1]
        temp = []
        for idx in index:
            temp.append(sentence[idx])
        ans.append(" ".join(temp))
    return ans

def get_Rouge_L_F(preds, labels):
    m = RougeL(multiref="best")
    score = 0
    for pred, label in zip(preds, labels):
        candidate = pred
        references = [label]
        m.update(([candidate], [references]))
        res = m.compute()
        score += res["Rouge-L-F"]
    score /= len(label)
    return score
        


def eval_data(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        all_score = 0
        cnt = 0
        for batch in tqdm(data_loader):
            probs = model(batch)
            labels = batch["labels"].to(device)
            preds = get_top_sentence(probs, batch["original_splited"])
            ground_truth = get_ground_truth(labels, batch["original_splited"])
            score = get_Rouge_L_F(preds, ground_truth)
            # print(score)
            cnt += 1
            all_score += score
        return all_score/cnt

for epoch in tqdm(range(epochs)):
    print(f"epoch {epoch} start!!")
    iter_num = 0
    for batch in tqdm(train_dataloader):
        model.train()
        probs = model(batch)
        labels = batch["labels"].to(device)
        loss = criterion(labels, probs)
        optimizer.zero_grad()
    
        loss.backward()
        optimizer.step()
        iter_num += 1
        if iter_num % 100 == 0:
            rouge = eval_data(model, val_dataloader, device)
            PATH = f"./saved_checkpoints/model_{iter_num}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                "rouge": rouge
            }, PATH)
            print("="*50)
            print(f"epoch : {epoch}, iter_num : {iter_num}, train_loss : {loss}, val_Rouge_score : {rouge}")
            print("="*50)