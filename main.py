# Tên tệp: main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re
from torch.cuda.amp import autocast
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# --- 0. TẢI BIẾN MÔI TRƯỜNG ---
load_dotenv()


# --- 1. HÀM LÀM SẠCH VĂN BẢN ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'On\s.*(wrote|writes):', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'>.*', '', text, flags=re.DOTALL)
    text = \
    re.split(r'(\n--\n|\n- \n|\n_+\n|\nRegards|\nCảm ơn|\nTrân trọng|\nSent from|\nWARNING:|\nConfidentiality Notice)',
             text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r'^(RE:|FW:|FWD:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,?!áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 2. KIẾN TRÚC MODEL ---
class BERTMultiTask(nn.Module):
    def __init__(self, model_name, num_type, num_priority, pri_weights=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.type_head = nn.Linear(hidden, num_type)
        self.priority_head = nn.Linear(hidden, num_priority)

    def forward(self, input_ids, attention_mask, type_label=None, priority_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        type_logits = self.type_head(pooled)
        priority_logits = self.priority_head(pooled)
        return {"loss": None, "type_logits": type_logits, "priority_logits": priority_logits}


# --- 3. CẤU HÌNH & TẢI MODEL ---
class InputData(BaseModel):
    text: str


MODEL_PATH = "best_model_text_cleaned.pt"
MODEL_NAME = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TYPE_LABELS_SORTED = ['Change', 'Incident', 'Problem', 'Request']
type2id = {label: i for i, label in enumerate(TYPE_LABELS_SORTED)}
id2type = {i: label for label, i in type2id.items()}

PRIORITY_ORDER = ['low', 'medium', 'high']
priority2id = {label: i for i, label in enumerate(PRIORITY_ORDER)}
id2priority = {i: label for label, i in priority2id.items()}


def load_model_and_tokenizer():
    print(f"--- Đang tải model {MODEL_NAME} lên {device}... ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BERTMultiTask(MODEL_NAME, len(type2id), len(priority2id), pri_weights=None)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("--- Tải model thành công! ---")
        return model, tokenizer
    except FileNotFoundError:
        print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy tệp model '{MODEL_PATH}'.")
        return None, None
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return None, None


# --- 4. KẾT NỐI DATABASE ---
def connect_to_db():
    MONGO_URL = os.environ.get("MONGODB_URL")
    if not MONGO_URL:
        print("CẢNH BÁO: Không tìm thấy MONGODB_URL. Sẽ không lưu vào database.")
        return None
    try:
        client = MongoClient(MONGO_URL)
        db = client.get_database("email_app")  # Tên database (tùy chọn)
        collection = db.get_collection("predictions")  # Tên collection
        print("--- Kết nối MongoDB thành công! ---")
        return collection
    except Exception as e:
        print(f"Lỗi khi kết nối MongoDB: {e}")
        return None


# --- 5. KHỞI TẠO APP & ENDPOINT ---
app = FastAPI()
model, tokenizer = load_model_and_tokenizer()
db_collection = connect_to_db()


@app.post("/predict")
async def predict(data: InputData):
    if not model or not tokenizer:
        return {"error": "Model không được tải. Kiểm tra log."}

    cleaned_text = clean_text(data.text)

    inputs = tokenizer(
        [cleaned_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=384
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), autocast():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    type_pred_idx = torch.argmax(outputs['type_logits'], dim=1).item()
    pri_pred_idx = torch.argmax(outputs['priority_logits'], dim=1).item()

    type_label = id2type.get(type_pred_idx, "Unknown")
    pri_label = id2priority.get(pri_pred_idx, "Unknown")

    frontend_result = {
        "cleaned_text": cleaned_text,
        "predicted_type": type_label,
        "predicted_priority": pri_label
    }

    if db_collection is not None:
        db_entry = frontend_result.copy()
        db_entry["timestamp"] = datetime.now()
        try:
            db_collection.insert_one(db_entry)
        except Exception as e:
            print(f"Lỗi khi lưu vào DB: {e}")
    return frontend_result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)