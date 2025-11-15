SmartMail Classifier
Ứng dụng DistilBERT trong tự động phân loại và ưu tiên email hỗ trợ khách hàng

SmartMail Classifier là hệ thống ứng dụng Deep Learning – cụ thể là mô hình DistilBERT – nhằm tự động đọc, hiểu và phân loại email hỗ trợ khách hàng. Dự án được thiết kế để giải quyết các bài toán trong vận hành doanh nghiệp, đặc biệt là khi số lượng email tới từ khách hàng ngày càng lớn và khó kiểm soát.

Hệ thống giúp:

🔍 Nhận diện chủ đề email tự động (Incident, Request, Change, Problem, v.v.)

⚡ Xác định mức độ ưu tiên (Priority) dựa trên nội dung

📥 Xử lý nhanh khối lượng email lớn mà không cần can thiệp thủ công

🎯 Giảm thời gian phản hồi và cải thiện SLA của bộ phận CS (Customer Support)

🔄 Tích hợp dễ dàng vào các hệ thống Helpdesk như Jira Service Desk / ServiceNow / Freshdesk

SmartMail Classifier hướng tới một giải pháp:

Chính xác, nhờ sức mạnh của mô hình ngôn ngữ Transformer

Nhẹ và nhanh, nhờ sử dụng DistilBERT (phiên bản rút gọn BERT nhưng hiệu năng cao)

Có thể mở rộng & huấn luyện lại theo dữ liệu riêng của doanh nghiệp

📘 Overview

SmartMail Classifier là hệ thống phân loại email thông minh dành cho bộ phận hỗ trợ khách hàng (Customer Support).
Mục tiêu chính: tự động xác định loại yêu cầu (Incident, Request, Change, Problem…) và mức độ ưu tiên dựa trên nội dung email.

Hệ thống này sử dụng DistilBERT, một biến thể rút gọn của BERT nhưng vẫn giữ được 95% hiệu năng trong khi nhanh hơn 60%.
Giải pháp giúp doanh nghiệp giảm tải khối lượng xử lý email thủ công, tăng tốc phản hồi, cải thiện SLA, và tối ưu vận hành.

✨ Features

SmartMail Classifier cung cấp các chức năng chính:

📥 1. Email Classification

Tự động phân loại email vào các nhóm:

Incident

Request

Change

Problem

🎯 2. Priority Prediction

Dựa trên nội dung email, hệ thống dự đoán mức độ ưu tiên:

 High / Medium / Low

Evaluation & Saving

Đánh giá trên tập test:

Accuracy

Precision / Recall

F1-score

Tiền xử lý:

Loại bỏ ký tự đặc biệt

Lowercase

Tokenization bằng DistilBERT tokenizer

Chia dữ liệu:

Train: 80%

Validation: 10%

Test: 10%

🔧 Training Pipeline

Quy trình huấn luyện được thực hiện theo 6 bước:

1️⃣ Data Loading & Preprocessing

Đọc dữ liệu từ CSV/Excel

Làm sạch văn bản

2️⃣ Dataset & Dataloader

Tạo TensorDataset

Batch training

3️⃣ Class Weighting

Để xử lý mất cân bằng dữ liệu:

criterion = nn.CrossEntropyLoss(weight=class_weights)

4️⃣ Model Initialization
from transformers import DistilBertModel

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels = num_classes
)

5️⃣ Training Loop

AdamW optimizer

Learning rate scheduler

Epoch-based training

Backpropagation

Gradient clipping

Theo dõi:

Train loss

Validation accuracy

F1-score

6️⃣ Evaluation & Saving

Đánh giá trên tập test:

Accuracy

Precision / Recall

F1-score

