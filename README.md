# Invoice Extraction Pipeline using DeepSeek-VLM & LLM 🧾

Hệ thống trích xuất thông tin hóa đơn tự động (End-to-End Invoice Extraction) sử dụng kết hợp mô hình thị giác ngôn ngữ (VLM) và mô hình ngôn ngữ lớn (LLM).


## 🚀 Tính năng nổi bật
* **Pipeline tự động hóa 100%:** Từ ảnh đầu vào -> JSON kết quả -> Báo cáo đánh giá.
* **DeepSeek-OCR (VLM):** Sử dụng `DeepSeek-VL` chạy trên nền tảng **vLLM** cho tốc độ xử lý cực nhanh (High Throughput).
* **DeepSeek-LLM (Extraction):** Sử dụng `DeepSeek-LLM-7B`.
* **Auto Evaluation:** Tự động so sánh kết quả với Ground Truth và tính toán các chỉ số: Precision, Recall, F1, WER, CER, Edit Distance.
* **No-Slicing Strategy:** Xử lý ảnh hóa đơn gốc trực tiếp với chế độ Crop thông minh, không cần cắt ảnh thủ công.

## 🛠️ Cấu trúc thư mục
```text
.
├── inputs/               # Chứa ảnh hóa đơn đầu vào (.jpg, .png)
├── ground_truth/         # Chứa file JSON nhãn chuẩn (để đánh giá)
├── outputs/              # Chứa file JSON kết quả trích xuất
├── ocr_results/          # Chứa file Markdown trung gian từ OCR
├── DeepSeek-OCR/         # Source code DeepSeek-OCR (vLLM version)
├── master_pipeline.py    # Script chính điều khiển toàn bộ quy trình
├── deepseek_llm_7b.py    # Module trích xuất thông tin (LLM)
└── parse_level_evaluate.py # Module đánh giá kết quả
```

## ⚙️ Cài đặt
Clone repository:

Bash

git clone [https://github.com/duythanh0106/Invoice_Extraction_UsingDeepSeek.git](https://github.com/duythanh0106/Invoice_Extraction_UsingDeepSeek.git)
cd Invoice_Extraction_UsingDeepSeek
