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
1. Clone repository:
```text
   git clone https://github.com/duythanh0106/Invoice_Extraction_UsingDeepSeek.git
   cd Invoice_Extraction_UsingDeepSeek
```
2. Cài đặt thư viện:

   download the vllm-0.8.5 [https://github.com/vllm-project/vllm/releases/tag/v0.8.5](https://github.com/vllm-project/vllm/releases/tag/v0.8.5)
```text
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
   pip install flash-attn==2.7.3 --no-build-isolation
   pip install -r requirements.txt
```
   _Lưu ý: Cần cài đặt thêm các thư viện hệ thống nếu chạy trên Linux:_
```text   
   sudo apt-get update && sudo apt-get install libgl1
```

## Hướng dẫn chạy
```text
   python master_pipeline.py
```
Quy trình xử lý bên trong:

Step 1: Quét ảnh từ thư mục inputs/.

Step 2 (OCR): Chạy DeepSeek-OCR (vLLM) để chuyển đổi ảnh sang định dạng Markdown. Kết quả lưu tại ocr_results/.

Step 3 (Extraction): Chạy DeepSeek-LLM-7B để trích xuất thông tin từ Markdown sang JSON theo Schema định sẵn.

Step 4 (Evaluation): So khớp file JSON kết quả với ground_truth/ và xuất báo cáo final_evaluation_report.json.

## Kết quả đánh giá (10 ảnh):
```text
════════════════════════════════════════
       📊 PERFORMANCE SUMMARY
════════════════════════════════════════
 Precision:       47.13%
 Recall:          64.65%
 F1 Score:        54.20%
 Accuracy:        51.68%
────────────────────────────────────────
 Avg Edit Dist:   8.7761
 Avg WER:         0.5294
 Avg CER:         0.5279
════════════════════════════════════════
```

## Schema JSON:
```text
{
  "retailer_name": "BÁCH HÓA XANH",
  "store_name": null,
  "store_address": null,
  "bill_id": "OV109141411144292",
  "bill_id_barcode": null,
  "buy_date": "01/11/2024",
  "buy_time": 07:24,
  "line_items": [
    {
      "product_SKU": null,
      "quantity": 2,
      "product_name": "nước tăng lực sting dâu...",
      "unit_price": 49000,
      "product_total": 98000
    }
  ]
}
```

## Contributing
Mọi đóng góp vui lòng tạo Pull Request hoặc mở Issue

## License
Project này sử dụng mã nguồn từ DeepSeek-AI. Tuân thủ giấy phép của repo gốc
