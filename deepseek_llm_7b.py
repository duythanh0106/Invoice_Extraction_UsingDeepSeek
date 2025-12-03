import os
import json
import torch
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 1. Load Model & Tokenizer
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FIX: Gán pad_token nếu chưa có 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True 
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

def extract_json_from_text(file_text):
    prompt = f"""
You are a generic invoice extraction system.
Your task is to extract data from the provided OCR text into a JSON object.

### GUIDELINES:

1. **Retailer Name (Flexible)**: 
   - Look at the header (top) of the text.
   - Identify the main **Brand Name** (e.g., "Bách Hóa Xanh", "Co.opmart", "Highlands Coffee").
   - **Constraint**: If the text contains the brand name, extract it. If the top section is garbled, missing, or unclear, set `"retailer_name": null`.
   - **Do NOT** default to "Bách Hóa Xanh" if the text does not support it.

2. **Prices & Line Items (Math Logic)**:
   - OCR often misses the "Unit Price" column.
   - **Logic**: If `unit_price` is missing but `product_total` and `quantity` exist, CALCULATE: `unit_price` = `product_total` / `quantity`.
   - If `quantity` is found on a separate line below the product name, link them together.

3. **General Rules**:
   - Dates: Convert to DD/MM/YYYY.
   - Nulls: Use `null` for any missing field.

### JSON SCHEMA:
{{
  "retailer_name": "Brand name found in text (or null)",
  "store_name": "Store/Branch name (or null)",
  "store_address": "Address string (or null)",
  "bill_id": "Invoice Number (Look for 'Số CT', 'Số HD', 'Inv No'...)",
  "bill_id_barcode": "Lookup code/Barcode string (or null)",
  "buy_date": "DD/MM/YYYY",
  "buy_time": "HH:MM",
  "line_items": [
    {{
      "product_SKU": "Product code (or null)",
      "quantity": "Integer",
      "product_name": "String",
      "unit_price": "Integer",
      "product_total": "Integer"
    }}
  ]
}}

### INPUT TEXT:
{file_text}

### OUTPUT JSON:
"""
    
    messages = [{"role": "user", "content": prompt}]
    
    # Tạo input tensor
    input_tensor = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # FIX: Đưa input vào đúng device của model
    input_tensor = input_tensor.to(model.device)
    
    # Tạo mask (đã fix pad_token ở trên nên dòng này sẽ chạy đúng)
    attention_mask = input_tensor.ne(tokenizer.pad_token_id).long()

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=1000, # Tăng lên chút để tránh bị cắt giữa chừng nếu hóa đơn dài
            do_sample=True,     # Greedy decoding để kết quả ổn định (deterministic)
            temperature=0.1
        )

    # Decode
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    # FIX: Dùng Regex để tìm JSON object chuẩn xác hơn
    # Tìm chuỗi bắt đầu bằng { và kết thúc bằng } (non-greedy)
    match = re.search(r'\{.*\}', result, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    
    return result

if __name__ == "__main__":
    # Thêm bộ đọc tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder chứa file .md")
    parser.add_argument("--output_dir", required=True, help="Folder lưu .json")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"LLM Processing from: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(".md") and not filename.endswith("det.md"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}...")
        
            with open(file_path, "r", encoding="utf-8") as f:
                file_text = f.read()

            # Bỏ qua file rỗng hoặc quá ngắn
            if len(file_text.strip()) < 10:
                print(f"Skipping empty file: {filename}")
                continue

            json_text = extract_json_from_text(file_text)

            try:
                data = json.loads(json_text)
                output_path = os.path.join(output_dir, filename.replace(".md", ".json"))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Saved: {output_path}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from: {filename}")
                print(f"Error: {e}")
                # Ghi log lỗi để debug
                with open(os.path.join(output_dir, f"ERROR_{filename}"), "w", encoding="utf-8") as f:
                    f.write(json_text)
