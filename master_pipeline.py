import os
import shutil
import subprocess
import sys
import json
import re
import cv2
import numpy as np

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
INPUT_DIR = "inputs"
GT_DIR = "ground_truth"
FINAL_OUTPUT_DIR = "outputs"      
OCR_SAVE_DIR = "ocr_results"      # Folder lưu kết quả OCR
TEMP_DIR = "temp"                 # Folder tạm chứa ảnh đã qua xử lý
EVAL_REPORT_FILE = "final_evaluation_report.json"

# --- CẤU HÌNH DEEPSEEK (SỬA CHO ĐÚNG MÁY BẠN) ---
DEEPSEEK_REPO_DIR = "DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm" 
PATH_TO_OCR_SCRIPT = os.path.join(DEEPSEEK_REPO_DIR, "run_dpsk_ocr_eval_batch.py")
PATH_TO_CONFIG_FILE = os.path.join(DEEPSEEK_REPO_DIR, "config.py")

PATH_TO_LLM_SCRIPT = "deepseek_llm_7b.py"
PATH_TO_EVAL_SCRIPT = "parse_level_evaluate.py"

# ================= CÁC HÀM TIỆN ÍCH HIỂN THỊ =================

def print_styled_table(title, headers, rows, col_widths):
    """Hàm in bảng đẹp với khung Unicode (Copy từ evaluate script)"""
    TL, TM, TR = '┌', '┬', '┐'; BL, BM, BR = '└', '┴', '┘'
    VL, VR = '│', '│'; HL, VM = '─', '┼'; ML, MR = '├', '┤'

    fmt_parts = [f" {{{i}:{'<' if i==0 else '>'}{w}}} " for i, w in enumerate(col_widths)]
    row_fmt = VL + VL.join(fmt_parts) + VR
    
    def get_sep(left, mid, right, cross):
        segs = [mid * (w + 2) for w in col_widths]
        return left + cross.join(segs) + right

    print("\n" + " " + title.upper())
    print(get_sep(TL, HL, TR, TM))
    print(row_fmt.format(*headers))
    print(get_sep(ML, HL, MR, VM))
    for row in rows:
        print(row_fmt.format(*row))
    print(get_sep(BL, HL, BR, BM))

# ================= CÁC HÀM PIPELINE =================

def setup_dirs():
    # Dọn dẹp folder tạm
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OCR_SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(GT_DIR):
        os.makedirs(GT_DIR)

def update_deepseek_config(config_path, input_path, output_path):
    print(f"Updating config file...")
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)
    
    # [QUAN TRỌNG] Thêm dấu / vào cuối đường dẫn output để tránh lỗi dính tên file
    if not abs_output.endswith(os.sep):
        abs_output += os.sep

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex thay thế đường dẫn
        content = re.sub(r"INPUT_PATH\s*=\s*['\"].*?['\"]", f"INPUT_PATH = '{abs_input}'", content)
        content = re.sub(r"OUTPUT_PATH\s*=\s*['\"].*?['\"]", f"OUTPUT_PATH = '{abs_output}'", content)
        # Bắt buộc bật chế độ tự cắt ảnh (CROP_MODE) cho ảnh dài
        content = re.sub(r"CROP_MODE\s*=\s*(False|True)", "CROP_MODE = True", content)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content) 
        print(" Config updated successfully!")
    except Exception as e:
        print(f"Error updating config: {e}")
        sys.exit(1)

def run_deepseek_ocr():
    print("\n>>> STEP 1: Running DeepSeek-OCR...")
    
    # Update config để trỏ vào folder ảnh đầu vào
    update_deepseek_config(PATH_TO_CONFIG_FILE, INPUT_DIR, OCR_SAVE_DIR)
    
    working_dir = os.path.dirname(PATH_TO_OCR_SCRIPT)
    command = [sys.executable, PATH_TO_OCR_SCRIPT]
    
    # Chạy OCR và ẩn bớt output rác nếu muốn, ở đây để hiện để debug
    result = subprocess.run(command, cwd=working_dir)
    if result.returncode != 0:
        print(" ERROR: DeepSeek-OCR failed!")
        sys.exit(1)

def run_deepseek_llm():
    print("\n>>> STEP 2: Running DeepSeek-LLM Extraction...")
    
    # Xóa file rác _det.md sinh ra từ bước OCR
    for f in os.listdir(OCR_SAVE_DIR):
        if "_det.md" in f:
            try: os.remove(os.path.join(OCR_SAVE_DIR, f))
            except: pass

    if not os.listdir(OCR_SAVE_DIR):
        print("No markdown files found. Skipping LLM...")
        return

    command = [
        sys.executable, PATH_TO_LLM_SCRIPT,
        "--input_dir", OCR_SAVE_DIR, 
        "--output_dir", FINAL_OUTPUT_DIR
    ]
    subprocess.run(command, check=True)

def evaluate():
    print("\n>>> STEP 3: Evaluating Results...")
    
    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith('.json')]
    if not gt_files:
        print(f"Skipping evaluation (No GT files in '{GT_DIR}').")
        return

    # Chạy script đánh giá và lưu kết quả vào JSON, không in ra màn hình console của subprocess
    command = [
        sys.executable, PATH_TO_EVAL_SCRIPT,
        "--gt_dir", GT_DIR,
        "--pred_dir", FINAL_OUTPUT_DIR,
        "--out", EVAL_REPORT_FILE
    ]
    
    # capture_output=True để script đánh giá không in bảng 2 lần (1 lần trong subprocess, 1 lần ở đây)
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Evaluation Calculation Complete! Reading report...")
        
        if not os.path.exists(EVAL_REPORT_FILE):
            print("Error: Report file not found.")
            return

        try:
            with open(EVAL_REPORT_FILE, 'r', encoding='utf-8') as f:
                report = json.load(f)
                
            summ = report.get('summary', {})
            total = summ.get('total_images', 0)
            
            print("\n" * 2)
            print(f"📊  BÁO CÁO ĐÁNH GIÁ TỔNG HỢP (Images: {total})")
            print("=" * 135)
            
            # Cấu hình bảng hiển thị
            headers = ["FIELD", "T_PRE", "T_REC", "T_F1", "T_ACC", "C_PRE", "C_REC", "C_F1", "C_ACC", "EDIT", "WER", "CER"]
            widths =  [18,      6,       6,       6,      6,       6,       6,       6,       6,       6,      6,     6]

            # 1. OVERALL SYSTEM
            ov = summ.get("overall", {})
            rows_ov = [[
                "OVERALL",
                f"{ov.get('precision',0):.1%}", f"{ov.get('recall',0):.1%}", f"{ov.get('f1_score',0):.1%}", f"{ov.get('accuracy',0):.1%}",
                f"{ov.get('char_precision',0):.1%}", f"{ov.get('char_recall',0):.1%}", f"{ov.get('char_f1',0):.1%}", f"{ov.get('char_accuracy',0):.1%}",
                f"{ov.get('avg_edit_distance',0):.2f}", f"{ov.get('avg_wer',0):.2f}", f"{ov.get('avg_cer',0):.2f}"
            ]]
            print_styled_table("🔷 TỔNG QUAN (OVERALL SYSTEM)", headers, rows_ov, widths)

            # 2. GENERAL FIELDS
            rows_gen = []
            for k, v in summ.get("fields", {}).items():
                rows_gen.append([
                    k,
                    f"{v['precision']:.1%}", f"{v['recall']:.1%}", f"{v['f1_score']:.1%}", f"{v['accuracy']:.1%}",
                    f"{v['char_precision']:.1%}", f"{v['char_recall']:.1%}", f"{v['char_f1']:.1%}", f"{v['char_accuracy']:.1%}",
                    f"{v['edit_distance']:.2f}", f"{v['wer']:.2f}", f"{v['cer']:.2f}"
                ])
            print_styled_table("🔷 THÔNG TIN CHUNG (HEADER FIELDS)", headers, rows_gen, widths)

            # 3. LINE ITEMS
            rows_li = []
            li_gen = summ.get("line_item", {}).get("general", {})
            li_subs = summ.get("line_item", {}).get("sub_fields", {})
            
            # System Level (Detection only)
            rows_li.append([
                "► LI (SYSTEM)",
                f"{li_gen.get('precision',0):.1%}", f"{li_gen.get('recall',0):.1%}", f"{li_gen.get('f1_score',0):.1%}", f"{li_gen.get('accuracy',0):.1%}",
                "-", "-", "-", "-", 
                "-", "-", "-"
            ])
            
            # Sub-fields
            for k, v in li_subs.items():
                rows_li.append([
                    f"  └ {k}",
                    f"{v['precision']:.1%}", f"{v['recall']:.1%}", f"{v['f1_score']:.1%}", f"{v['accuracy']:.1%}",
                    f"{v['char_precision']:.1%}", f"{v['char_recall']:.1%}", f"{v['char_f1']:.1%}", f"{v['char_accuracy']:.1%}",
                    f"{v['edit_distance']:.2f}", f"{v['wer']:.2f}", f"{v['cer']:.2f}"
                ])
                
            print_styled_table("🔷 CHI TIẾT SẢN PHẨM (LINE ITEMS)", headers, rows_li, widths)
            
            print("\n📝 GHI CHÚ:")
            print("  - T_...: Token Metrics (Theo từ).")
            print("  - C_...: Char Metrics (Theo ký tự).")
            print("  - EDIT: Edit Distance (Số thao tác sửa đổi).")
            print("=" * 135 + "\n")

        except Exception as e:
            print(f"Error displaying report: {e}")
            # Nếu lỗi hiển thị bảng, in raw output từ subprocess để debug
            print("Raw output from eval script:")
            print(result.stdout)
            print(result.stderr)
    else:
        print("Evaluation Script Failed!")
        print(result.stderr)

if __name__ == "__main__":
    setup_dirs()
    run_deepseek_ocr()
    run_deepseek_llm()
    evaluate()