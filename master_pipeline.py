import os
import shutil
import subprocess
import sys
import json
import re

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
INPUT_DIR = "inputs"            
GT_DIR = "ground_truth"
FINAL_OUTPUT_DIR = "outputs"      # lưu JSON 
OCR_SAVE_DIR = "ocr_outputs/"      # lưu file Markdown OCR 
EVAL_REPORT_FILE = "final_evaluation_report.json"

# --- CẤU HÌNH DEEPSEEK (SỬA CHO ĐÚNG MÁY BẠN) ---
DEEPSEEK_REPO_DIR = "DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm" 
PATH_TO_OCR_SCRIPT = os.path.join(DEEPSEEK_REPO_DIR, "run_dpsk_ocr_eval_batch.py")
PATH_TO_CONFIG_FILE = os.path.join(DEEPSEEK_REPO_DIR, "config.py")

PATH_TO_LLM_SCRIPT = "deepseek_llm_7b.py"
PATH_TO_EVAL_SCRIPT = "parse_level_evaluate.py"


def setup_dirs():
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OCR_SAVE_DIR, exist_ok=True)
    if not os.path.exists(GT_DIR): 
        os.makedirs(GT_DIR)
        print(f"Created empty '{GT_DIR}'. Please put Ground Truth JSONs here!")

def update_deepseek_config(config_path, input_path, output_path):
    print(f"🔧 Updating config file...")
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)
    
    if not abs_output.endswith(os.sep):
        abs_output += os.sep

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(r"INPUT_PATH\s*=\s*['\"].*?['\"]", f"INPUT_PATH = '{abs_input}'", content)
        content = re.sub(r"OUTPUT_PATH\s*=\s*['\"].*?['\"]", f"OUTPUT_PATH = '{abs_output}'", content)
        
        content = re.sub(r"CROP_MODE\s*=\s*(False|True)", "CROP_MODE = True", content)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content) 
        print("Config updated!")
    except Exception as e:
        print(f"Error updating config: {e}")
        sys.exit(1)

def run_deepseek_ocr():
    print("\n>>> STEP 1: Running DeepSeek-OCR...")
    
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print("Error: 'inputs' folder is empty!")
        sys.exit(1)

    update_deepseek_config(PATH_TO_CONFIG_FILE, INPUT_DIR, OCR_SAVE_DIR)
    
    # Chạy OCR
    working_dir = os.path.dirname(PATH_TO_OCR_SCRIPT)
    command = [sys.executable, PATH_TO_OCR_SCRIPT]
    
    # Lưu ý: Cần set cwd để script tìm thấy config.py
    result = subprocess.run(command, cwd=working_dir)
    
    if result.returncode != 0:
        print("ERROR: DeepSeek-OCR failed!")
        sys.exit(1)

def run_deepseek_llm():
    print("\n>>> STEP 2: Running DeepSeek-LLM Extraction...")
    
    # Dọn dẹp file rác _det.md
    for f in os.listdir(OCR_SAVE_DIR):
        if "_det.md" in f:
            try: os.remove(os.path.join(OCR_SAVE_DIR, f))
            except: pass

    if not os.listdir(OCR_SAVE_DIR):
        print("No markdown files found. Skipping LLM...")
        return

    # LLM đọc từ OCR_SAVE_DIR
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

    command = [
        sys.executable, PATH_TO_EVAL_SCRIPT,
        "--gt_dir", GT_DIR,
        "--pred_dir", FINAL_OUTPUT_DIR,
        "--out", EVAL_REPORT_FILE
    ]
    
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print(f"Evaluation Complete! Report saved to: {EVAL_REPORT_FILE}")
        
        try:
            with open(EVAL_REPORT_FILE, 'r', encoding='utf-8') as f:
                report = json.load(f)
                summary = report.get('summary', {}).get('overall_summary', {})
                
                print("\n" + "═"*40)
                print("       📊 PERFORMANCE SUMMARY       ")
                print("═"*40)
                print(f" Precision:       {summary.get('precision', 0):.2%}")
                print(f" Recall:          {summary.get('recall', 0):.2%}")
                print(f" F1 Score:        {summary.get('f1_score', 0):.2%}")
                print(f" Accuracy:        {summary.get('accuracy', 0):.2%}")
                print("─" * 40)
                print(f" Avg Edit Dist:   {summary.get('avg_edit_distance', 0):.4f}")
                print(f" Avg WER:         {summary.get('avg_wer', 0):.4f}")
                print(f" Avg CER:         {summary.get('avg_cer', 0):.4f}")
                print("═"*40 + "\n")
        except Exception as e:
            print(f"Could not print summary to console: {e}")
    else:
        print("ERROR: Evaluation script failed!")

if __name__ == "__main__":
    setup_dirs()
    
    run_deepseek_ocr()
    run_deepseek_llm()
    evaluate()
    
