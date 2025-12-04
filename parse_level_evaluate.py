import os
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import argparse
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

def normalize_numeric(text: str) -> str:
    """
    Loại bỏ tất cả dấu chấm, phẩy, chữ cái, chỉ giữ lại số.
    Ví dụ: "168.000" -> "168000", "24,500đ" -> "24500"
    """
    if not text: return ""
    return re.sub(r"[^0-9]", "", str(text))

# =============================================================================
# 1. DATA STRUCTURES & PARSING
# =============================================================================

@dataclass
class LineItem:
    product_SKU: str = ""
    quantity: str = ""
    product_name: str = ""
    unit_price: str = ""
    product_total: str = ""

@dataclass
class InvoiceDoc:
    retailer_name: str = ""
    store_name: str = ""
    store_address: str = ""
    bill_id: str = ""
    bill_id_barcode: str = ""
    buy_date: str = ""
    buy_time: str = ""
    line_items: List[LineItem] = field(default_factory=list)
    raw_text: str = ""

def parse_json_invoice(json_text: str) -> InvoiceDoc:
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return InvoiceDoc()
    
    if not isinstance(data, dict): return InvoiceDoc()
    
    def get_str(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return str(d[k])
        return ""

    return InvoiceDoc(
        retailer_name=str(data.get("retailer_name", "") or ""),
        store_name=str(data.get("store_name", "") or ""),
        store_address=str(data.get("store_address", "") or ""),
        bill_id=str(data.get("bill_id", "") or ""),
        bill_id_barcode=str(data.get("bill_id_barcode", "") or ""),
        buy_date=str(data.get("buy_date", "") or ""),
        buy_time=str(data.get("buy_time", "") or ""),
        line_items=[
            LineItem(
                product_SKU=get_str(item, ["product_SKU", "sku"]),
                quantity=get_str(item, ["quantity", "qty"]),
                product_name=get_str(item, ["product_name", "description", "name"]),
                unit_price=get_str(item, ["unit_price", "price"]),
                product_total=get_str(item, ["product_total", "total", "line_total"])
            ) for item in (data.get("line_items", []) or data.get("line_item", []) or []) if isinstance(item, dict)
        ],
        raw_text=json_text
    )

# =============================================================================
# 2. METRICS CALCULATION
# =============================================================================

def exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()

def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b: return 0.0
    dist = Levenshtein.distance(a, b)
    denom = max(len(a), len(b))
    return dist / denom if denom > 0 else float(dist)

def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words and not hyp_words: return 0.0
    dist = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
    return dist / max(len(ref_words), len(hyp_words), 1)

def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis: return 0.0
    ref_chars = reference.replace(' ', '')
    hyp_chars = hypothesis.replace(' ', '')
    if not ref_chars and not hyp_chars: return 0.0
    dist = Levenshtein.distance(ref_chars, hyp_chars)
    return dist / max(len(ref_chars), len(hyp_chars))

def calculate_char_metrics(gt_val: str, pred_val: str) -> Dict[str, float]:
    s1 = str(gt_val).lower() if gt_val else ""
    s2 = str(pred_val).lower() if pred_val else ""
    if not s1 and not s2: return {"char_precision": 1.0, "char_recall": 1.0, "char_f1": 1.0, "char_accuracy": 1.0}
    if not s1 or not s2: return {"char_precision": 0.0, "char_recall": 0.0, "char_f1": 0.0, "char_accuracy": 0.0}

    ops = Levenshtein.editops(s1, s2)
    deletions = sum(1 for op in ops if op[0] == 'delete')
    insertions = sum(1 for op in ops if op[0] == 'insert')
    substitutions = sum(1 for op in ops if op[0] == 'replace')
    
    tp = len(s1) - (deletions + substitutions)
    fp = insertions + substitutions
    fn = deletions + substitutions

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"char_precision": p, "char_recall": r, "char_f1": f1, "char_accuracy": acc}

def token_set_metrics(gt_val: str, pred_val: str):
    gt_tokens = set(re.findall(r"\w+", gt_val.lower()))
    pred_tokens = set(re.findall(r"\w+", pred_val.lower()))
    if not gt_tokens and not pred_tokens: return 1.0, 1.0, 1.0, 1.0
    
    tp = len(gt_tokens & pred_tokens)
    fp = len(pred_tokens - gt_tokens)
    fn = len(gt_tokens - pred_tokens)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    
    union = len(gt_tokens | pred_tokens)
    acc = tp / union if union > 0 else 0.0
    return p, r, f1, acc

def calculate_index_accuracy(gt_items, pred_items):
    if not gt_items and not pred_items: return 1.0
    if not gt_items or not pred_items: return 0.0
    matches = 0
    min_len = min(len(gt_items), len(pred_items))
    for i in range(min_len):
        if fuzz.ratio(gt_items[i].product_name.lower(), pred_items[i].product_name.lower()) > 70:
            matches += 1
    return matches / max(len(gt_items), len(pred_items))

# =============================================================================
# 3. HELPER: RECURSIVE ROUNDING
# =============================================================================

def recursive_round(obj, precision=4):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: recursive_round(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_round(x, precision) for x in obj]
    return obj

# =============================================================================
# 4. EVALUATION LOGIC
# =============================================================================

def evaluate_pair(gt_json: str, pred_json: str, file_name: str) -> Dict[str, Any]:
    gt = parse_json_invoice(gt_json)
    pred = parse_json_invoice(pred_json)
    
    # --- 1. Field Metrics (General Info) ---
    field_results = {}
    ov_accum = {
        "precision": 0, "recall": 0, "f1_score": 0, "accuracy": 0,
        "char_precision": 0, "char_recall": 0, "char_f1": 0, "char_accuracy": 0,
        "edit_distance": 0, "wer": 0, "cer": 0
    }
    total_components = 0

    field_map = [
        ("retailer_name", gt.retailer_name, pred.retailer_name),
        ("store_name", gt.store_name, pred.store_name),
        ("store_address", gt.store_address, pred.store_address),
        ("bill_id", gt.bill_id, pred.bill_id),
        ("bill_id_barcode", gt.bill_id_barcode, pred.bill_id_barcode),
        ("buy_date", gt.buy_date, pred.buy_date),
        ("buy_time", gt.buy_time, pred.buy_time)
    ]

    for fname, g_val, p_val in field_map:
        g_val = str(g_val); p_val = str(p_val)
        
        # Tính toán các chỉ số
        p, r, f1, acc = token_set_metrics(g_val, p_val)
        char_m = calculate_char_metrics(g_val, p_val)
        ed = Levenshtein.distance(g_val, p_val)
        wer = word_error_rate(g_val, p_val)
        cer = character_error_rate(g_val, p_val)
        
        field_results[fname] = {
            "predicted": p_val, "ground_truth": g_val,
            "precision": p, "recall": r, "f1_score": f1, "accuracy": acc,
            "char_precision": char_m["char_precision"], "char_recall": char_m["char_recall"],
            "char_f1": char_m["char_f1"], "char_accuracy": char_m["char_accuracy"],
            "match": exact_match(g_val, p_val),
            "edit_distance": ed, "wer": wer, "cer": cer
        }
        
        # Cộng dồn vào Overall
        ov_accum["precision"] += p; ov_accum["recall"] += r; ov_accum["f1_score"] += f1; ov_accum["accuracy"] += acc
        ov_accum["char_precision"] += char_m["char_precision"]; ov_accum["char_recall"] += char_m["char_recall"]
        ov_accum["char_f1"] += char_m["char_f1"]; ov_accum["char_accuracy"] += char_m["char_accuracy"]
        ov_accum["edit_distance"] += ed; ov_accum["wer"] += wer; ov_accum["cer"] += cer
        total_components += 1

    # --- 2. Line Item Metrics ---
    # A. Matching Logic (Tìm cặp dòng tương ứng dựa trên tên sản phẩm)
    match_candidates = []
    for i, g in enumerate(gt.line_items):
        for j, p in enumerate(pred.line_items):
            # Chỉ ghép cặp nếu tên sản phẩm giống nhau > 70%
            if fuzz.ratio(g.product_name.lower(), p.product_name.lower()) > 70:
                match_candidates.append((fuzz.ratio(g.product_name, p.product_name), i, j))
    match_candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_pairs = []
    used_g, used_p = set(), set()
    for _, i, j in match_candidates:
        if i not in used_g and j not in used_p:
            matched_pairs.append((gt.line_items[i], pred.line_items[j]))
            used_g.add(i); used_p.add(j)
    
    # B. Tính System Metrics (Đếm số dòng bắt được)
    tp = len(matched_pairs)
    fp = len(pred.line_items) - tp
    fn = len(gt.line_items) - tp
    
    li_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    li_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    li_f1 = (2 * li_prec * li_rec) / (li_prec + li_rec) if (li_prec + li_rec) > 0 else 0.0
    li_acc = tp / len(gt.line_items) if len(gt.line_items) > 0 else 0.0

    # C. Tính chi tiết từng trường con (Sub-fields)
    item_details = []
    li_accum = {k: 0 for k in ov_accum} 
    li_comps = 0
    
    sub_fields = ["product_SKU", "quantity", "product_name", "unit_price", "product_total"]
    # Các trường cần chuẩn hóa số học
    NUMERIC_FIELDS = ["quantity", "unit_price", "product_total"]
    
    # C1. Xử lý các cặp đã khớp (Matched Pairs)
    for idx, (g_item, p_item) in enumerate(matched_pairs):
        detail = {"item_index": idx}
        for sf in sub_fields:
            gv = getattr(g_item, sf)
            pv = getattr(p_item, sf)
            
            # --- LOGIC CHUẨN HÓA SỐ HỌC ---
            if sf in NUMERIC_FIELDS:
                # Nếu là số, loại bỏ dấu chấm/phẩy trước khi tính điểm
                gv_calc = normalize_numeric(gv)
                pv_calc = normalize_numeric(pv)
            else:
                # Nếu là chữ (tên sp, sku), giữ nguyên
                gv_calc = gv
                pv_calc = pv
            # -------------------------------

            # Tính điểm dựa trên giá trị đã (hoặc không) chuẩn hóa
            sp, sr, sf1, sacc = token_set_metrics(gv_calc, pv_calc)
            scm = calculate_char_metrics(gv_calc, pv_calc)
            sed = Levenshtein.distance(gv_calc, pv_calc)
            swer = word_error_rate(gv_calc, pv_calc)
            scer = character_error_rate(gv_calc, pv_calc)
            
            # Kiểm tra match: Với số thì so sánh chuỗi số sạch, với chữ thì exact match
            is_match = (gv_calc == pv_calc) if sf in NUMERIC_FIELDS else exact_match(gv, pv)

            detail[sf] = {
                "predicted": pv, "ground_truth": gv, # Hiển thị giá trị gốc để dễ debug
                "precision": sp, "recall": sr, "f1_score": sf1, "accuracy": sacc,
                "char_precision": scm["char_precision"], "char_recall": scm["char_recall"],
                "char_f1": scm["char_f1"], "char_accuracy": scm["char_accuracy"],
                "match": is_match, 
                "edit_distance": sed, "wer": swer, "cer": scer
            }
            
            # Cộng dồn thống kê cho Line Item
            li_accum["edit_distance"] += sed; li_accum["wer"] += swer; li_accum["cer"] += scer
            li_accum["char_precision"] += scm["char_precision"]; li_accum["char_recall"] += scm["char_recall"]
            li_accum["char_f1"] += scm["char_f1"]; li_accum["char_accuracy"] += scm["char_accuracy"]
            li_comps += 1
        item_details.append(detail)
        
    # C2. Xử lý các dòng GT bị bỏ sót (Unmatched GT - False Negatives)
    for i, g_item in enumerate(gt.line_items):
        if i not in used_g:
            detail = {"item_index": len(item_details)}
            for sf in sub_fields:
                gv = getattr(g_item, sf)
                # Với dòng thiếu, điểm số là 0, lỗi là tối đa
                val_len = len(gv)
                detail[sf] = {
                    "predicted": "N/A", "ground_truth": gv,
                    "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0,
                    "char_precision": 0.0, "char_recall": 0.0, "char_f1": 0.0, "char_accuracy": 0.0,
                    "match": False, "edit_distance": val_len, "wer": 1.0, "cer": 1.0
                }
                li_accum["edit_distance"] += val_len; li_accum["wer"] += 1.0; li_accum["cer"] += 1.0
                li_comps += 1
            item_details.append(detail)

    # Đóng gói kết quả Line Item
    li_result = {
        "index_accuracy": calculate_index_accuracy(gt.line_items, pred.line_items),
        "total_items": len(gt.line_items),
        "correct_items": tp,
        "precision": li_prec, "recall": li_rec, "f1_score": li_f1, "accuracy": li_acc,
        "edit_metrics": {
            "avg_edit_distance": li_accum["edit_distance"] / li_comps if li_comps else 0,
            "avg_wer": li_accum["wer"] / li_comps if li_comps else 0,
            "avg_cer": li_accum["cer"] / li_comps if li_comps else 0
        },
        "item_details": item_details
    }
    field_results["line_item"] = li_result

    # --- 3. Final Overall Calculation ---
    # Thêm điểm hệ thống của Line Item (detection) vào Overall
    ov_accum["precision"] += li_prec
    ov_accum["recall"] += li_rec
    ov_accum["f1_score"] += li_f1
    ov_accum["accuracy"] += li_acc
    
    # Thêm điểm nội dung (content) của Line Item vào Overall (Trung bình cộng)
    if li_comps > 0:
        ov_accum["char_precision"] += li_accum["char_precision"] / li_comps
        ov_accum["char_recall"] += li_accum["char_recall"] / li_comps
        ov_accum["char_f1"] += li_accum["char_f1"] / li_comps
        ov_accum["char_accuracy"] += li_accum["char_accuracy"] / li_comps
        ov_accum["edit_distance"] += li_accum["edit_distance"] / li_comps
        ov_accum["wer"] += li_accum["wer"] / li_comps
        ov_accum["cer"] += li_accum["cer"] / li_comps
        
    total_components += 1

    overall = {
        "index_accuracy": li_result["index_accuracy"],
        **{k: v / total_components for k, v in ov_accum.items()}
    }
    overall["avg_edit_distance"] = overall.pop("edit_distance")
    overall["avg_wer"] = overall.pop("wer")
    overall["avg_cer"] = overall.pop("cer")

    # Làm tròn toàn bộ kết quả trả về
    return recursive_round({
        "image_id": file_name.replace(".json", ""),
        "text_metrics": {
            "exact_match": exact_match(gt.raw_text, pred.raw_text),
            "normalized_edit_distance": normalized_levenshtein(gt.raw_text, pred.raw_text),
            "wer": word_error_rate(gt.raw_text, pred.raw_text),
            "cer": character_error_rate(gt.raw_text, pred.raw_text)
        },
        "field_metrics": field_results,
        "overall_image_score": overall
    }, 4)

def evaluate_dir(gt_dir: str, pred_dir: str) -> Dict[str, Any]:
    per_image_results = []
    files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    
    for fn in files:
        gt_path = os.path.join(gt_dir, fn)
        pred_path = os.path.join(pred_dir, fn)
        if not os.path.exists(gt_path): continue
        with open(gt_path, 'r', encoding='utf-8') as f: gt = f.read()
        with open(pred_path, 'r', encoding='utf-8') as f: pd = f.read()
        per_image_results.append(evaluate_pair(gt, pd, fn))

    if not per_image_results: return {}
    
    count = len(per_image_results)
    metric_keys = [
        "precision", "recall", "f1_score", "accuracy",
        "char_precision", "char_recall", "char_f1", "char_accuracy",
        "edit_distance", "wer", "cer"
    ]

    # --- AGGREGATION ---
    agg_overall = {k: 0.0 for k in metric_keys}
    agg_overall["index_accuracy"] = 0.0
    agg_overall["avg_edit_distance"] = 0.0; agg_overall["avg_wer"] = 0.0; agg_overall["avg_cer"] = 0.0
    
    for res in per_image_results:
        ov = res["overall_image_score"]
        agg_overall["index_accuracy"] += ov["index_accuracy"]
        agg_overall["precision"] += ov["precision"]
        agg_overall["recall"] += ov["recall"]
        agg_overall["f1_score"] += ov["f1_score"]
        agg_overall["accuracy"] += ov["accuracy"]
        agg_overall["char_precision"] += ov["char_precision"]
        agg_overall["char_recall"] += ov["char_recall"]
        agg_overall["char_f1"] += ov["char_f1"]
        agg_overall["char_accuracy"] += ov["char_accuracy"]
        agg_overall["avg_edit_distance"] += ov["avg_edit_distance"]
        agg_overall["avg_wer"] += ov["avg_wer"]
        agg_overall["avg_cer"] += ov["avg_cer"]

    field_names = ["retailer_name", "store_name", "store_address", "bill_id", "bill_id_barcode", "buy_date", "buy_time"]
    agg_fields = {fn: {k: 0.0 for k in metric_keys} for fn in field_names}
    
    agg_li_gen = {"precision": 0, "recall": 0, "f1_score": 0, "accuracy": 0, "index_accuracy": 0}
    
    sub_fields = ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]
    agg_li_sub = {sf: {k: 0.0 for k in metric_keys} for sf in sub_fields}

    for res in per_image_results:
        fm = res["field_metrics"]
        for fn in field_names:
            d = fm[fn]
            for k in metric_keys:
                if k in d: agg_fields[fn][k] += d[k]
            
        li = fm["line_item"]
        for k in agg_li_gen:
            if k in li: agg_li_gen[k] += li[k]
        
        temp_sub = {sf: {k: [] for k in metric_keys} for sf in sub_fields}
        for detail in li["item_details"]:
            for sf in sub_fields:
                if sf in detail:
                    d = detail[sf]
                    for k in metric_keys:
                        if k in d: temp_sub[sf][k].append(d[k])
        
        for sf in sub_fields:
            if temp_sub[sf]["f1_score"]:
                n = len(temp_sub[sf]["f1_score"])
                for k in metric_keys:
                    agg_li_sub[sf][k] += sum(temp_sub[sf][k]) / n

    summary = {
        "total_images": count,
        "overall": {k: v/count for k, v in agg_overall.items() if k not in ["edit_distance", "wer", "cer"]},
        "fields": {k: {m: v[m]/count for m in metric_keys} for k, v in agg_fields.items()},
        "line_item": {
            "general": {k: v/count for k, v in agg_li_gen.items()},
            "sub_fields": {k: {m: v[m]/count for m in metric_keys} for k, v in agg_li_sub.items()}
        }
    }

    final_result = {"per_image_results": per_image_results, "summary": summary}
    return recursive_round(final_result, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out", default="evaluation_output.json")
    args = parser.parse_args()

    if os.path.exists(args.gt_dir) and os.path.exists(args.pred_dir):
        result = evaluate_dir(args.gt_dir, args.pred_dir)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Calculation complete. Results saved to: {args.out}")
    else:
        print("❌ Error: Directory not found.")