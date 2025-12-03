import os
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import argparse

# Xóa zss nếu không dùng để tránh dependency thừa
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

# ---------- Data structures & Parsing helpers (GIỮ NGUYÊN) ----------
# ... (Giữ nguyên phần Class LineItem, InvoiceDoc và hàm parse_json_invoice của bạn) ...
# Để tiết kiệm không gian, tôi chỉ paste lại những phần thay đổi logic quan trọng bên dưới.

# [COPY LẠI PHẦN CLASS VÀ PARSE CỦA BẠN VÀO ĐÂY]

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
    # ... (Giữ nguyên code hàm này của bạn) ...
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return InvoiceDoc()
    
    retailer_name = data.get("retailer_name", "")
    store_name = data.get("store_name", "")
    store_address = data.get("store_address", "")
    bill_id = data.get("bill_id", "")
    bill_id_barcode = data.get("bill_id_barcode", "")
    buy_date = data.get("buy_date", "")
    buy_time = data.get("buy_time", "")
    
    line_items = []
    items_data = data.get("line_items", []) or data.get("line_item", [])
    
    for item_data in items_data:
        product_SKU = item_data.get("product_SKU", "") or item_data.get("sku", "")
        quantity = item_data.get("quantity", "") or item_data.get("qty", "")
        product_name = item_data.get("product_name", "") or item_data.get("description", "")
        unit_price = item_data.get("unit_price", "") or item_data.get("price", "")
        product_total = item_data.get("product_total", "") or item_data.get("line_total", "") or item_data.get("total", "")
        
        line_items.append(LineItem(
            product_SKU=str(product_SKU),
            quantity=str(quantity),
            product_name=str(product_name),
            unit_price=str(unit_price),
            product_total=str(product_total)
        ))
    
    return InvoiceDoc(
        retailer_name=str(retailer_name),
        store_name=str(store_name),
        store_address=str(store_address),
        bill_id=str(bill_id),
        bill_id_barcode=str(bill_id_barcode),
        buy_date=str(buy_date),
        buy_time=str(buy_time),
        line_items=line_items,
        raw_text=json_text
    )

# ---------- Metrics (text-level) (GIỮ NGUYÊN) ----------
# ... (Giữ nguyên các hàm metrics text-level) ...

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
    ref_str = ' '.join(ref_words)
    hyp_str = ' '.join(hyp_words)
    if not ref_str and not hyp_str: return 0.0
    word_distance = Levenshtein.distance(ref_str, hyp_str)
    max_len = max(len(ref_str), len(hyp_str))
    return word_distance / max_len if max_len > 0 else 0.0

def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis: return 0.0
    ref_chars = reference.replace(' ', '')
    hyp_chars = hypothesis.replace(' ', '')
    if not ref_chars and not hyp_chars: return 0.0
    char_distance = Levenshtein.distance(ref_chars, hyp_chars)
    max_len = max(len(ref_chars), len(hyp_chars))
    return char_distance / max_len if max_len > 0 else 0.0

def calculate_edit_metrics(gt_val: str, pred_val: str) -> Dict[str, Any]:
    gt_v = str(gt_val) if gt_val is not None else ""
    pred_v = str(pred_val) if pred_val is not None else ""
    return {
        "edit_distance": Levenshtein.distance(gt_v, pred_v),
        "normalized_edit_distance": normalized_levenshtein(gt_v, pred_v),
        "wer": word_error_rate(gt_v, pred_v),
        "cer": character_error_rate(gt_v, pred_v),
        "exact_match": exact_match(gt_v, pred_v)
    }

# ---------- Metrics (field-level) (GIỮ NGUYÊN) ----------
# ... (Giữ nguyên các hàm metrics field-level) ...

def token_based_accuracy(gt_val: str, pred_val: str) -> float:
    gt_tokens = set(re.findall(r"\w+", gt_val.lower()))
    pred_tokens = set(re.findall(r"\w+", pred_val.lower()))
    if not gt_tokens and not pred_tokens: return 1.0
    tp = len(gt_tokens & pred_tokens)
    total = len(gt_tokens | pred_tokens) # Sửa nhẹ: Jaccard = TP / Union
    return tp / total if total > 0 else 0.0

def token_set_metrics(gt_val: str, pred_val: str) -> Tuple[float, float, float]:
    gt_tokens = set(re.findall(r"\w+", gt_val.lower()))
    pred_tokens = set(re.findall(r"\w+", pred_val.lower()))
    if not gt_tokens and not pred_tokens: return 1.0, 1.0, 1.0
    tp = len(gt_tokens & pred_tokens)
    fp = len(pred_tokens - gt_tokens)
    fn = len(gt_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def field_metrics(gt_doc: InvoiceDoc, pred_doc: InvoiceDoc) -> Dict[str, Any]:
    # ... (Giữ nguyên logic hàm này của bạn, không thay đổi gì lớn) ...
    # Để code ngắn gọn tôi copy lại phần chính
    per_field = {}
    micro_tp = micro_fp = micro_fn = 0
    f1s = []
    accuracies = []
    
    total_metrics = {"edit_distance": 0, "norm_edit": 0.0, "wer": 0.0, "cer": 0.0}
    field_count = 0

    field_mappings = [
        ("retailer_name", gt_doc.retailer_name, pred_doc.retailer_name),
        ("store_name", gt_doc.store_name, pred_doc.store_name),
        ("store_address", gt_doc.store_address, pred_doc.store_address),
        ("bill_id", gt_doc.bill_id, pred_doc.bill_id),
        ("bill_id_barcode", gt_doc.bill_id_barcode, pred_doc.bill_id_barcode),
        ("buy_date", gt_doc.buy_date, pred_doc.buy_date),
        ("buy_time", gt_doc.buy_time, pred_doc.buy_time),
    ]
    
    for field_name, gt_val, pred_val in field_mappings:
        gt_v = str(gt_val) if gt_val is not None else ""
        pred_v = str(pred_val) if pred_val is not None else ""
        p, r, f1 = token_set_metrics(gt_v, pred_v)
        em = exact_match(gt_v, pred_v)
        edit_metrics = calculate_edit_metrics(gt_v, pred_v)
        accuracy = token_based_accuracy(gt_v, pred_v)
        
        per_field[field_name] = {
            "precision": p, "recall": r, "f1": f1, "accuracy": accuracy, "exact_match": em,
            **edit_metrics
        }
        
        gt_tokens = set(re.findall(r"\w+", gt_v.lower()))
        pred_tokens = set(re.findall(r"\w+", pred_v.lower()))
        micro_tp += len(gt_tokens & pred_tokens)
        micro_fp += len(pred_tokens - gt_tokens)
        micro_fn += len(gt_tokens - pred_tokens)
        f1s.append(f1)
        accuracies.append(accuracy)
        
        total_metrics["edit_distance"] += edit_metrics["edit_distance"]
        total_metrics["norm_edit"] += edit_metrics["normalized_edit_distance"]
        total_metrics["wer"] += edit_metrics["wer"]
        total_metrics["cer"] += edit_metrics["cer"]
        field_count += 1

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_acc = micro_tp / (micro_tp + micro_fp + micro_fn) if (micro_tp + micro_fp + micro_fn) > 0 else 0.0
    
    return {
        "per_field": per_field,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1, "accuracy": micro_acc},
        "macro": {"f1": sum(f1s)/len(f1s) if f1s else 0.0},
        "accuracy": sum(accuracies)/len(accuracies) if accuracies else 0.0,
        "edit_metrics": {
            "avg_edit_distance": total_metrics["edit_distance"] / field_count if field_count else 0,
            "avg_normalized_edit_distance": total_metrics["norm_edit"] / field_count if field_count else 0,
            "avg_wer": total_metrics["wer"] / field_count if field_count else 0,
            "avg_cer": total_metrics["cer"] / field_count if field_count else 0
        }
    }

# ---------- Line item metrics (SỬA LẠI LOGIC MATCHING) ----------

def line_item_field_metrics(gt_item: LineItem, pred_item: LineItem) -> Dict[str, Any]:
    # ... (Giữ nguyên) ...
    field_results = {}
    line_item_fields = [
        ("product_SKU", gt_item.product_SKU, pred_item.product_SKU),
        ("product_name", gt_item.product_name, pred_item.product_name),
        ("quantity", gt_item.quantity, pred_item.quantity),
        ("unit_price", gt_item.unit_price, pred_item.unit_price),
        ("product_total", gt_item.product_total, pred_item.product_total),
    ]
    for field_name, gt_val, pred_val in line_item_fields:
        gt_v = str(gt_val) if gt_val is not None else ""
        pred_v = str(pred_val) if pred_val is not None else ""
        p, r, f1 = token_set_metrics(gt_v, pred_v)
        em = exact_match(gt_v, pred_v)
        edit_metrics = calculate_edit_metrics(gt_v, pred_v)
        accuracy = token_based_accuracy(gt_v, pred_v)
        
        field_results[field_name] = {
            "precision": p, "recall": r, "f1": f1, "accuracy": accuracy, "exact_match": em,
            **edit_metrics
        }
    return field_results

def line_item_metrics(gt_items: List[LineItem], pred_items: List[LineItem]) -> Dict[str, Any]:
    # Handle empty cases (Giữ nguyên logic của bạn cho case rỗng)
    if not gt_items and not pred_items:
        # ... (Return empty match 1.0) ...
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
            "matched_count": 0, "unmatched_gt_count": 0, "unmatched_pred_count": 0,
            "per_field_metrics": {k: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0, "exact_match": True, "edit_distance": 0, "normalized_edit_distance": 0.0, "wer": 0.0, "cer": 0.0} for k in ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]},
            "field_accuracy_summary": {k: 1.0 for k in ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]},
            "edit_metrics": {"avg_edit_distance": 0.0, "avg_normalized_edit_distance": 0.0, "avg_wer": 0.0, "avg_cer": 0.0}
        }
    
    if not gt_items or not pred_items:
         # ... (Return mismatched 0.0) ...
         return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "matched_count": 0, "unmatched_gt_count": len(gt_items), "unmatched_pred_count": len(pred_items),
            "per_field_metrics": {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "exact_match": False, "edit_distance": 1, "normalized_edit_distance": 1.0, "wer": 1.0, "cer": 1.0} for k in ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]},
            "field_accuracy_summary": {k: 0.0 for k in ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]},
            "edit_metrics": {"avg_edit_distance": 1.0, "avg_normalized_edit_distance": 1.0, "avg_wer": 1.0, "avg_cer": 1.0}
        }

    # --- CẢI TIẾN LOGIC MATCHING ---
    # Tính tất cả các cặp điểm số có thể
    match_candidates = []
    for gt_idx, gt_item in enumerate(gt_items):
        for pred_idx, pred_item in enumerate(pred_items):
            score = fuzz.ratio(gt_item.product_name.lower(), pred_item.product_name.lower())
            if score > 70: # Threshold
                match_candidates.append((score, gt_idx, pred_idx))
    
    # Sort theo score giảm dần để ưu tiên cặp khớp nhất
    match_candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    matched_pairs = []
    
    for score, gt_idx, pred_idx in match_candidates:
        if gt_idx not in matched_gt_indices and pred_idx not in matched_pred_indices:
            matched_pairs.append((gt_items[gt_idx], pred_items[pred_idx]))
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            
    unmatched_gt = [item for i, item in enumerate(gt_items) if i not in matched_gt_indices]
    unmatched_pred = [item for i, item in enumerate(pred_items) if i not in matched_pred_indices]
    
    # ... (Phần tính toán metrics sau khi match giữ nguyên như code cũ của bạn) ...
    # Copy lại logic tính toán metrics từ matched_pairs của bạn
    
    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / len(gt_items) if len(gt_items) > 0 else 0.0

    field_metrics_aggregated = {k: {"precision": [], "recall": [], "f1": [], "accuracy": [], "edit_distance": [], "normalized_edit_distance": [], "wer": [], "cer": []} for k in ["product_SKU", "product_name", "quantity", "unit_price", "product_total"]}
    
    total_metrics = {"edit_distance": 0, "norm_edit": 0.0, "wer": 0.0, "cer": 0.0}
    total_comparisons = 0
    
    for gt_item, pred_item in matched_pairs:
        item_metrics = line_item_field_metrics(gt_item, pred_item)
        for fname, metrics in item_metrics.items():
            for mNAME in ["precision", "recall", "f1", "accuracy", "edit_distance", "normalized_edit_distance", "wer", "cer"]:
                field_metrics_aggregated[fname][mNAME].append(metrics[mNAME])
            
            total_metrics["edit_distance"] += metrics["edit_distance"]
            total_metrics["norm_edit"] += metrics["normalized_edit_distance"]
            total_metrics["wer"] += metrics["wer"]
            total_metrics["cer"] += metrics["cer"]
            total_comparisons += 1
            
    per_field_metrics = {}
    field_accuracy_summary = {}
    for fname, metrics_dict in field_metrics_aggregated.items():
        per_field_metrics[fname] = {}
        for mNAME, values in metrics_dict.items():
            per_field_metrics[fname][mNAME] = sum(values)/len(values) if values else 0.0
        field_accuracy_summary[fname] = per_field_metrics[fname]["accuracy"]
        
    avg_edit_metrics = {
        "avg_edit_distance": total_metrics["edit_distance"] / total_comparisons if total_comparisons else 0.0,
        "avg_normalized_edit_distance": total_metrics["norm_edit"] / total_comparisons if total_comparisons else 0.0,
        "avg_wer": total_metrics["wer"] / total_comparisons if total_comparisons else 0.0,
        "avg_cer": total_metrics["cer"] / total_comparisons if total_comparisons else 0.0
    }
    
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "per_field_metrics": per_field_metrics, "field_accuracy_summary": field_accuracy_summary,
        "matched_count": len(matched_pairs), "unmatched_gt_count": len(unmatched_gt), "unmatched_pred_count": len(unmatched_pred),
        "edit_metrics": avg_edit_metrics
    }

# ---------- Formatting helpers (GIỮ NGUYÊN) ----------
# ... (Giữ nguyên format_field_metrics_result) ...

def format_field_metrics_result(field_metrics_result, gt_doc, pred_doc):
    # Copy từ code cũ
    formatted_result = {}
    field_mappings = [
        ("retailer_name", gt_doc.retailer_name, pred_doc.retailer_name),
        ("store_name", gt_doc.store_name, pred_doc.store_name),
        ("store_address", gt_doc.store_address, pred_doc.store_address),
        ("bill_id", gt_doc.bill_id, pred_doc.bill_id),
        ("bill_id_barcode", gt_doc.bill_id_barcode, pred_doc.bill_id_barcode),
        ("buy_date", gt_doc.buy_date, pred_doc.buy_date),
        ("buy_time", gt_doc.buy_time, pred_doc.buy_time),
    ]
    for field_name, gt_val, pred_val in field_mappings:
        gt_v = str(gt_val) if gt_val is not None else ""
        pred_v = str(pred_val) if pred_val is not None else ""
        field_data = field_metrics_result["per_field"].get(field_name, {})
        formatted_result[field_name] = {
            "precision": round(field_data.get("precision", 0.0), 4),
            "recall": round(field_data.get("recall", 0.0), 4),
            "f1_score": round(field_data.get("f1", 0.0), 4),
            "accuracy": round(field_data.get("accuracy", 0.0), 4),
            "predicted": pred_v, "ground_truth": gt_v, "match": exact_match(gt_v, pred_v),
            "edit_distance": field_data.get("edit_distance", 0),
            "normalized_edit_distance": round(field_data.get("normalized_edit_distance", 0.0), 4),
            "wer": round(field_data.get("wer", 0.0), 4),
            "cer": round(field_data.get("cer", 0.0), 4)
        }
    return formatted_result

def format_line_item_result(line_items_result, gt_items, pred_items):
    # Cần update logic matching ở đây để đồng bộ với hàm line_item_metrics
    # Logic matching:
    match_candidates = []
    for gt_idx, gt_item in enumerate(gt_items):
        for pred_idx, pred_item in enumerate(pred_items):
            score = fuzz.ratio(gt_item.product_name.lower(), pred_item.product_name.lower())
            if score > 70:
                match_candidates.append((score, gt_idx, pred_idx))
    match_candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    matched_pairs = []
    
    for score, gt_idx, pred_idx in match_candidates:
        if gt_idx not in matched_gt_indices and pred_idx not in matched_pred_indices:
            matched_pairs.append((gt_items[gt_idx], pred_items[pred_idx]))
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            
    unmatched_gt = [item for i, item in enumerate(gt_items) if i not in matched_gt_indices]
    
    item_details = []
    for idx, (gt_item, pred_item) in enumerate(matched_pairs):
        item_field_metrics = line_item_field_metrics(gt_item, pred_item)
        item_detail = {"item_index": idx}
        for fname in ["product_SKU", "quantity", "product_name", "unit_price", "product_total"]:
            m = item_field_metrics[fname]
            item_detail[fname] = {
                "precision": round(m["precision"], 4), "recall": round(m["recall"], 4), "f1_score": round(m["f1"], 4), "accuracy": round(m["accuracy"], 4),
                "match": exact_match(getattr(gt_item, fname), getattr(pred_item, fname)),
                "edit_distance": m["edit_distance"], "normalized_edit_distance": round(m["normalized_edit_distance"], 4),
                "wer": round(m["wer"], 4), "cer": round(m["cer"], 4)
            }
        item_details.append(item_detail)

    for idx, gt_item in enumerate(unmatched_gt, start=len(matched_pairs)):
        item_detail = {"item_index": idx}
        for fname in ["product_SKU", "quantity", "product_name", "unit_price", "product_total"]:
            val_len = len(str(getattr(gt_item, fname)))
            item_detail[fname] = {
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0,
                "match": False, "edit_distance": val_len, "normalized_edit_distance": 1.0, "wer": 1.0, "cer": 1.0
            }
        item_details.append(item_detail)
        
    return {
        "total_items": len(gt_items), "correct_items": len(matched_pairs),
        "precision": round(line_items_result.get("precision", 0.0), 4),
        "recall": round(line_items_result.get("recall", 0.0), 4),
        "f1_score": round(line_items_result.get("f1", 0.0), 4),
        "accuracy": round(line_items_result.get("accuracy", 0.0), 4),
        "edit_metrics": {
            "avg_edit_distance": round(line_items_result.get("edit_metrics", {}).get("avg_edit_distance", 0.0), 4),
            "avg_normalized_edit_distance": round(line_items_result.get("edit_metrics", {}).get("avg_normalized_edit_distance", 0.0), 4),
            "avg_wer": round(line_items_result.get("edit_metrics", {}).get("avg_wer", 0.0), 4),
            "avg_cer": round(line_items_result.get("edit_metrics", {}).get("avg_cer", 0.0), 4)
        },
        "item_details": item_details
    }

# ---------- End-to-end evaluation & MAIN FIXED FUNCTION ----------

def evaluate_pair(gt_json: str, pred_json: str, file_name: str) -> Dict[str, Any]:
    # ... (Giữ nguyên) ...
    gt_doc = parse_json_invoice(gt_json)
    pred_doc = parse_json_invoice(pred_json)
    
    text_ed = normalized_levenshtein(gt_doc.raw_text, pred_doc.raw_text)
    text_em = exact_match(gt_doc.raw_text, pred_doc.raw_text)
    
    fields_res = field_metrics(gt_doc, pred_doc)
    line_items_res = line_item_metrics(gt_doc.line_items, pred_doc.line_items)
    
    field_metrics_formatted = format_field_metrics_result(fields_res, gt_doc, pred_doc)
    line_items_formatted = format_line_item_result(line_items_res, gt_doc.line_items, pred_doc.line_items)
    
    field_acc = fields_res.get("accuracy", 0.0)
    line_item_acc = line_items_res.get("accuracy", 0.0)
    field_weight = 0.6
    line_item_weight = 0.4
    overall_accuracy = (field_acc * field_weight + line_item_acc * line_item_weight)
    
    overall_precision = (fields_res["micro"]["precision"] * field_weight + line_items_res["precision"] * line_item_weight)
    overall_recall = (fields_res["micro"]["recall"] * field_weight + line_items_res["recall"] * line_item_weight)
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    overall_edit_distance = (fields_res["edit_metrics"]["avg_edit_distance"] * field_weight + line_items_res["edit_metrics"]["avg_edit_distance"] * line_item_weight)
    overall_norm_edit_distance = (fields_res["edit_metrics"]["avg_normalized_edit_distance"] * field_weight + line_items_res["edit_metrics"]["avg_normalized_edit_distance"] * line_item_weight)
    overall_wer = (fields_res["edit_metrics"]["avg_wer"] * field_weight + line_items_res["edit_metrics"]["avg_wer"] * line_item_weight)
    overall_cer = (fields_res["edit_metrics"]["avg_cer"] * field_weight + line_items_res["edit_metrics"]["avg_cer"] * line_item_weight)

    return {
        "image_id": file_name.replace(".json", ""),
        "text_metrics": {
            "exact_match": text_em,
            "normalized_edit_distance": round(text_ed, 4),
            "wer": round(word_error_rate(gt_doc.raw_text, pred_doc.raw_text), 4),
            "cer": round(character_error_rate(gt_doc.raw_text, pred_doc.raw_text), 4)
        },
        "field_metrics": {**field_metrics_formatted, "line_item": line_items_formatted},
        "overall_image_score": {
            "precision": round(overall_precision, 4), "recall": round(overall_recall, 4), "f1_score": round(overall_f1, 4), "accuracy": round(overall_accuracy, 4),
            "avg_edit_distance": round(overall_edit_distance, 4), "avg_normalized_edit_distance": round(overall_norm_edit_distance, 4), "avg_wer": round(overall_wer, 4), "avg_cer": round(overall_cer, 4)
        }
    }

def load_json_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def evaluate_dir(gt_dir: str, pred_dir: str) -> Dict[str, Any]:
    per_image_results = []
    
    for fn in os.listdir(pred_dir):
        if not fn.endswith(".json"): continue
        gt_path = os.path.join(gt_dir, fn)
        pred_path = os.path.join(pred_dir, fn)
        if not os.path.exists(gt_path): continue
            
        gt_json = load_json_file(gt_path)
        pred_json = load_json_file(pred_path)
        per_image_results.append(evaluate_pair(gt_json, pred_json, fn))
    
    if not per_image_results:
        return {"per_image_results": [], "summary": {}}
    
    total_images = len(per_image_results)
    
    # --- SỬA LẠI LOGIC AGGREGATION ---
    
    # Init accumulators
    text_metrics_summary = {"exact_match_rate": 0.0, "avg_normalized_edit_distance": 0.0, "avg_wer": 0.0, "avg_cer": 0.0}
    field_names = ["retailer_name", "store_name", "store_address", "bill_id", "bill_id_barcode", "buy_date", "buy_time"]
    field_level_summary = {fn: {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1_score": 0.0, "avg_accuracy": 0.0, "avg_edit_distance": 0.0, "avg_normalized_edit_distance": 0.0, "avg_wer": 0.0, "avg_cer": 0.0, "total_correct": 0, "total_samples": total_images} for fn in field_names}
    
    line_item_summary = {
        "avg_precision": 0.0, "avg_recall": 0.0, "avg_f1_score": 0.0, "avg_accuracy": 0.0, "avg_edit_distance": 0.0, "avg_normalized_edit_distance": 0.0, "avg_wer": 0.0, "avg_cer": 0.0,
        "sub_fields": {sf: {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1_score": 0.0, "avg_accuracy": 0.0, "avg_edit_distance": 0.0, "avg_normalized_edit_distance": 0.0, "avg_wer": 0.0, "avg_cer": 0.0} for sf in ["product_SKU", "quantity", "product_name", "unit_price", "product_total"]}
    }
    
    overall_accs = {"precision": 0, "recall": 0, "f1_score": 0, "accuracy": 0, "avg_edit_distance": 0, "avg_normalized_edit_distance": 0, "avg_wer": 0, "avg_cer": 0}

    for result in per_image_results:
        # Text metrics
        tm = result["text_metrics"]
        text_metrics_summary["exact_match_rate"] += 1.0 if tm["exact_match"] else 0.0
        text_metrics_summary["avg_normalized_edit_distance"] += tm["normalized_edit_distance"]
        text_metrics_summary["avg_wer"] += tm["wer"]
        text_metrics_summary["avg_cer"] += tm["cer"]
        
        # Field metrics
        for fn in field_names:
            if fn in result["field_metrics"]:
                fd = result["field_metrics"][fn]
                for k in ["precision", "recall", "f1_score", "accuracy", "edit_distance", "normalized_edit_distance", "wer", "cer"]:
                    field_level_summary[fn][f"avg_{k}" if k != "edit_distance" and "avg" not in k else f"avg_{k}"] += fd[k]
                if fd["match"]: field_level_summary[fn]["total_correct"] += 1
        
        # Line item metrics (General)
        lid = result["field_metrics"]["line_item"]
        for k in ["precision", "recall", "f1_score", "accuracy"]:
            line_item_summary[f"avg_{k}"] += lid[k]
        line_item_summary["avg_edit_distance"] += lid["edit_metrics"]["avg_edit_distance"]
        line_item_summary["avg_normalized_edit_distance"] += lid["edit_metrics"]["avg_normalized_edit_distance"]
        line_item_summary["avg_wer"] += lid["edit_metrics"]["avg_wer"]
        line_item_summary["avg_cer"] += lid["edit_metrics"]["avg_cer"]
        
        # Line item subfields (Correct aggregation logic)
        # B1: Tính trung bình cho ẢNH HIỆN TẠI
        current_img_subfields = {sf: {m: [] for m in ["precision", "recall", "f1_score", "accuracy", "edit_distance", "normalized_edit_distance", "wer", "cer"]} for sf in line_item_summary["sub_fields"]}
        
        for item_detail in lid["item_details"]:
            for sf in line_item_summary["sub_fields"]:
                if sf in item_detail:
                    for m in current_img_subfields[sf]:
                        current_img_subfields[sf][m].append(item_detail[sf][m])
        
        # B2: Cộng trung bình của ảnh hiện tại vào tổng tích lũy
        for sf in line_item_summary["sub_fields"]:
            for m in current_img_subfields[sf]:
                vals = current_img_subfields[sf][m]
                if vals:
                    avg_val = sum(vals) / len(vals)
                    line_item_summary["sub_fields"][sf][f"avg_{m}"] += avg_val
                # Nếu ảnh không có item nào, coi như 0 (hoặc logic khác tùy bạn, ở đây giữ 0)

        # Overall
        ov = result["overall_image_score"]
        for k in overall_accs:
            overall_accs[k] += ov[k]

    # Calculate final averages (Chia cho tổng số ảnh)
    for k in text_metrics_summary: text_metrics_summary[k] /= total_images
    
    formatted_field_summary = {}
    for fn in field_names:
        for k in field_level_summary[fn]:
             if "avg" in k: field_level_summary[fn][k] /= total_images
        formatted_field_summary[fn] = {k: round(v, 4) for k, v in field_level_summary[fn].items()}
        
    for k in line_item_summary:
        if isinstance(line_item_summary[k], float):
            line_item_summary[k] /= total_images
    
    formatted_line_item_summary = {k: round(v, 4) for k, v in line_item_summary.items() if isinstance(v, float)}
    formatted_line_item_summary["sub_fields"] = {}
    
    for sf in line_item_summary["sub_fields"]:
        formatted_line_item_summary["sub_fields"][sf] = {}
        for k in line_item_summary["sub_fields"][sf]:
            line_item_summary["sub_fields"][sf][k] /= total_images
            formatted_line_item_summary["sub_fields"][sf][k] = round(line_item_summary["sub_fields"][sf][k], 4)
            
    overall_summary = {k: round(v / total_images, 4) for k, v in overall_accs.items()}
    
    return {
        "per_image_results": per_image_results,
        "summary": {
            "total_images": total_images,
            "text_metrics_summary": {k: round(v, 4) for k,v in text_metrics_summary.items()},
            "field_level_summary": formatted_field_summary,
            "line_item": formatted_line_item_summary,
            "overall_summary": overall_summary
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    report = evaluate_dir(args.gt_dir, args.pred_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {args.out}")