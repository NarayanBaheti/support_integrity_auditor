import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from transformers import AutoTokenizer

# Import custom src modules
from src.evidence_generator import (
    extract_attention_attributions,
    build_evidence_dossier,
    validate_dossier
)
from src.pseudo_labeling import score_to_severity_label

# Load configurations
PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
REVERSE_PRIORITY_MAP = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

def softmax(logits):
    """
    Computes softmax probabilities for classification logits.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def run_onnx_inference(ticket_df, mode='audit', classifier_onnx='models/classifier/model_quantized.onnx', session=None, tokenizer=None):
    """
    Runs inference on the ticket dataframe using the DeBERTa quantized classifier in ONNX Runtime in a vectorized batch.
    """
    if session is None or tokenizer is None:
        if not os.path.exists(classifier_onnx):
            raise FileNotFoundError(f"Quantized classifier ONNX model not found at {classifier_onnx}. Run train_pipeline.py first.")
        
    if session is None:
        print(f"Loading quantized classifier ONNX model from {classifier_onnx}...")
        session = ort.InferenceSession(classifier_onnx, providers=['CPUExecutionProvider'])
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    
    # Load metadata encoders and scalers
    encoders = joblib.load('models/metadata_encoders.joblib')
    scaler = joblib.load('models/resolution_scaler.joblib')
    
    # Load calibrated decision threshold if available, else fallback to 0.55
    threshold_path = 'models/decision_threshold.joblib'
    if os.path.exists(threshold_path):
        try:
            threshold = joblib.load(threshold_path)
        except Exception:
            threshold = 0.55
    else:
        threshold = 0.55
    
    # Process text for all rows at once
    combined_texts = []
    for idx, row in ticket_df.iterrows():
        combined_text = str(row.get('Ticket_Subject', '')) + " " + str(row.get('Ticket_Description', ''))
        combined_texts.append(combined_text)
        
    inputs = tokenizer(
        combined_texts, 
        padding='max_length', 
        truncation=True, 
        max_length=64, 
        return_tensors="np"
    )
    
    # Process categoricals in vectorized way
    cat_columns = ['Issue_Category', 'Ticket_Channel', 'customer_tier', 'Priority_Level']
    X_cat_list = []
    for col in cat_columns:
        mapping = encoders[col]
        # Map values, defaulting to 0 for unknown values
        mapped_col = ticket_df[col].fillna('unknown').astype(str).map(lambda x: mapping.get(x, mapping.get('unknown', 0))).values
        X_cat_list.append(mapped_col)
    X_cat = np.stack(X_cat_list, axis=1).astype(np.int64)
    
    # Process numericals in vectorized way
    if mode == 'real_time':
        X_num = np.zeros((len(ticket_df), 1), dtype=np.float32)
    else:
        res_hours = ticket_df['Resolution_Time_Hours'].fillna(0.0).values
        log_res = np.log1p(res_hours)
        scaled_res = (log_res - scaler['mean']) / (scaler['std'] + 1e-5)
        X_num = scaled_res.reshape(-1, 1).astype(np.float32)
        
    # Prepare ONNX Inputs
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64),
        'cat_features': X_cat,
        'num_features': X_num
    }
    
    # Run session for the entire batch
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]
    attentions = ort_outs[1]
    
    # Softmax probabilities and predictions
    probs = softmax(logits)
    pred_classes = (probs[:, 1] >= threshold).astype(np.int64)
    confidences = probs[:, 1]
    
    return pred_classes, confidences, attentions, tokenizer

def predict_single_ticket(ticket_dict, mode='audit', session=None, tokenizer=None):
    """
    Runs inference on a single ticket, generates and validates evidence dossier.
    """
    # Create single-row dataframe
    df_ticket = pd.DataFrame([ticket_dict])
    
    # Add domain and customer tier columns
    email = str(ticket_dict.get('Customer_Email', ''))
    domain = email.split('@')[-1] if '@' in email else 'unknown'
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com', 'example.org', 'example.net']
    df_ticket['customer_tier'] = 'Standard' if domain in free_domains or domain == 'unknown' else 'Enterprise'
    
    # Run ONNX inference
    pred_classes, confidences, attentions, tokenizer = run_onnx_inference(df_ticket, mode=mode, session=session, tokenizer=tokenizer)
    
    pred_class = pred_classes[0]
    confidence = confidences[0]
    
    dossier = None
    if pred_class == 1:
        # Generate inferred severity from text/meta using a temporary fallback
        # Let's map it based on simple rules or surrogate outputs
        # (For single ticket display we run a simple heuristic fusion of rules + resolution time)
        # Check rule-based score
        from src.severity_engine import compute_signal_d_rules
        rule_score = compute_signal_d_rules(df_ticket)[0]
        
        # We estimate inferred severity category
        # Since LLM is not loaded, we use rule-based + priority delta mappings
        assigned = ticket_dict.get('Priority_Level', 'Medium')
        assigned_num = PRIORITY_MAP[assigned]
        
        # If mismatch detected, inferred severity is shifted
        # We check details to classify Mismatch Type
        text = (str(ticket_dict.get('Ticket_Subject', '')) + " " + str(ticket_dict.get('Ticket_Description', ''))).lower()
        critical_keywords = ['outage', 'crash', 'down', 'offline', 'breach', 'security', 'hack', 'data loss']
        
        is_critical_indicators = any(kw in text for kw in critical_keywords)
        
        if assigned_num <= 1 and is_critical_indicators:
            inferred_severity = 'Critical'
            mismatch_type = 'Hidden Crisis'
            severity_delta = 3 - assigned_num
        elif assigned_num >= 2 and not is_critical_indicators:
            inferred_severity = 'Low'
            mismatch_type = 'False Alarm'
            severity_delta = 0 - assigned_num
        else:
            # Shift by 1
            if rule_score > 2.5:
                inferred_severity = 'Critical' if assigned_num < 3 else 'Critical'
                mismatch_type = 'Hidden Crisis'
                severity_delta = 1
            else:
                inferred_severity = 'Low' if assigned_num > 0 else 'Low'
                mismatch_type = 'False Alarm'
                severity_delta = -1
                
        # Extract attention attributions
        text_full = str(ticket_dict.get('Ticket_Subject', '')) + " " + str(ticket_dict.get('Ticket_Description', ''))
        attributions = extract_attention_attributions(attentions, tokenizer, text_full)
        
        # Build dossier
        dossier = build_evidence_dossier(
            ticket_dict, 
            inferred_severity, 
            mismatch_type, 
            severity_delta, 
            confidence, 
            attributions,
            mode=mode
        )
        
        # Validate dossier grounding
        is_valid = validate_dossier(dossier, ticket_dict)
        if not is_valid:
            print("WARNING: Evidence dossier failed grounding validation checks!")
            dossier['validation_status'] = 'Failed Grounding'
        else:
            dossier['validation_status'] = 'Passed Grounding'
            
    return pred_class, confidence, dossier

def predict_batch_csv(csv_path, output_path, mode='audit', session=None, tokenizer=None):
    """
    Processes a batch CSV file of support tickets, flags mismatches, and outputs results.
    """
    df_batch = pd.read_csv(csv_path)
    print(f"Loaded batch file: {len(df_batch)} rows.")
    
    # Ensure necessary columns are present
    required_cols = ['Ticket_ID', 'Ticket_Subject', 'Ticket_Description', 'Customer_Email', 'Priority_Level', 'Ticket_Channel']
    for col in required_cols:
        if col not in df_batch.columns:
            raise KeyError(f"Required column '{col}' is missing from the input CSV.")
            
    # Add domain and customer tier columns
    email_domains = df_batch['Customer_Email'].fillna('').apply(lambda x: x.split('@')[-1] if '@' in str(x) else 'unknown')
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com', 'example.org', 'example.net']
    df_batch['customer_tier'] = email_domains.apply(lambda d: 'Standard' if d in free_domains or d == 'unknown' else 'Enterprise')
    
    # Run ONNX inference
    pred_classes, confidences, attentions, tokenizer = run_onnx_inference(df_batch, mode=mode, session=session, tokenizer=tokenizer)
    
    df_batch['Priority_Mismatch_Pred'] = pred_classes
    df_batch['Confidence'] = confidences
    
    # Generate dossiers for all flagged rows
    dossiers_list = []
    
    for idx, row in df_batch.iterrows():
        if row['Priority_Mismatch_Pred'] == 1:
            ticket_dict = row.to_dict()
            text = (str(row['Ticket_Subject']) + " " + str(row['Ticket_Description'])).lower()
            critical_keywords = ['outage', 'crash', 'down', 'offline', 'breach', 'security', 'hack', 'data loss']
            is_critical = any(kw in text for kw in critical_keywords)
            assigned_num = PRIORITY_MAP.get(row['Priority_Level'], 1)
            
            if assigned_num <= 1 and is_critical:
                inferred_severity = 'Critical'
                mismatch_type = 'Hidden Crisis'
                severity_delta = 3 - assigned_num
            elif assigned_num >= 2 and not is_critical:
                inferred_severity = 'Low'
                mismatch_type = 'False Alarm'
                severity_delta = 0 - assigned_num
            else:
                inferred_severity = 'Critical' if assigned_num < 2 else 'Low'
                mismatch_type = 'Hidden Crisis' if assigned_num < 2 else 'False Alarm'
                severity_delta = 1 if assigned_num < 2 else -1
                
            # Extract individual attention slice for batch item: shape (1, num_heads, seq_len, seq_len)
            item_attention = attentions[idx:idx+1]
            text_full = str(row['Ticket_Subject']) + " " + str(row['Ticket_Description'])
            attributions = extract_attention_attributions(item_attention, tokenizer, text_full)
            
            dossier = build_evidence_dossier(
                ticket_dict, 
                inferred_severity, 
                mismatch_type, 
                severity_delta, 
                row['Confidence'], 
                attributions,
                mode=mode
            )
            
            # Grounding check
            if validate_dossier(dossier, ticket_dict):
                dossier['validation_status'] = 'Passed Grounding'
            else:
                dossier['validation_status'] = 'Failed Grounding'
                
            dossiers_list.append(dossier)
            
            # Save dossier file to disk
            t_id = dossier['ticket_id']
            dossier_path = f"dossiers/dossier_{t_id}.json"
            os.makedirs(os.path.dirname(dossier_path), exist_ok=True)
            with open(dossier_path, 'w') as f:
                json.dump(dossier, f, indent=4)
                
    # Save output CSV
    df_batch.to_csv(output_path, index=False)
    print(f"Successfully processed batch. Predictions outputted to {output_path}")
    print(f"Flagged {len(dossiers_list)} priority mismatches. Dossiers written to dossiers/ directory.")
    
    return df_batch

def main():
    parser = argparse.ArgumentParser(description="SIA Inference Script (ONNX Runtime)")
    parser.add_argument('--ticket', type=str, help="Path to single ticket JSON file")
    parser.add_argument('--csv', type=str, help="Path to batch CSV file")
    parser.add_argument('--output', type=str, default="data/predictions_batch_output.csv", help="Output path for batch predictions CSV")
    parser.add_argument('--mode', type=str, choices=['audit', 'real_time'], default='audit', help="leakage prevention mode: audit (uses resolution time) or real_time (excludes resolution time)")
    
    args = parser.parse_args()
    
    if args.ticket:
        if not os.path.exists(args.ticket):
            print(f"Error: Ticket JSON file not found at {args.ticket}")
            return
            
        with open(args.ticket, 'r') as f:
            ticket = json.load(f)
            
        pred_class, conf, dossier = predict_single_ticket(ticket, mode=args.mode)
        print("\n==========================================")
        print("SIA SINGLE TICKET AUDIT RESULTS")
        print("==========================================")
        print(f"Priority Mismatch Detected: {bool(pred_class)}")
        print(f"Classifier Softmax Confidence: {conf:.4f}")
        
        if dossier:
            print("\n--- Validated Evidence Dossier ---")
            print(json.dumps(dossier, indent=4))
        print("==========================================")
        
    elif args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: Batch CSV file not found at {args.csv}")
            return
            
        predict_batch_csv(args.csv, args.output, mode=args.mode)
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
