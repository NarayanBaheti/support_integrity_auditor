import json
import re
import numpy as np

def extract_attention_attributions(attentions, tokenizer, text, max_len=64):
    """
    Extracts attention weights from the CLS token to all other tokens
    from the last layer, averaging across all attention heads.
    """
    # attentions shape from ONNX: (1, num_heads, seq_len, seq_len)
    if attentions is None or len(attentions.shape) != 4:
        return []
        
    num_heads = attentions.shape[1]
    seq_len = attentions.shape[2]
    
    # Tokenize text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Extract CLS token attention to other tokens: attentions[0, :, 0, :] (shape: num_heads, seq_len)
    cls_attention = attentions[0, :, 0, :]
    mean_attention = np.mean(cls_attention, axis=0) # shape: (seq_len,)
    
    # Map tokens to their attention weights, filtering out special tokens
    attributions = []
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
    
    for idx, (token, weight) in enumerate(zip(tokens, mean_attention)):
        if token not in special_tokens and not token.startswith('[') and not token.startswith('<'):
            clean_token = token.replace('\u2581', '').replace('##', '').strip()
            if len(clean_token) > 1:
                attributions.append({
                    'token': clean_token,
                    'weight': float(weight)
                })
            
    # Sort by weight descending
    attributions = sorted(attributions, key=lambda x: x['weight'], reverse=True)
    return attributions[:8] # Return top 8 tokens

def build_evidence_dossier(ticket_row, inferred_severity, mismatch_type, severity_delta, confidence_score, attributions, mode='audit'):
    """
    Constructs the structured evidence dossier according to the exact required schema.
    """
    assigned = ticket_row.get('Priority_Level', 'UNKNOWN')
    
    # 1. Keyword Evidence Extraction (Grounded in text)
    evidence = []
    subject = str(ticket_row.get('Ticket_Subject', '')).lower()
    description = str(ticket_row.get('Ticket_Description', '')).lower()
    full_text = subject + ' ' + description
    
    high_priority_keywords = {
        'production down': '4', 'service unavailable': '4', 'outage': '4',
        'lost revenue': '3.5', 'customer churn': '3.5', 'security incident': '3.5',
        'data loss': '3.5', 'breach': '3.5', 'all users affected': '3.0',
        'payment failed': '3.0', 'account locked': '3.0', 'urgent': '2.0',
        'unable to login': '2.0', 'crash': '2.5', 'failed': '1.5', 'broken': '1.5'
    }
    
    # Find exact matching keywords present in raw text
    for kw, weight in high_priority_keywords.items():
        if re.search(rf'\b{kw}', full_text):
            evidence.append({
                "signal": "keyword",
                "value": kw,
                "weight": weight
            })
            
    # 2. Resolution Time Evidence (Grounded in metadata)
    res_hours = ticket_row.get('Resolution_Time_Hours', None)
    if mode != 'real_time' and res_hours is not None:
        evidence.append({
            "signal": "resolution_time",
            "value": f"{res_hours} hours",
            "interpretation": "historically associated with higher severity ticket classification" if res_hours > 48 else "historically aligned with low/medium severity tickets"
        })
        
    # 3. Add top attention token attribution as evidence
    if attributions:
        top_token = attributions[0]['token'].replace(' ', '') # Clean subwords
        if len(top_token) > 2:
            evidence.append({
                "signal": "attention_token",
                "value": top_token,
                "weight": float(attributions[0]['weight'])
            })
            
    # 4. Generate grounded constraint analysis
    if mismatch_type == 'Hidden Crisis':
        analysis = (
            f"Ticket reported critical characteristics (severity level inferred as '{inferred_severity}') "
            f"but was triaged with a '{assigned}' priority level. Key indicators include keyword cues and metadata flags "
            f"indicating high urgency that requires immediate SLA escalation."
        )
    elif mismatch_type == 'False Alarm':
        analysis = (
            f"Ticket was assigned a high priority level of '{assigned}' but its content details general informational "
            f"queries (severity level inferred as '{inferred_severity}'). Re-triaging to a lower priority is recommended "
            f"to prevent resource waste."
        )
    else:
        analysis = f"Assigned priority matches inferred severity level of '{inferred_severity}'."
        
    dossier = {
        "ticket_id": str(ticket_row.get('Ticket_ID', 'UNKNOWN')),
        "assigned_priority": str(assigned),
        "inferred_severity": str(inferred_severity),
        "mismatch_type": str(mismatch_type),
        "severity_delta": int(severity_delta),
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
        "confidence": float(confidence_score)
    }
    
    return dossier

def validate_dossier(dossier, ticket_raw):
    """
    Performs strict validation checks to ensure zero-hallucination.
    Every feature evidence item value must map back to a raw ticket field.
    """
    subject = str(ticket_raw.get('Ticket_Subject', '')).lower()
    description = str(ticket_raw.get('Ticket_Description', '')).lower()
    full_text = subject + ' ' + description
    
    evidence_items = dossier.get("feature_evidence", [])
    
    for item in evidence_items:
        sig_type = item.get("signal")
        val = str(item.get("value")).lower()
        
        # Validate based on signal type
        if sig_type == "keyword" or sig_type == "attention_token":
            # Check if keyword exists in full text
            # We strip common token markers like ##
            clean_val = val.replace('##', '').replace('\u2581', '')
            if clean_val not in full_text:
                print(f"Validation Failure: Evidence value '{clean_val}' not found in raw ticket text.")
                return False
                
        elif sig_type == "resolution_time":
            # Check if resolution time matches actual hours
            raw_res = str(ticket_raw.get('Resolution_Time_Hours', ''))
            if raw_res not in val:
                print(f"Validation Failure: Resolution time '{val}' does not match raw value '{raw_res}'.")
                return False
                
        elif sig_type == "channel":
            raw_chan = str(ticket_raw.get('Ticket_Channel', '')).lower()
            if raw_chan not in val:
                print(f"Validation Failure: Channel '{val}' does not match raw channel '{raw_chan}'.")
                return False
                
    return True
