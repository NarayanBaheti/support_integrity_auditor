import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
REVERSE_PRIORITY_MAP = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

def fuse_severity_signals(sig_a, sig_b, sig_c, sig_d):
    """
    Fuses 4 independent severity signals using the fixed weighted consensus formula:
    0.40 LLM + 0.25 Cluster + 0.20 Resolution + 0.15 Rule
    """
    fused_score = 0.40 * sig_a + 0.25 * sig_b + 0.20 * sig_c + 0.15 * sig_d
    return fused_score

def score_to_severity_label(score):
    """
    Maps continuous severity scores to categorical severity levels.
    """
    if score <= 0.5:
        return 'Low'
    elif score <= 1.5:
        return 'Medium'
    elif score <= 2.5:
        return 'High'
    else:
        return 'Critical'

def generate_pseudo_labels(df, fused_scores):
    """
    Compares assigned priority vs inferred severity to generate target binary labels,
    mismatch types (Hidden Crisis, False Alarm), and severity delta.
    """
    df_labels = df.copy()
    
    # Map inferred severity
    df_labels['inferred_severity'] = [score_to_severity_label(s) for s in fused_scores]
    
    # Map to numeric values
    assigned_numeric = df_labels['Priority_Level'].map(PRIORITY_MAP)
    inferred_numeric = df_labels['inferred_severity'].map(PRIORITY_MAP)
    
    # Mismatch is 1 if continuous difference is >= 1.3
    priority_mismatch = (np.abs(fused_scores - assigned_numeric) >= 1.3).astype(int)
    
    # Calculate severity delta and mismatch type
    mismatch_types = []
    severity_deltas = []
    for i in range(len(df)):
        if priority_mismatch[i] == 0:
            mismatch_types.append('None')
            severity_deltas.append(0)
        else:
            delta = int(inferred_numeric[i] - assigned_numeric[i])
            if delta == 0:
                # Fallback if categories are identical but continuous score was slightly over threshold
                priority_mismatch[i] = 0
                mismatch_types.append('None')
                severity_deltas.append(0)
            else:
                severity_deltas.append(delta)
                if delta > 0:
                    mismatch_types.append('Hidden Crisis')
                else:
                    mismatch_types.append('False Alarm')
                    
    df_labels['Priority_Mismatch'] = priority_mismatch
    df_labels['mismatch_type'] = mismatch_types
    df_labels['severity_delta'] = severity_deltas
    
    return df_labels

def compute_pairwise_cohens_kappa(sig_a, sig_b, sig_c, sig_d):
    """
    Computes Cohen's Kappa pairwise agreement by mapping continuous signals
    to categorical bins first.
    """
    signals = {
        'Signal A (LLM)': [score_to_severity_label(s) for s in sig_a],
        'Signal B (Cluster)': [score_to_severity_label(s) for s in sig_b],
        'Signal C (Resolution)': [score_to_severity_label(s) for s in sig_c],
        'Signal D (Rule)': [score_to_severity_label(s) for s in sig_d]
    }
    
    keys = list(signals.keys())
    matrix = np.zeros((len(keys), len(keys)))
    
    for i in range(len(keys)):
        for j in range(len(keys)):
            matrix[i, j] = cohen_kappa_score(signals[keys[i]], signals[keys[j]])
            
    df_kappa = pd.DataFrame(matrix, index=keys, columns=keys)
    return df_kappa

def run_ablation_metrics(sig_a, sig_b, sig_c, sig_d, assigned_priority, true_priority_mismatch):
    """
    Runs ablation analysis by comparing different subsets of signals
    against the final consensus target labels.
    """
    configs = {
        'Signal A Only': 1.0 * sig_a,
        'Signals A + B': (0.60 * sig_a + 0.40 * sig_b),
        'Signals A + B + C': (0.50 * sig_a + 0.30 * sig_b + 0.20 * sig_c),
        'Fused Consensus (A+B+C+D)': 0.40 * sig_a + 0.25 * sig_b + 0.20 * sig_c + 0.15 * sig_d
    }
    
    ablation_results = []
    
    for config_name, fused_score in configs.items():
        # Compare with true priority level using threshold 1.3
        assigned_num = np.array([PRIORITY_MAP[l] for l in assigned_priority])
        pred_mismatch = (np.abs(fused_score - assigned_num) >= 1.3).astype(int)
        
        # We calculate alignment with target labels
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        
        acc = accuracy_score(true_priority_mismatch, pred_mismatch)
        f1 = f1_score(true_priority_mismatch, pred_mismatch, average='macro')
        recall_pos = recall_score(true_priority_mismatch, pred_mismatch, pos_label=1, zero_division=0)
        
        ablation_results.append({
            'Configuration': config_name,
            'Agreement Accuracy': acc,
            'Macro F1': f1,
            'Recall (Mismatched)': recall_pos
        })
        
    return pd.DataFrame(ablation_results)
