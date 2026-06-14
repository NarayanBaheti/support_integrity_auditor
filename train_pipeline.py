import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
from transformers import AutoTokenizer, AutoModel

# Import custom src modules
from src.severity_engine import (
    get_text_embeddings,
    compute_signal_b_embeddings_clustering,
    compute_signal_c_resolution_time,
    compute_signal_d_rules,
    compute_signal_a_surrogate
)
from src.pseudo_labeling import (
    fuse_severity_signals,
    generate_pseudo_labels,
    compute_pairwise_cohens_kappa,
    run_ablation_metrics
)
from src.classifier import (
    SIADataset,
    SIADebertaTabularClassifier,
    train_classifier,
    run_5fold_cross_validation,
    export_to_onnx_and_quantize
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class SIADistilBertSurrogate(nn.Module):
    """
    Custom DistilBERT surrogate model for regression.
    Outputs a single continuous score [1.0 - 4.0].
    """
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.pre_classifier(cls_emb)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def main():
    import torch
    torch.set_num_threads(4)
    print("==================================================")
    print("Support Integrity Auditor (SIA) - Pipeline Started (Optimized CPU)")
    print("==================================================")
    
    # 1. Load data & filter for CPU efficiency
    df_raw = pd.read_csv('data/customer_support_tickets.csv')
    seed_labels = pd.read_csv('data/severity_labels_1000.csv')
    
    # Extract the 1000 seed rows and 4000 other rows to form a 5000 subset
    df_seed = df_raw[df_raw['Ticket_ID'].isin(seed_labels['Ticket_ID'])].copy()
    df_other = df_raw[~df_raw['Ticket_ID'].isin(seed_labels['Ticket_ID'])].copy()
    
    df = pd.concat([
        df_seed,
        df_other.sample(n=4000, random_state=SEED)
    ]).reset_index(drop=True)
    
    print(f"Loaded main CRM dataset. Optimized to {len(df)} rows for CPU execution.")
    
    # Preprocess text and email domain
    df['combined_text'] = df['Ticket_Subject'].fillna('').astype(str) + " " + df['Ticket_Description'].fillna('').astype(str)
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com', 'example.org', 'example.net']
    df['email_domain'] = df['Customer_Email'].fillna('').apply(lambda x: x.split('@')[-1] if '@' in str(x) else 'unknown')
    df['customer_tier'] = df['email_domain'].apply(lambda d: 'Standard' if d in free_domains or d == 'unknown' else 'Enterprise')
    
    # 2. Train/Load DistilBERT Surrogate for Signal A
    surr_onnx_path = "models/severity_surrogate/model_quantized.onnx"
    texts_all = df['combined_text'].tolist()
    
    if os.path.exists(surr_onnx_path):
        print("\n--- Quantized DistilBERT Severity Surrogate already exists. Skipping training. ---")
        print("Computing Signal A (Surrogate) via ONNX Runtime on CPU...")
        sig_a = compute_signal_a_surrogate(texts_all, onnx_path=surr_onnx_path)
    else:
        print("\n--- Training DistilBERT Severity Surrogate ---")
        # Merge back to get text
        surr_df = pd.merge(seed_labels, df[['Ticket_ID', 'combined_text']], on='Ticket_ID', how='inner')
        
        # Map text severity labels to numerical scores [0.0, 3.0]
        severity_label_map = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0, 'Critical': 3.0}
        surr_df['severity_num'] = surr_df['severity_llm'].map(severity_label_map)
        
        # Tokenizer & Model
        surr_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        surr_model = SIADistilBertSurrogate('distilbert-base-uncased')
        
        # Custom simple regression dataset (max_len=64 for CPU efficiency)
        class SurrogateDataset(Dataset):
            def __init__(self, texts, scores, tokenizer, max_len=64):
                self.texts = texts
                self.scores = torch.tensor(scores, dtype=torch.float)
                self.tokenizer = tokenizer
                self.max_len = max_len
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                inputs = self.tokenizer(str(self.texts[idx]), padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'label': self.scores[idx]
                }
                
        surr_ds = SurrogateDataset(surr_df['combined_text'].tolist(), surr_df['severity_num'].tolist(), surr_tokenizer)
        surr_loader = DataLoader(surr_ds, batch_size=8, shuffle=True)
        
        # Quick 1 epoch fine-tuning on CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        surr_model.to(device)
        surr_optimizer = optim.AdamW(surr_model.parameters(), lr=3e-5)
        criterion = nn.MSELoss()
        
        surr_model.train()
        print("Fine-tuning DistilBERT on 1,000 seed labels...")
        for epoch in range(1):
            total_loss = 0.0
            for batch in surr_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                surr_optimizer.zero_grad()
                outputs = surr_model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(-1), labels)
                loss.backward()
                surr_optimizer.step()
                total_loss += loss.item()
            print(f"Surrogate Train Loss: {total_loss/len(surr_loader):.4f}")
            
        # Export surrogate to ONNX
        print("Exporting surrogate to quantized ONNX...")
        export_to_onnx_and_quantize(
            surr_model, 
            "models/severity_surrogate", 
            surr_tokenizer, 
            cat_dim=0, 
            num_dim=0, 
            is_surrogate=True
        )
        
        # Inference to compute sig_a
        surr_model.eval()
        sig_a_list = []
        print("Computing Signal A (Surrogate) via PyTorch on CPU...")
        with torch.no_grad():
            for i in range(0, len(texts_all), 64):
                batch_texts = texts_all[i:i+64]
                inputs = surr_tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                )
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                outputs = surr_model(input_ids, attention_mask)
                batch_scores = outputs.squeeze(-1).cpu().numpy()
                if batch_scores.ndim == 0:
                    sig_a_list.append(float(batch_scores))
                else:
                    sig_a_list.extend(batch_scores.tolist())
        sig_a = np.clip(np.array(sig_a_list), 0.0, 3.0)
        
    # 3. Compute 4 Signals for all tickets in our subset
    print("\n--- Generating 4 Severity Signals ---")
    
    # Signal B (MiniLM Embeddings clustering)
    embeddings = get_text_embeddings(texts_all)
    sig_b = compute_signal_b_embeddings_clustering(embeddings, df)
    
    # Signal C (Resolution Time Regressor)
    sig_c = compute_signal_c_resolution_time(df, is_train=True)
    
    # Signal D (Rule-Based NLP rules)
    sig_d = compute_signal_d_rules(df)
    
    # 4. Signal Fusion & Pseudo-Label Generation
    print("\n--- Running Signal Fusion & Label Generation ---")
    fused_scores = fuse_severity_signals(sig_a, sig_b, sig_c, sig_d)
    df_labeled = generate_pseudo_labels(df, fused_scores)
    
    # Save labeled dataset
    df_labeled.to_csv('data/enhanced_customer_support_data.csv', index=False)
    print("Enhanced pseudo-labeled dataset saved to data/enhanced_customer_support_data.csv")
    
    # Compute Cohen's Kappa agreement
    df_kappa = compute_pairwise_cohens_kappa(sig_a, sig_b, sig_c, sig_d)
    print("\nPairwise Cohen's Kappa Signal Agreement Matrix:")
    print(df_kappa)
    
    # Compute Ablation Study Metrics
    df_ablation = run_ablation_metrics(
        sig_a, sig_b, sig_c, sig_d, 
        df_labeled['Priority_Level'].values, 
        df_labeled['Priority_Mismatch'].values
    )
    print("\nAblation Study Results:")
    print(df_ablation)
    
    # 5. 5-Fold Cross Validation on Fast Tabular Model
    print("\n--- Running 5-Fold Tabular Cross-Validation ---")
    cat_columns = ['Issue_Category', 'Ticket_Channel', 'customer_tier', 'Priority_Level']
    encoders = {}
    X_meta_cat = []
    cat_sizes = []
    
    for col in cat_columns:
        categories = sorted(df_labeled[col].fillna('unknown').unique().tolist())
        encoders[col] = {cat: i for i, cat in enumerate(categories)}
        emb_dim = max(4, min(50, len(categories) // 2 + 2))
        cat_sizes.append((len(categories), emb_dim))
        X_meta_cat.append(df_labeled[col].fillna('unknown').map(encoders[col]).values)
        
    X_meta_cat = np.stack(X_meta_cat, axis=1)
    
    log_res_time = np.log1p(df_labeled['Resolution_Time_Hours'].fillna(0).values).reshape(-1, 1)
    res_mean = log_res_time.mean()
    res_std = log_res_time.std()
    X_meta_num = (log_res_time - res_mean) / (res_std + 1e-5)
    
    joblib.dump(encoders, 'models/metadata_encoders.joblib')
    joblib.dump({'mean': res_mean, 'std': res_std}, 'models/resolution_scaler.joblib')
    print("Saved metadata encoders and resolution scalers.")
    
    y = df_labeled['Priority_Mismatch'].values
    cv_metrics = run_5fold_cross_validation(embeddings, X_meta_cat, X_meta_num, y, cat_sizes, num_dim=1)
    
    # 6. Fine-Tune DeBERTa-v3 Classifier
    print("\n--- Fine-Tuning DeBERTa-v3-Small + Metadata Classifier ---")
    deberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    
    # Sample 4000 rows for classifier training, keep 1000 for validation threshold calibration
    train_idx, val_idx = train_test_split(
        np.arange(len(df_labeled)), 
        train_size=4000, 
        stratify=y, 
        random_state=SEED
    )
    
    t_train = df_labeled['combined_text'].values[train_idx]
    c_train = X_meta_cat[train_idx]
    n_train = X_meta_num[train_idx]
    y_train = y[train_idx]
    
    # Max length = 64
    train_ds = SIADataset(t_train, c_train, n_train, y_train, deberta_tokenizer, max_len=64)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = SIADebertaTabularClassifier(
        model_name='microsoft/deberta-v3-small', 
        cat_sizes=cat_sizes, 
        num_dim=1
    )
    
    # Freeze all layers of the transformer backbone except the last 2 layers to speed up CPU training with higher capacity
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.transformer.encoder.layer[-2:].parameters():
        param.requires_grad = True
        
    # Check if pre-trained weights exist to skip retraining on CPU
    weights_path = 'models/classifier/model_weights.pt'
    if os.path.exists(weights_path):
        print(f"\n--- Found pre-trained weights at {weights_path}. Loading model ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        # Train 5 epochs on CPU
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = train_classifier(model, train_loader, epochs=5, lr=1e-4, pos_weight=pos_weight)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"Saved trained model weights to {weights_path}")
    
    # Calibrate decision threshold on validation set
    print("\n--- Decision Threshold Calibration ---")
    t_val = df_labeled['combined_text'].values[val_idx]
    c_val = X_meta_cat[val_idx]
    n_val = X_meta_num[val_idx]
    y_val = y[val_idx]
    
    val_ds = SIADataset(t_val, c_val, n_val, y_val, deberta_tokenizer, max_len=64)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_features = batch['cat_features'].to(device)
            num_features = batch['num_features'].to(device)
            
            logits, _ = model(input_ids, attention_mask, cat_features, num_features)
            probs = torch.softmax(logits, dim=-1)
            val_probs.extend(probs[:, 1].cpu().numpy())
            
    val_probs = np.array(val_probs)
    
    from sklearn.metrics import f1_score
    best_th = 0.50
    best_f1 = 0.0
    for th in np.arange(0.35, 0.76, 0.01):
        preds = (val_probs >= th).astype(int)
        f1 = f1_score(y_val, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    print(f"Optimal Decision Threshold: {best_th:.4f} (Validation Macro F1: {best_f1:.4f})")
    joblib.dump(best_th, 'models/decision_threshold.joblib')
    print("Saved calibrated decision threshold to models/decision_threshold.joblib")
    
    # Export to quantized ONNX
    print("\nExporting final DeBERTa classifier to quantized ONNX with output_attentions=True...")
    export_to_onnx_and_quantize(
        model, 
        "models/classifier", 
        deberta_tokenizer, 
        cat_dim=X_meta_cat.shape[1], 
        num_dim=X_meta_num.shape[1], 
        is_surrogate=False
    )
    
    # 7. Save Metrics
    metrics_path = 'models/best_model/metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics_data = {
        'ablation_study': df_ablation.to_dict(orient='records'),
        'cohens_kappa': df_kappa.to_dict(orient='dict'),
        '5fold_cv': cv_metrics,
        'final_verification': {
            'accuracy': cv_metrics['accuracy'],
            'macro_f1': cv_metrics['macro_f1'],
            'mismatched_recall': cv_metrics['mismatched_recall']
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\nFinal metrics report saved to {metrics_path}")
    print("==================================================")
    print("Pipeline Execution Completed Successfully!")
    print("==================================================")

if __name__ == '__main__':
    main()
