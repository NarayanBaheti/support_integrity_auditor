import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoModel, AutoTokenizer
import joblib

SEED = 42
torch.manual_seed(SEED)

class SIADataset(Dataset):
    """
    Custom Dataset for ticket text and metadata features.
    """
    def __init__(self, texts, cat_features, num_features, labels, tokenizer, max_len=128):
        self.texts = texts
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.num_features = torch.tensor(num_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'cat_features': self.cat_features[idx],
            'num_features': self.num_features[idx],
            'label': self.labels[idx]
        }

class SIADebertaTabularClassifier(nn.Module):
    """
    Fine-tuned DeBERTa-v3-small model integrated with metadata features.
    Outputs classification logits and the last layer's attention map for explainability.
    """
    def __init__(self, model_name='microsoft/deberta-v3-small', cat_sizes=None, num_dim=0):
        super().__init__()
        # Load backbone with output_attentions=True in float32 to prevent CPU NaN instabilities
        self.transformer = AutoModel.from_pretrained(model_name, output_attentions=True, torch_dtype=torch.float32)
        
        # Categorical metadata embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in cat_sizes
        ])
        
        total_cat_dim = sum(emb_dim for _, emb_dim in cat_sizes) if cat_sizes else 0
        meta_in_dim = total_cat_dim + num_dim
        
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # DeBERTa-v3-small hidden dimension is 768
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, input_ids, attention_mask, cat_features=None, num_features=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract CLS token
        cls_emb = outputs.last_hidden_state[:, 0, :]
        
        # Process metadata
        meta_embs = []
        if cat_features is not None and len(self.embeddings) > 0:
            for i, emb_layer in enumerate(self.embeddings):
                meta_embs.append(emb_layer(cat_features[:, i]))
        if num_features is not None and num_features.shape[1] > 0:
            meta_embs.append(num_features)
            
        if meta_embs:
            meta_features = torch.cat(meta_embs, dim=1)
            meta_enc = self.meta_encoder(meta_features)
        else:
            meta_enc = torch.zeros(cls_emb.size(0), 64, device=cls_emb.device)
            
        # Concatenate CLS token with metadata representation
        combined = torch.cat([cls_emb, meta_enc], dim=1)
        logits = self.classifier(combined)
        
        # Extract last layer's attention map for ONNX explainability
        # attentions shape: list of (batch, heads, seq, seq)
        last_attention = outputs.attentions[-1] if outputs.attentions is not None else torch.zeros(1, 1, 1, 1)
        
        return logits, last_attention

def train_classifier(model, dataloader, epochs=1, lr=2e-5, pos_weight=1.0):
    """
    Trains the DeBERTa + Metadata classifier. Addresses class imbalance via weighted loss.
    Supports incremental checkpointing and resuming to handle server preemption.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Weighted Cross Entropy Loss
    weight = torch.tensor([1.0, float(pos_weight)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    checkpoint_path = 'models/classifier/training_checkpoint.pt'
    start_epoch = 0
    
    # Check if a training checkpoint exists
    import os
    if os.path.exists(checkpoint_path):
        try:
            print(f"\n--- Found training checkpoint at {checkpoint_path}. Resuming training ---")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch + 1}/{epochs}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
        
    print(f"Training classifier on {device}...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_features = batch['cat_features'].to(device)
            num_features = batch['num_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, cat_features, num_features)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
        # Save checkpoint after each epoch
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        
    # Clean up checkpoint file when training completes successfully
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print("Cleaned up training checkpoint.")
        except Exception:
            pass
            
    return model

def run_5fold_cross_validation(X_text_emb, X_meta_cat, X_meta_num, y, cat_sizes, num_dim):
    """
    Performs 5-Fold Stratified Cross Validation on a lightweight PyTorch classifier
    (using frozen text embeddings) to report metrics quickly on CPU.
    """
    print("\nRunning 5-Fold Stratified Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    acc_scores = []
    f1_scores = []
    recall_scores = []
    
    # Custom MLP that matches the classifier head
    class MLPClassifier(nn.Module):
        def __init__(self, text_dim, cat_sizes, num_dim):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in cat_sizes
            ])
            total_cat_dim = sum(emb_dim for _, emb_dim in cat_sizes) if cat_sizes else 0
            self.meta_encoder = nn.Sequential(
                nn.Linear(total_cat_dim + num_dim, 64),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(text_dim + 64, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
            
        def forward(self, text_emb, cat_features, num_features):
            meta_embs = []
            for i, emb_layer in enumerate(self.embeddings):
                meta_embs.append(emb_layer(cat_features[:, i]))
            meta_embs.append(num_features)
            meta_enc = self.meta_encoder(torch.cat(meta_embs, dim=1))
            combined = torch.cat([text_emb, meta_enc], dim=1)
            return self.classifier(combined)
            
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_emb, y)):
        # Prepare fold data
        t_tr, t_val = torch.tensor(X_text_emb[train_idx]), torch.tensor(X_text_emb[val_idx])
        c_tr, c_val = torch.tensor(X_meta_cat[train_idx], dtype=torch.long), torch.tensor(X_meta_cat[val_idx], dtype=torch.long)
        n_tr, n_val = torch.tensor(X_meta_num[train_idx], dtype=torch.float), torch.tensor(X_meta_num[val_idx], dtype=torch.float)
        y_tr, y_val = torch.tensor(y[train_idx], dtype=torch.long), y[val_idx]
        
        # Calculate class weights for imbalance
        pos_weight = (len(y_tr) - y_tr.sum().item()) / y_tr.sum().item()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
        
        mlp = MLPClassifier(X_text_emb.shape[1], cat_sizes, num_dim)
        optimizer = optim.AdamW(mlp.parameters(), lr=1e-3)
        
        # Train fold
        mlp.train()
        for epoch in range(200): # Increased from 10 to 200 for proper convergence
            optimizer.zero_grad()
            logits = mlp(t_tr, c_tr, n_tr)
            loss = criterion(logits, y_tr)
            loss.backward()
            optimizer.step()
            
        # Evaluate
        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(t_val, c_val, n_val)
            preds = torch.argmax(val_logits, dim=1).numpy()
            
        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average='macro'))
        recall_scores.append(recall_score(y_val, preds, pos_label=1, zero_division=0))
        
        print(f"Fold {fold+1} - Accuracy: {acc_scores[-1]:.4f}, F1: {f1_scores[-1]:.4f}, Recall: {recall_scores[-1]:.4f}")
        
    print(f"\nCV Mean Accuracy: {np.mean(acc_scores):.4f} \u00b1 {np.std(acc_scores):.4f}")
    print(f"CV Mean Macro F1: {np.mean(f1_scores):.4f} \u00b1 {np.std(f1_scores):.4f}")
    print(f"CV Mean Recall (Mismatched): {np.mean(recall_scores):.4f} \u00b1 {np.std(recall_scores):.4f}")
    
    return {
        'accuracy': f"{np.mean(acc_scores):.4f} \u00b1 {np.std(acc_scores):.4f}",
        'macro_f1': f"{np.mean(f1_scores):.4f} \u00b1 {np.std(f1_scores):.4f}",
        'mismatched_recall': f"{np.mean(recall_scores):.4f} \u00b1 {np.std(recall_scores):.4f}"
    }

def export_to_onnx_and_quantize(model, output_dir, tokenizer, cat_dim, num_dim, is_surrogate=False):
    """
    Exports PyTorch model to ONNX with output_attentions support and applies dynamic quantization.
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")
    quantized_path = os.path.join(output_dir, "model_quantized.onnx")
    
    model.eval()
    
    # Define inputs depending on whether it is surrogate (text only) or classifier (text + metadata)
    if is_surrogate:
        dummy_input_ids = torch.ones(1, 64, dtype=torch.long)
        dummy_attn_mask = torch.ones(1, 64, dtype=torch.long)
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attn_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False
        )
    else:
        dummy_input_ids = torch.ones(1, 64, dtype=torch.long)
        dummy_attn_mask = torch.ones(1, 64, dtype=torch.long)
        dummy_cat_feat = torch.zeros(1, cat_dim, dtype=torch.long)
        dummy_num_feat = torch.zeros(1, num_dim, dtype=torch.float)
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attn_mask, dummy_cat_feat, dummy_num_feat),
            onnx_path,
            input_names=['input_ids', 'attention_mask', 'cat_features', 'num_features'],
            output_names=['logits', 'attentions'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'cat_features': {0: 'batch_size'},
                'num_features': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'attentions': {0: 'batch_size', 2: 'seq_len', 3: 'seq_len'}
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False
        )
        
    print(f"Exported model to ONNX at {onnx_path}")
    
    # Force PyTorch memory release and garbage collect
    import gc
    del dummy_input_ids
    del dummy_attn_mask
    if not is_surrogate:
        del dummy_cat_feat
        del dummy_num_feat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Run dynamic quantization with a self-healing fallback to float32 ONNX
    import subprocess
    import sys
    import shutil
    
    # We run quantization in a separate subprocess to avoid OOM due to PyTorch memory footprint
    try:
        print(f"Quantizing ONNX model at {onnx_path} in a separate subprocess...")
        # Format paths for Python string literal compatibility
        p_in = onnx_path.replace("\\", "/")
        p_out = quantized_path.replace("\\", "/")
        
        cmd = [
            sys.executable, "-c",
            f"from onnxruntime.quantization import quantize_dynamic, QuantType; "
            f"quantize_dynamic('{p_in}', '{p_out}', weight_type=QuantType.QUInt8)"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Saved quantized ONNX model to {quantized_path}")
            # Clean up the large non-quantized ONNX model to save space
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        else:
            print(f"Subprocess quantization failed with code {result.returncode}: {result.stderr}")
            raise RuntimeError(result.stderr)
            
    except Exception as e:
        print(f"WARNING: Dynamic quantization failed due to shape inference, OOM, or exporter limits: {e}")
        print("Falling back to standard float32 ONNX model for maximum runtime stability.")
        shutil.copy(onnx_path, quantized_path)
        # Keep the original onnx path as well for backup
        print(f"Maintained float32 model fallback at {quantized_path}")
