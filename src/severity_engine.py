import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import onnxruntime as ort
from transformers import AutoTokenizer

SEED = 42

def get_text_embeddings(texts, cache_path='data/embeddings.npy'):
    """
    Generates sentence embeddings using MiniLM-L6-v2 and caches them as float16 to save space.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached text embeddings from {cache_path}...")
        return np.load(cache_path).astype(np.float32)
    
    print("Generating sentence embeddings using all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Save as float16 to save 50% storage/memory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings.astype(np.float16))
    print(f"Saved cached embeddings to {cache_path} (shape: {embeddings.shape})")
    
    return embeddings

def compute_signal_b_embeddings_clustering(embeddings, df, n_clusters=4):
    """
    Clusters ticket embeddings and maps cluster IDs to severity values [0, 1, 2, 3]
    based on average resolution time and frequency of high severity keywords.
    """
    print("Running KMeans embedding clustering (Signal B)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters to align cluster IDs to severity levels [0, 1, 2, 3]
    cluster_metrics = []
    
    # Check for presence of outage keywords
    outage_keywords = ['outage', 'crash', 'down', 'offline', 'breach', 'security', 'failed']
    desc_series = df['Ticket_Subject'].fillna('') + ' ' + df['Ticket_Description'].fillna('')
    has_outage = desc_series.str.lower().apply(lambda text: any(kw in text for kw in outage_keywords)).values
    
    for c_id in range(n_clusters):
        idx = (cluster_labels == c_id)
        mean_res = df.loc[idx, 'Resolution_Time_Hours'].mean() if idx.sum() > 0 else 0
        outage_pct = has_outage[idx].mean() if idx.sum() > 0 else 0
        
        # Calculate cluster intensity score
        intensity_score = mean_res * 0.05 + outage_pct * 3.0
        cluster_metrics.append((c_id, intensity_score))
        
    # Sort cluster IDs by intensity score ascending (so index 0 is Low, index 3 is Critical)
    sorted_clusters = sorted(cluster_metrics, key=lambda x: x[1])
    cluster_mapping = {sorted_clusters[i][0]: float(i) for i in range(n_clusters)}
    
    mapped_signals = np.array([cluster_mapping[label] for label in cluster_labels])
    return mapped_signals

def compute_signal_c_resolution_time(df, is_train=False, regressor_path='models/resolution_regressor.joblib'):
    """
    Trains or loads an XGBoost regressor to predict expected resolution times,
    and returns a severity score [0.0 - 4.0] based on the deviation.
    """
    # Feature engineering for XGBoost
    df_feat = pd.DataFrame()
    
    # Domain tier
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com', 'example.org', 'example.net']
    email_domains = df['Customer_Email'].fillna('').apply(lambda x: x.split('@')[-1] if '@' in str(x) else '')
    df_feat['is_enterprise'] = (~email_domains.isin(free_domains)).astype(int)
    
    # Categoricals
    df_feat = pd.concat([
        df_feat,
        pd.get_dummies(df['Issue_Category'], prefix='cat', drop_first=True),
        pd.get_dummies(df['Ticket_Channel'], prefix='chan', drop_first=True)
    ], axis=1)
    
    # Ensure all features align (in case of batch inference missing some category values)
    feature_cols = joblib.load('models/resolution_feature_columns.joblib') if os.path.exists('models/resolution_feature_columns.joblib') else None
    
    if is_train:
        os.makedirs(os.path.dirname(regressor_path), exist_ok=True)
        # Train XGBoost regressor
        X = df_feat
        y = df['Resolution_Time_Hours'].fillna(df['Resolution_Time_Hours'].median())
        
        # Save columns list
        joblib.dump(X.columns.tolist(), 'models/resolution_feature_columns.joblib')
        
        regressor = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=SEED)
        regressor.fit(X, y)
        joblib.dump(regressor, regressor_path)
        print(f"Trained and saved XGBoost resolution regressor to {regressor_path}")
    else:
        if not os.path.exists(regressor_path):
            raise FileNotFoundError(f"Resolution regressor not found at {regressor_path}. Run training first.")
        regressor = joblib.load(regressor_path)
        
        # Reindex features to match training columns
        if feature_cols:
            for col in feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0
            df_feat = df_feat[feature_cols]
            
    # Predict expected resolution time
    predicted_res = regressor.predict(df_feat)
    # Avoid zero or negative values
    predicted_res = np.clip(predicted_res, 1.0, None)
    
    # Calculate severity index based on deviation
    # Severity = 2.0 + log2(actual / predicted). Clip to [0.0, 4.0]
    actual_res = df['Resolution_Time_Hours'].values
    deviation_ratio = actual_res / predicted_res
    
    # Map to 0-4 range
    severity_scores = 1.0 + np.log2(deviation_ratio + 0.1)
    return np.clip(severity_scores, 0.0, 3.0)

def compute_signal_d_rules(df):
    """
    Computes rule-based severity using an expanded escalation lexicon and negation checks.
    """
    escalation_lexicon = {
        'production down': 3.0,
        'service unavailable': 3.0,
        'outage': 3.0,
        'lost revenue': 2.5,
        'customer churn': 2.5,
        'security incident': 2.5,
        'data loss': 2.5,
        'breach': 2.5,
        'all users affected': 2.0,
        'payment failed': 2.0,
        'account locked': 2.0,
        'urgent': 1.0,
        'unable to login': 1.0,
        'crash': 1.5,
        'failed': 0.5,
        'broken': 0.5
    }
    
    negations = ['no', 'not', 'none', 'without', 'never', 'resolved', 'fixed', 'clear', 'fixed the', 'resolved the']
    
    scores = []
    desc_series = df['Ticket_Subject'].fillna('') + ' ' + df['Ticket_Description'].fillna('')
    
    for text in desc_series.values:
        text_lower = text.lower()
        ticket_score = 0.0  # Base neutral score
        
        # Match keywords and check for negation within a 3-word preceding window
        matches = []
        for phrase, weight in escalation_lexicon.items():
            pattern = rf'\b{phrase}\b'
            for match in re.finditer(pattern, text_lower):
                start_idx = match.start()
                # Extract preceding 30 characters (approx 3-4 words)
                preceding = text_lower[max(0, start_idx - 30):start_idx]
                words_preceding = re.findall(rf'\b\w+\b', preceding)
                
                # Check for negation words
                is_negated = any(neg in words_preceding for neg in negations)
                if not is_negated:
                    matches.append(weight)
                    
        if matches:
            # Take the max weight matching as primary indicator, and add small increments for extra keywords
            ticket_score = max(matches) + 0.25 * (len(matches) - 1)
            
        scores.append(min(3.0, ticket_score))
        
    return np.clip(np.array(scores), 0.0, 3.0)

def compute_signal_a_surrogate(texts, tokenizer_name='distilbert-base-uncased', onnx_path='models/severity_surrogate/model_quantized.onnx'):
    """
    Runs the DistilBERT surrogate model using ONNX Runtime.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Quantized surrogate ONNX model not found at {onnx_path}. Run training first.")
        
    print(f"Loading surrogate ONNX model from {onnx_path}...")
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    scores = []
    # Process in batches to optimize CPU memory
    batch_size = 1
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=64, 
            return_tensors="np"
        )
        
        # Prepare inputs for ONNX session
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        ort_outs = session.run(None, ort_inputs)
        # Assuming the model returns continuous regression logits
        batch_scores = ort_outs[0].squeeze(-1)
        if batch_scores.ndim == 0:
            scores.append(float(batch_scores))
        else:
            scores.extend(batch_scores.tolist())
        
    # Scale output from surrogate to [0.0 - 3.0]
    return np.clip(np.array(scores), 0.0, 3.0)
