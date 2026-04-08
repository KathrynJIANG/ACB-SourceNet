import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

# ============  ============
SEED = 42


KERNEL = 'rbf'  
C = 0.33  
GAMMA = 'auto'  
DEGREE = 3  

# seed
np.random.seed(SEED)

# ============ main ============
def main():
    print("="*70)
    print("Support Vector Machine (SVM) Gene Classifier")
    print("="*70)
    
    # ===== 1. load training data=====
    print("\nLoading training data...")
    
    train_gene_table_df = pd.read_csv('model_input/gene_presence_absence_train.csv', index_col=0)
    train_gene_table_df = train_gene_table_df.T
    train_gene_table = train_gene_table_df.values
    print(f"   Gene expression matrix shape: {train_gene_table.shape}")
    
    train_gene_table_binary = (train_gene_table > 0).astype(int)
    print(f"   Gene presence/absence matrix shape: {train_gene_table_binary.shape}")
    print(f"   Binary value distribution: 0={np.sum(train_gene_table_binary == 0):,}, 1={np.sum(train_gene_table_binary == 1):,}")
    print(f"   Gene presence rate: {(train_gene_table_binary == 1).sum() / train_gene_table_binary.size * 100:.2f}%")
    
    with open('model_input/final_clinical_samples.txt') as f:
        pos_ids = set(line.strip() for line in f if line.strip())
    with open('model_input/final_env_samples.txt') as f:
        neg_ids = set(line.strip() for line in f if line.strip())
    
    print(f"   Clinical samples (positive): {len(pos_ids)}")
    print(f"   Environmental samples (negative): {len(neg_ids)}")
    
    def get_label(id_):
        if id_ in pos_ids:
            return 1
        elif id_ in neg_ids:
            return 0
        else:
            raise ValueError(f"ID {id_} not found!")
    
    train_labels = np.array([get_label(idx) for idx in train_gene_table_df.index], dtype=int)
    X_train = train_gene_table_binary
    y_train = train_labels
    
    print(f"   Feature matrix shape: {X_train.shape}")
    print(f"   Label distribution: Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print(f"   Data standardized (mean=0, std=1)")
    
    # ===== 2. load test data =====
    print("\nLoading test data...")
    
    test_gene_table_df = pd.read_csv(
        'model_input/gene_presence_absence_test.csv', 
        index_col=0
    )
    test_gene_table_df = test_gene_table_df.T
    test_gene_table = test_gene_table_df.values
    print(f"   Test gene expression matrix shape: {test_gene_table.shape}")
    
    
    test_gene_table_binary = (test_gene_table > 0).astype(int)
    print(f"   Test gene presence/absence matrix shape: {test_gene_table_binary.shape}")
    
    
    test_labels_dict = {}
    with open('model_input/final_test_samples_with_source.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    sample_id, source = parts
                    # clinical -> 1, environmental -> 0
                    test_labels_dict[sample_id] = 1 if source == 'clinical' else 0
    
    print(f"   Test labels loaded: {len(test_labels_dict)} samples")
    
    
    test_labels = []
    test_samples_valid = []
    for idx in test_gene_table_df.index:
        if idx in test_labels_dict:
            test_labels.append(test_labels_dict[idx])
            test_samples_valid.append(idx)
        else:
            print(f"   Warning: Sample {idx} not found in test labels, skipping...")
    
    
    test_indices = [i for i, idx in enumerate(test_gene_table_df.index) if idx in test_labels_dict]
    X_test = test_gene_table_binary[test_indices]
    y_test = np.array(test_labels, dtype=int)
    
    print(f"   Test feature matrix shape: {X_test.shape}")
    print(f"   Test label distribution: Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()}")
    
    # ===== 3. Checking feature consistency =====
    print("\nChecking feature consistency...")
    
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature mismatch! Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
    
    
    train_genes = train_gene_table_df.columns.tolist()
    test_genes = test_gene_table_df.columns.tolist()
    
    if train_genes != test_genes:
        print("   Warning: Gene order differs between train and test!")
        print("   Reordering test data to match training data...")
        
        test_gene_table_df = test_gene_table_df[train_genes]
        X_test = (test_gene_table_df.values > 0).astype(int)[test_indices]
        print("   ✓ Test data reordered to match training data")
    else:
        print("   ✓ Gene order consistent between train and test")
    
    print(f"   Number of features: {X_train.shape[1]}")
    
    
    X_test = scaler.transform(X_test)
    print(f"   Test data standardized (mean=0, std=1)")
    
    # ===== 4. SVM =====
    print("\nCreating SVM model...")
    
    
    num_classes = 2
    counts = np.bincount(y_train, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    print(f"   Class weights: {class_weight_dict}")
    
    model = SVC(
        kernel=KERNEL,
        C=C,
        gamma=GAMMA,
        degree=DEGREE,
        # class_weight=class_weight_dict,
        class_weight='balanced',
        random_state=SEED,
        verbose=True,
        probability=True,
        max_iter=-1  
    )
    
    print(f"\n   Model Hyperparameters:")
    print(f"   - kernel: {KERNEL}")
    print(f"   - C (regularization): {C}")
    print(f"   - gamma: {GAMMA}")
    if KERNEL == 'poly':
        print(f"   - degree: {DEGREE}")
    print(f"   - class_weight: {class_weight_dict}")
    
    # ===== 5. train model =====
    print("\nTraining SVM...")
    print("="*70)
    
    model.fit(X_train, y_train)
    
    print("\nTraining completed!")
    print(f"   Number of support vectors: {model.n_support_}")
    print(f"   Support vectors per class: {dict(zip(range(num_classes), model.n_support_))}")
    
    # ===== 6. performance =====
    print("\n" + "="*70)
    print("Model Performance")
    print("="*70)
    
    
    test_probs = model.predict_proba(X_test)
    
   
    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='macro')
    test_recall = recall_score(y_test, test_preds, average='macro')
    test_f1 = f1_score(y_test, test_preds, average='macro')
    
    print("\nTest Set Performance:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")
    print("\n" + classification_report(y_test, test_preds, digits=4))
    
    # ===== 7. summary =====
    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    
    
    results_df = pd.DataFrame({
        'sample_id': test_samples_valid,
        'true_label': y_test,
        'predicted_label': test_preds,
        'prob_class_0': test_probs[:, 0],  
        'prob_class_1': test_probs[:, 1],  
        'true_class': ['Clinical' if l == 1 else 'Environmental' for l in y_test],
        'predicted_class': ['Clinical' if p == 1 else 'Environmental' for p in test_preds],
        'correct': y_test == test_preds
    })
    
    
    results_df.to_csv('svm_test_predictions.csv', index=False)
    print(f"   ✓ Predictions saved to: svm_test_predictions.csv")
    
    
    
    
    print(f"Model: Support Vector Machine (SVM)")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Number of features: {X_train.shape[1]}")
    print(f"   - Kernel: {KERNEL}")
    print(f"   - Number of support vectors: {model.n_support_.sum()}")
    print(f"\nBest Performance Metrics (Test):")
    print(f"   - Accuracy:  {test_accuracy:.4f}")
    print(f"   - Precision: {test_precision:.4f}")
    print(f"   - Recall:    {test_recall:.4f}")
    print(f"   - F1 Score:  {test_f1:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()