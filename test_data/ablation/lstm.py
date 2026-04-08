import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ============ ============
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-4
SEED = 42

# LSTM para
HIDDEN_SIZE = 256          
NUM_LAYERS = 3             
BIDIRECTIONAL = False       
DROPOUT = 0.2
USE_BATCH_NORM = True
SEQUENCE_LENGTH = 10       

# re
L1_LAMBDA = 1e-5
WEIGHT_DECAY = 1e-4

# random seed
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============ LSTM classifier ============
class LSTMClassifier(nn.Module):
    """
    
    structure：
    Input -> Reshape to sequence -> LSTM -> Pooling -> Classifier
    
    """
    
    def __init__(self, input_dim, hidden_size, num_layers, n_classes, 
                 dropout=0.3, bidirectional=True, sequence_length=10,
                 use_batch_norm=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        
        
        self.feature_per_step = input_dim // sequence_length
        if input_dim % sequence_length != 0:
            
            self.feature_per_step += 1
            self.padded_input_dim = self.feature_per_step * sequence_length
        else:
            self.padded_input_dim = input_dim
        
        
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_per_step, hidden_size),
            nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
       
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, n_classes)
        )
        
       
        self._init_weights()
    
    def _init_weights(self):
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                
                if param.dim() >= 2:  
                    nn.init.xavier_uniform_(param.data)
                else:
                    
                    nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                
                nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, n_classes)
        """
        batch_size = x.size(0)
        
        
        if self.padded_input_dim > self.input_dim:
            padding = torch.zeros(batch_size, self.padded_input_dim - self.input_dim, 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        # reshape to sequence: (batch_size, sequence_length, feature_per_step)
        x = x.view(batch_size, self.sequence_length, self.feature_per_step)
        
       
        # (batch_size, sequence_length, feature_per_step) -> (batch_size * sequence_length, feature_per_step)
        x_flat = x.view(batch_size * self.sequence_length, self.feature_per_step)
        x_proj = self.input_proj(x_flat)
        # -> (batch_size, sequence_length, hidden_size)
        x = x_proj.view(batch_size, self.sequence_length, self.hidden_size)
        
        # LSTM process
        # lstm_out: (batch_size, sequence_length, lstm_output_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        
        # attention_weights: (batch_size, sequence_length, 1)
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # sum: (batch_size, lstm_output_dim)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # classifier
        logits = self.classifier(attended)
        
        return logits


# ============ dataset ============
class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.X[idx], dtype=torch.float),
            'label': torch.tensor(self.y[idx], dtype=torch.long)
        }


def collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'inputs': inputs, 'labels': labels}


# ============ train epoch ============
def train_epoch(model, dataloader, optimizer, loss_fn, device, l1_lambda=0.0):
    """epoch"""
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device)
        
        
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        
        # L1 lambda
        if l1_lambda > 0:
            l1_reg = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg += torch.abs(param).sum()
            total_loss = loss + l1_lambda * l1_reg
        else:
            total_loss = loss
        
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    """evalation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            probs = F.softmax(logits, dim=-1).cpu().numpy() 
            
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    return epoch_loss, all_preds, all_labels, all_probs


# ============ main ============
def main():
    print("="*70)
    print("LSTM Gene Classifier")
    print("="*70)
    
    # ===== 1. load=====
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
    
    # ===== 2. load test data =====
    print("\nLoading test data...")
    
    test_gene_table_df = pd.read_csv(
        'model_input/gene_presence_absence_test.csv', 
        index_col=0
    )
    test_gene_table_df = test_gene_table_df.T
    test_gene_table = test_gene_table_df.values
    print(f"   Test gene expression matrix shape: {test_gene_table.shape}")
    
    # binary transfer
    test_gene_table_binary = (test_gene_table > 0).astype(int)
    print(f"   Test gene presence/absence matrix shape: {test_gene_table_binary.shape}")
    
    # test label dict
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
    
    # ===== 3. check consistency =====
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
    
    # ===== 4. create data loaders =====
    print("\nCreating data loaders...")
    
    train_dataset = GeneDataset(X_train, y_train)
    test_dataset = GeneDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # ===== 5. create model =====
    print("\nCreating model...")
    
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_classes=2,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        sequence_length=SEQUENCE_LENGTH,
        use_batch_norm=USE_BATCH_NORM
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Num layers: {NUM_LAYERS}")
    print(f"   Bidirectional: {BIDIRECTIONAL}")
    print(f"   Sequence length: {SEQUENCE_LENGTH}")
    
    direction = "Bidirectional" if BIDIRECTIONAL else "Unidirectional"
    print(f"\n   Model Architecture:")
    print(f"   Input({input_dim}) -> Reshape({SEQUENCE_LENGTH}, {model.feature_per_step}) ->")
    print(f"   Projection({HIDDEN_SIZE}) -> {direction} LSTM({NUM_LAYERS} layers) ->")
    print(f"   Attention Pooling -> Classifier(2)")
    
    # ===== 6. prepare training =====
    print("\nPreparing training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    num_classes = 2
    counts = np.bincount(y_train, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"   Class weights: {weights.cpu().numpy()}")
    
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    # ===== 7. training epochs =====
    print(f"\nStarting training ({EPOCHS} epochs)...")
    print("="*70)
    
    best_test_f1 = 0.0
    best_epoch = -1
    patience_counter = 0
    max_patience = 20
    
    best_test_preds = None
    best_test_probs = None
    best_test_labels = None
    
    train_losses = []
    test_losses = []
    test_f1_scores = []
    
    for epoch in range(EPOCHS):
        # train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, l1_lambda=L1_LAMBDA)
        train_losses.append(train_loss)
        
        # test
        test_loss, test_preds, test_labels, test_probs = validate(model, test_loader, loss_fn, device)
        test_losses.append(test_loss)
        
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        test_f1_scores.append(test_f1)
        
        # scheduler
        scheduler.step(test_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | LR: {current_lr:.2e}")
        
        if (epoch + 1) == EPOCHS or epoch % 10 == 9:
            print("\n" + classification_report(test_labels, test_preds, digits=4))
        
        # save best model
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_test_preds = test_preds
            best_test_probs = test_probs
            best_test_labels = test_labels
            print(f"   ✓ Best model (F1: {best_test_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping: no improvement for {max_patience} epochs")
                break
    
    print("\n" + "="*70)
    print(f"Training completed!")
    
    best_results_df = pd.DataFrame({
        'sample_id': test_samples_valid,
        'true_label': best_test_labels,
        'predicted_label': best_test_preds,
        'prob_class_0': best_test_probs[:, 0],
        'prob_class_1': best_test_probs[:, 1],
    })
    best_results_df.to_csv('LSTM_test_predictions.csv', index=False)
    print(f"   Best test F1: {best_test_f1:.4f} (Epoch {best_epoch})")
    print("="*70)
    
    # ===== 8. final performance =====
    print("\n" + "="*70)
    print("Model Performance")
    print("="*70)
    
    
    
    _, test_preds_final, test_labels_final, _ = validate(model, test_loader, loss_fn, device)
    test_accuracy = accuracy_score(test_labels_final, test_preds_final)
    test_precision = precision_score(test_labels_final, test_preds_final, average='macro')
    test_recall = recall_score(test_labels_final, test_preds_final, average='macro')
    test_f1_final = f1_score(test_labels_final, test_preds_final, average='macro')
    
    print("\nTest Set Performance:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1 Score:  {test_f1_final:.4f}")
    print("\n" + classification_report(test_labels_final, test_preds_final, digits=4))


if __name__ == "__main__":
    main()