import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============  ============
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 5e-4
SEED = 42


HIDDEN_DIMS = [512, 512, 256, 256]  

DROPOUT = 0.3
USE_BATCH_NORM = True
ACTIVATION = 'gelu'

# lambda
L1_LAMBDA = 1e-4
WEIGHT_DECAY = 1e-3

# seed
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============ Resent ============
class ResidualBlock(nn.Module):
    
    
    def __init__(self, dim, dropout=0.3, use_batch_norm=True, activation='gelu'):
        super().__init__()
        
        
        self.batch_norm1 = nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
        self.linear1 = nn.Linear(dim, dim)
        self.batch_norm2 = nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
    
    def forward(self, x):
        """Pre-activation Residual Block"""
        identity = x
        
      
        out = self.batch_norm1(x)
        out = self.activation(out)
        out = self.linear1(out)
        
        out = self.batch_norm2(out)
        out = self.activation(out)
        out = self.dropout(out) 
        out = self.linear2(out)
        
        
        out = out + identity
        
        return out

# ============ Resent MLP Classifier ============
class ResidualMLPClassifier(nn.Module):
    """
    Structure：
    Input -> Linear -> [ResBlock -> ResBlock -> ...] -> Downsample -> ... -> Classifier
    
    - ResBlock
    - Downsample
    """
    
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.3, 
                 use_batch_norm=True, activation='gelu', blocks_per_dim=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.blocks_per_dim = blocks_per_dim  
        
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # （input_dim -> first_hidden_dim）
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        )
        
        
        self.res_layers = nn.ModuleList()
        
        for i, dim in enumerate(hidden_dims):
            
            res_blocks = nn.ModuleList([
                ResidualBlock(dim, dropout, use_batch_norm, activation)
                for _ in range(blocks_per_dim)
            ])
            self.res_layers.append(res_blocks)
            
            
            if i < len(hidden_dims) - 1:
                next_dim = hidden_dims[i + 1]
                downsample = nn.Sequential(
                    nn.Linear(dim, next_dim),
                    nn.BatchNorm1d(next_dim) if use_batch_norm else nn.Identity(),
                    self.activation,
                    nn.Dropout(dropout)
                )
                self.res_layers.append(downsample)
        
        
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)
        
        
        self._init_weights()
    
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, n_classes)
        """
        
        out = self.input_proj(x)
        
        
        for layer in self.res_layers:
            if isinstance(layer, nn.ModuleList):
                
                for res_block in layer:
                    out = res_block(out)
            else:
                
                out = layer(out)
        
        
        logits = self.classifier(out)
        
        return logits
    
    def get_feature_importance(self, dataloader, device, target_class=1):
        """
        
        """
        self.eval()
        all_gradients = []
        
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            inputs.requires_grad = True
            
            
            logits = self.forward(inputs)
            
            
            target_logits = logits[:, target_class]
            target_logits.sum().backward()
            
            
            gradients = inputs.grad.detach().abs()
            all_gradients.append(gradients.cpu().numpy())
            
            
            inputs.grad = None
        
        
        avg_gradients = np.concatenate(all_gradients, axis=0).mean(axis=0)
        
        return avg_gradients
    
    
# ============ Focal Loss ============
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - 
        Returns:
            loss: scalar
        """
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        
        p = torch.exp(-ce_loss)
        
        
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        
        if self.alpha is not None:
            alpha_t = torch.ones_like(targets, dtype=torch.float)
            alpha_t[targets == 1] = self.alpha
            alpha_t[targets == 0] = 1 - self.alpha
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


# ============ visualize ============
def visualize_feature_importance(importance_scores, gene_names=None, top_k=30, 
                                 save_path='feature_importance.png'):
    
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_scores = importance_scores[top_indices]
    
    if gene_names is not None:
        labels = [gene_names[i] for i in top_indices]
    else:
        labels = [f'Feature_{i}' for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    bars = ax.barh(range(top_k), top_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Importance Score (Gradient Magnitude)', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important Features (Residual MLP)', fontsize=15, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', 
                ha='left', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.show()


def plot_training_history(train_losses, test_losses, test_f1_scores, save_path='training_history.png'):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # F1 Score
    ax2.plot(epochs, test_f1_scores, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    best_epoch = np.argmax(test_f1_scores)
    best_f1 = test_f1_scores[best_epoch]
    ax2.plot(best_epoch + 1, best_f1, 'r*', markersize=15, 
             label=f'Best: {best_f1:.4f} (Epoch {best_epoch + 1})')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.show()


# ============ train ============
def train_epoch(model, dataloader, optimizer, loss_fn, device, l1_lambda=0.0):
    """one epoch"""
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
    print("Residual MLP Gene Classifier with Skip Connections")
    print("="*70)
    
    # ===== 1. load training data =====
    print("\n[1] Loading TRAINING data...")
    
    
    train_gene_table_df = pd.read_csv(
        'model_input/gene_presence_absence_train.csv', 
        index_col=0
    )
    train_gene_table_df = train_gene_table_df.T
    train_gene_table = train_gene_table_df.values
    print(f"   Gene expression matrix shape: {train_gene_table.shape}")
    
    
    train_gene_table_binary = (train_gene_table > 0).astype(int)
    print(f"   Gene presence/absence matrix shape: {train_gene_table_binary.shape}")
    print(f"   Binary value distribution: 0={np.sum(train_gene_table_binary == 0):,}, 1={np.sum(train_gene_table_binary == 1):,}")
    print(f"   Gene presence rate: {(train_gene_table_binary == 1).sum() / train_gene_table_binary.size * 100:.2f}%")
    
    
    with open('model_input/final_clinical_samples.txt') as f:
        train_pos_ids = set(line.strip() for line in f if line.strip())
    with open('model_input/final_env_samples.txt') as f:
        train_neg_ids = set(line.strip() for line in f if line.strip())
    
    print(f"   Clinical samples (positive): {len(train_pos_ids)}")
    print(f"   Environmental samples (negative): {len(train_neg_ids)}")
    
    def get_train_label(id_):
        if id_ in train_pos_ids:
            return 1
        elif id_ in train_neg_ids:
            return 0
        else:
            raise ValueError(f"ID {id_} not found in training labels!")
    
    train_labels = np.array([get_train_label(idx) for idx in train_gene_table_df.index], dtype=int)
    X_train = train_gene_table_binary
    y_train = train_labels
    
    print(f"   Training feature matrix shape: {X_train.shape}")
    print(f"   Training label distribution: Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
    
    # ===== 2. loading test data =====
    print("\n[2] Loading TEST data...")
    
   
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
    
    # ===== 3. Checking feature consistency  =====
    print("\n[3] Checking feature consistency...")
    
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
    
    # ===== 4. Creating data loaders =====
    print("\n[4] Creating data loaders...")
    
    train_dataset = GeneDataset(X_train, y_train)
    test_dataset = GeneDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # ===== 5. create model =====
    print("\n[5] Creating model...")
    
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = ResidualMLPClassifier(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        n_classes=2,
        dropout=DROPOUT,
        use_batch_norm=USE_BATCH_NORM,
        activation=ACTIVATION,
        blocks_per_dim=2  
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Hidden dims: {HIDDEN_DIMS}")
    print(f"   Blocks per dim: 2 (残差块)")
    print(f"   Activation: {ACTIVATION}")
    
    print(f"\n   Model Architecture:")
    print(f"   Input({input_dim}) -> Proj({HIDDEN_DIMS[0]}) ->")
    for i, dim in enumerate(HIDDEN_DIMS):
        print(f"   [ResBlock({dim}) -> ResBlock({dim})]", end="")
        if i < len(HIDDEN_DIMS) - 1:
            print(f" -> Downsample({HIDDEN_DIMS[i+1]}) ->")
        else:
            print(f" -> Classifier(2)")
    
    # ===== 6. prepare training =====
    print("\n[6] Preparing training...")
    
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
    
    
    # ===== 7. epochs =====
    print(f"\n[7] Starting training ({EPOCHS} epochs)...")
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
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, l1_lambda=L1_LAMBDA)
        train_losses.append(train_loss)
        
        
        test_loss, test_preds, test_labels_actual, test_probs = validate(model, test_loader, loss_fn, device)
        test_losses.append(test_loss)
        
        test_f1 = f1_score(test_labels_actual, test_preds, average='macro')
        test_auc = roc_auc_score(test_labels_actual, test_probs[:, 1])
        test_f1_scores.append(test_f1)
        
        
        scheduler.step(test_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | LR: {current_lr:.2e}")
        
        if (epoch + 1) == EPOCHS or epoch % 10 == 9:
            print("\n" + classification_report(test_labels_actual, test_preds, digits=4))
        
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_test_preds = test_preds
            best_test_probs = test_probs
            best_test_labels = test_labels_actual
            
            torch.save(model.state_dict(), './best_residual_mlp_model.pth')
            
            print(f"   ✓ Best model saved (Test F1: {best_test_f1:.4f})")
            print('auc:', test_auc)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping: no improvement for {max_patience} epochs")
                break
    
    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"   Best test F1: {best_test_f1:.4f} (Epoch {best_epoch})")
    print("="*70)
    
    
    print("\n[8] Saving best model predictions...")
    
    best_results_df = pd.DataFrame({
        'sample_id': test_samples_valid,  
        'true_label': best_test_labels,
        'predicted_label': best_test_preds,
        'prob_class_0': best_test_probs[:, 0],  
        'prob_class_1': best_test_probs[:, 1],  
        'true_class': ['Clinical' if l == 1 else 'Environmental' for l in best_test_labels],
        'predicted_class': ['Clinical' if p == 1 else 'Environmental' for p in best_test_preds],
        'correct': best_test_labels == best_test_preds
    })
    
    best_results_df.to_csv('./resnet_test_predictions.csv', index=False)
    print(f"   ✓ Best model predictions saved to: resnet_test_predictions.csv")
    print(f"   ✓ Best epoch: {best_epoch}, Best F1: {best_test_f1:.4f}")



if __name__ == "__main__":
    main()