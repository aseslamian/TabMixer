# This is a simple code for using TabMixer for tabular data supervisd learning for Classification Task.

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tabmixer import TabMixer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler

########################### YOU SHOULD DO ###########################
# 1 # Load your data
file_path = '/path/to/your/data/your_Excel_File.csv'
data = pd.read_csv(file_path)

# 2 # Define your features and label
features = ['feature1', 'feature2','feature3','feature4','feature5','feature6']
target = ['ClassLabels']

# 3 # Define number of classes
NUM_CLASS = 3
######################################################################

## Feature Engineering & Data Preprocessing 
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[features] = data[features].fillna(data[features].mean())
data = data.dropna(subset=[target])
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])
data[target] = data[target].astype(int)
X = data[features].astype(float).values
y = data[target].values

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to tensors and move to GPU if available
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_val)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

y_train = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=NUM_CLASS).to(device)
y_val = torch.nn.functional.one_hot(torch.tensor(y_val), num_classes=NUM_CLASS).to(device)
y_test = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=NUM_CLASS).to(device)

NUM_FEATURES = X_train.shape[1] 
NUM_TOKENS = 64 
DIM_FORWARD = 256 # define the number of neuron for each MLP latyer in tabMixer block

class TabMixerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dim = NUM_TOKENS
        self.feature_embedding = nn.Linear(1, self.dim)

        self.tabmixer = TabMixer(
            dim_tokens=NUM_FEATURES,       
            dim_features=self.dim,    
            dim_feedforward= DIM_FORWARD 
        )

        self.head = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.GELU(),
            nn.Linear(32, NUM_CLASS)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.tabmixer(x)
        x = x.mean(dim=1) 
        return self.head(x)


# Initialize model
model = TabMixerClassifier().to(device)
# model = TabMixerClassifier().to(device).double()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = loss_fn(outputs, torch.argmax(y_train, dim=1))  # convert one-hot to label

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_labels = torch.argmax(y_val, dim=1)
        val_acc = accuracy_score(val_labels.cpu(), val_preds.cpu())

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")


model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
    y_true = torch.argmax(y_test, dim=1).cpu().numpy()

# Multiclass ROC AUC
y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASS))
auc = roc_auc_score(y_true_bin, test_probs, average='macro')
print("Test ROC AUC:", auc)
