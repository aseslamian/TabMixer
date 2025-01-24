# A novel data augmentation approach to fault diagnosis with class-imbalance problem

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def load_and_preprocess_data(file_path, target_column):
    """Load and preprocess the dataset"""
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    print(df[target_column].value_counts())
    print("\nImbalance ratio:", 
          df[target_column].value_counts().max() / df[target_column].value_counts().min())
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='if_binary', sparse=False), categorical_columns)
        ])
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    if len(numerical_columns) > 0:
        feature_names.extend(numerical_columns.tolist())
    if len(categorical_columns) > 0:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
        feature_names.extend(cat_features.tolist())
    
    # Convert target to numeric
    unique_classes = y.unique()
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = y.map(class_mapping)
    
    return X_transformed, y_encoded, preprocessor, feature_names, class_mapping

def vae_loss(recon_x, x, mu, logvar, alpha=None):
    """Custom loss function with adaptive weight"""
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 0, 1)
    
    # Reconstruction loss (BCE)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Adaptive weight
    if alpha is None:
        alpha = KLD / (BCE + KLD)
    
    return BCE + alpha * KLD

def train_vae_for_minority(X_train, y_train, minority_class, batch_size=32, num_epochs=100, learning_rate=1e-3):
    """Train VAE specifically on minority class data"""
    # Get minority class samples
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask]
    
    print(f"\nTraining VAE on minority class {minority_class}")
    print(f"Number of minority samples: {X_minority.shape[0]}")
    
    # Initialize VAE
    input_dim = X_minority.shape[1]
    hidden_dim = max(128, input_dim * 2)
    latent_dim = max(10, input_dim // 4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Prepare data loader
    X_tensor = torch.FloatTensor(X_minority).to(device)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Calculate loss with adaptive weight
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Average Loss: {total_loss / len(dataloader.dataset):.4f}')
    
    return vae

def generate_balanced_dataset(X_train, y_train, vae, minority_class, majority_count):
    """Generate synthetic samples to balance the dataset"""
    # Get current minority class count
    minority_count = np.sum(y_train == minority_class)
    
    # Calculate number of synthetic samples needed
    n_synthetic = majority_count - minority_count
    
    print(f"\nGenerating {n_synthetic} synthetic samples for class {minority_class}")
    
    # Generate synthetic samples
    vae.eval()
    device = next(vae.parameters()).device
    
    with torch.no_grad():
        z = torch.randn(n_synthetic, vae.fc_mu.out_features).to(device)
        synthetic_samples = vae.decode(z).cpu().numpy()
        synthetic_samples = np.clip(synthetic_samples, 0, 1)
    
    # Create synthetic labels
    synthetic_labels = np.full(n_synthetic, minority_class)
    
    # Combine with original data
    X_balanced = np.vstack([X_train, synthetic_samples])
    y_balanced = np.concatenate([y_train, synthetic_labels])
    
    return X_balanced, y_balanced

def main():
    # Set parameters
    # file_path = 'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\Credit-g(CG)\credit-g.csv'
    file_path = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\Credit-g(CG)\credit-g.csv'

    target_column = 'class'        # Replace with your target column name
    test_size = 0.2
    random_state = 42
    
    try:
        # Load and preprocess data
        X, y, preprocessor, feature_names, class_mapping = load_and_preprocess_data(
            file_path, target_column
        )
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Get class distribution
        class_counts = pd.Series(y_train).value_counts()
        majority_count = class_counts.max()
        minority_classes = class_counts[class_counts < majority_count].index
        
        # Initialize storage for balanced data
        X_balanced = X_train.copy()
        y_balanced = y_train.copy()
        
        # Train VAE and generate synthetic samples for each minority class
        for minority_class in minority_classes:
            # Train VAE on minority class
            vae = train_vae_for_minority(X_train, y_train, minority_class)
            
            # Generate synthetic samples
            X_balanced, y_balanced = generate_balanced_dataset(
                X_balanced, y_balanced, vae, minority_class, majority_count
            )
        
        # Print final class distribution
        print("\nFinal class distribution:")
        print(pd.Series(y_balanced).value_counts())
        
        # Save balanced dataset
        balanced_df = pd.DataFrame(X_balanced, columns=feature_names)
        balanced_df['class'] = y_balanced
        balanced_df.to_csv('balanced_dataset.csv', index=False)
        
        print("\nBalanced dataset saved to 'balanced_dataset.csv'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()