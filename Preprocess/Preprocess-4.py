import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from collections import Counter

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

def preprocess_data(file_path, target_column='class'):
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original class distribution:\n{df[target_column].value_counts()}")
    
    # Identify minority and majority classes
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.index[-1]
    majority_class = class_counts.index[0]
    
    print(f"\nMinority class: {minority_class} (Count: {class_counts[minority_class]})")
    print(f"Majority class: {majority_class} (Count: {class_counts[majority_class]})")
    
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
    
    # Ensure all values are between 0 and 1
    X_transformed = np.clip(X_transformed, 0, 1)
    
    # Get indices for minority and majority classes
    minority_indices = y[y == minority_class].index
    majority_indices = y[y == majority_class].index
    
    # Extract minority and majority class samples
    X_minority = X_transformed[minority_indices]
    X_majority = X_transformed[majority_indices]
    
    return (X_minority, X_majority, minority_class, majority_class, 
            preprocessor, feature_names, X_transformed, y)

def vae_loss(recon_x, x, mu, logvar):
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 0, 1)
    
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae(X_train, batch_size=32, num_epochs=100, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining VAE on device: {device}")
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize model parameters
    input_dim = X_train.shape[1]
    hidden_dim = max(128, input_dim * 2)
    latent_dim = max(10, input_dim // 4)
    
    # Create model
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Prepare data loader
    X_tensor = torch.FloatTensor(X_train).to(device)
    X_tensor = torch.clamp(X_tensor, 0, 1)
    
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return vae

def generate_synthetic_samples(vae, num_samples, input_dim, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, vae.fc_mu.out_features).to(device)
        synthetic_samples = vae.decode(z).cpu().numpy()
        synthetic_samples = np.clip(synthetic_samples, 0, 1)
    return synthetic_samples

def main():
    input_file = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\income(IC)\income_1995.csv'
    output_file = 'Income_balanced.csv'
    
    try:
        # Preprocess data and get minority/majority class samples
        (X_minority, X_majority, minority_class, majority_class, 
         preprocessor, feature_names, X_full, y_full) = preprocess_data(input_file)
        
        # Train VAE on minority class data
        print("\nTraining VAE on minority class data...")
        vae = train_vae(X_minority)
        
        # Calculate number of synthetic samples needed
        num_synthetic = len(X_majority) - len(X_minority)
        print(f"\nGenerating {num_synthetic} synthetic samples for minority class...")
        
        # Generate synthetic samples
        synthetic_minority = generate_synthetic_samples(vae, num_synthetic, X_minority.shape[1])
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_full, synthetic_minority])
        y_synthetic = np.array([minority_class] * num_synthetic)
        y_combined = np.concatenate([y_full, y_synthetic])
        
        # Convert to DataFrame
        final_df = pd.DataFrame(X_combined, columns=feature_names)
        final_df['class'] = y_combined
        
        # Verify final class distribution
        print("\nFinal class distribution:")
        print(final_df['class'].value_counts())
        
        # Save results
        final_df.to_csv(output_file, index=False)
        print(f"\nBalanced dataset saved to {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()