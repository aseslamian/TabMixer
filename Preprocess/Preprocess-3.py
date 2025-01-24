import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

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
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipeline with MinMaxScaler for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='if_binary', sparse=False), categorical_columns)
        ])
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = []
    if len(numerical_columns) > 0:
        feature_names.extend(numerical_columns.tolist())
    if len(categorical_columns) > 0:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
        feature_names.extend(cat_features.tolist())
    
    # Ensure all values are between 0 and 1
    X_transformed = np.clip(X_transformed, 0, 1)
    
    # Convert target to numeric (handle both binary and multi-class cases)
    unique_classes = y.unique()
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = y.map(class_mapping)
    
    return X_transformed, y_encoded, preprocessor, feature_names, class_mapping

def vae_loss(recon_x, x, mu, logvar):
    # Ensure input tensors are between 0 and 1
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 0, 1)
    
    # Calculate reconstruction loss (BCE)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Calculate KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae(X_train, batch_size=64, num_epochs=50, learning_rate=1e-3):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Calculate loss
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # Backward pass
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
    # File paths
    input_file = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\insurance+company(IO)\Data.csv'
    output_file = 'Insurance+Company.csv'
    
    try:
        # Preprocess data
        X_transformed, y_encoded, preprocessor, feature_names, class_mapping = preprocess_data(input_file)
        
        # Train VAE
        vae = train_vae(X_transformed)
        
        # Generate synthetic samples
        num_synthetic = len(X_transformed)
        synthetic_samples = generate_synthetic_samples(vae, num_synthetic, X_transformed.shape[1])
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_transformed, synthetic_samples])
        y_combined = np.concatenate([y_encoded, y_encoded])
        
        # Convert to DataFrame
        final_df = pd.DataFrame(X_combined, columns=feature_names)
        
        # Map classes back to original labels
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        final_df['class'] = pd.Series(y_combined).map(reverse_mapping)
        
        # Save results
        final_df.to_csv(output_file, index=False)
        print(f"Balanced dataset saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main()