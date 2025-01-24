## A novel data augmentation approach to fault diagnosis with class-imbalance problem

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
from sklearn.metrics.pairwise import rbf_kernel

class WM_CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, dropout_rate=0.2):
        super(WM_CVAE, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
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
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x, labels):
        x_cond = torch.cat([x, labels], dim=1)  # Concatenate input with labels
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        z_cond = torch.cat([z, labels], dim=1)  # Concatenate latent space with labels
        return self.decoder(z_cond)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar
    
def wm_cvae_loss(recon_x, x, mu, logvar, kmm_weights=None):
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 0, 1)
    
    # Reconstruction Loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='none')
    if kmm_weights is not None:
        BCE = (BCE * kmm_weights).sum()  # Apply KMM weights to the reconstruction loss
    else:
        BCE = BCE.sum()
    
    # KL Divergence Loss with weight adaptation
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    weighted_KLD = KLD * 0.1  # Apply a weighting factor to avoid KL vanishing (adjustable)

    return BCE + weighted_KLD

def compute_kmm_weights(X_real, X_gen, gamma=0.5):
    """
    Compute KMM weights to assign higher weights to samples closer to real data.
    """
    kernel_real = rbf_kernel(X_real, X_real, gamma=gamma)
    kernel_gen = rbf_kernel(X_gen, X_real, gamma=gamma)
    
    # Calculate weights based on similarity to real data
    weights = np.mean(kernel_real, axis=1) / np.mean(kernel_gen, axis=1)
    weights = np.clip(weights, 0, 1)  # Normalize weights between 0 and 1
    return torch.FloatTensor(weights).unsqueeze(1)

def train_wm_cvae(X_train, y_train, num_classes, batch_size=32, num_epochs=100, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining WM-CVAE on device: {device}")
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize model parameters
    input_dim = X_train.shape[1]
    hidden_dim = max(128, input_dim * 2)
    latent_dim = max(10, input_dim // 4)
    
    # One-hot encode labels for conditional input
    y_train_onehot = torch.eye(num_classes)[y_train]
    
    # Create model
    wm_cvae = WM_CVAE(input_dim, hidden_dim, latent_dim, num_classes).to(device)
    optimizer = optim.Adam(wm_cvae.parameters(), lr=learning_rate)
    
    # Prepare data loader
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train_onehot).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    wm_cvae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Generate samples to calculate KMM weights
            with torch.no_grad():
                generated_samples, _, _ = wm_cvae(data, labels)
                kmm_weights = compute_kmm_weights(data.cpu().numpy(), generated_samples.cpu().numpy()).to(device)
            
            # Forward pass with labels as conditional inputs
            recon_batch, mu, logvar = wm_cvae(data, labels)
            loss = wm_cvae_loss(recon_batch, data, mu, logvar, kmm_weights)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return wm_cvae

def generate_synthetic_samples(wm_cvae, num_samples, input_dim, condition, device=None):
    """
    Generate synthetic samples for a specified class using the trained WM-CVAE model.
    
    Parameters:
    - wm_cvae (WM_CVAE): The trained WM-CVAE model.
    - num_samples (int): Number of synthetic samples to generate.
    - input_dim (int): Dimensionality of the input features.
    - condition (torch.Tensor): One-hot encoded tensor for the class label.
    - device (torch.device, optional): Device to perform computations on.
    
    Returns:
    - synthetic_samples (np.ndarray): Generated synthetic samples as a NumPy array.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wm_cvae.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Generate random latent space samples
        z = torch.randn(num_samples, wm_cvae.fc_mu.out_features).to(device)
        
        # Repeat the condition for all samples
        conditions = condition.to(device).repeat(num_samples, 1)
        
        # Decode latent space with the condition to generate synthetic samples
        synthetic_samples = wm_cvae.decode(z, conditions).cpu().numpy()
        
        # Clip values to ensure they stay in the [0, 1] range
        synthetic_samples = np.clip(synthetic_samples, 0, 1)
    
    return synthetic_samples


def load_and_preprocess_data(file_path, target_column):
    """Load and preprocess the dataset with minority/majority class separation"""
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    class_dist = df[target_column].value_counts()
    print(class_dist)
    
    # Identify minority and majority classes
    minority_class = class_dist.index[-1]
    majority_class = class_dist.index[0]
    print(f"\nMinority class: {minority_class} (Count: {class_dist[minority_class]})")
    print(f"Majority class: {majority_class} (Count: {class_dist[majority_class]})")
    
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
    unique_classes = sorted(y.unique())  # Sort to ensure consistent mapping
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = pd.Series(y.map(class_mapping))
    
    # Print class mapping for debugging
    print("\nClass mapping:", class_mapping)
    
    # Separate minority and majority samples using numeric indices
    minority_idx = class_mapping[minority_class]
    majority_idx = class_mapping[majority_class]
    
    minority_mask = y_encoded == minority_idx
    majority_mask = y_encoded == majority_idx
    
    X_minority = X_transformed[minority_mask]
    X_majority = X_transformed[majority_mask]
    
    # Add diagnostic information
    diagnose_class_distribution(y_encoded, "Encoded labels")
    
    return (X_minority, X_majority, minority_class, majority_class, 
            preprocessor, feature_names, X_transformed, y_encoded)

def main():
    # Paths and parameters
    # input_file = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\Credit-g(CG)\credit-g.csv'
    # input_file = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\Blastchar(BL)\Blastchar.csv'
    # input_file = r'C:\Users\Ali\Code\TransTAB\TabMixer\DATA\dress-sale(DS)\dress-sale.csv'
    input_file = r'/home/data3/Ali/Code/TabMixer-review/DATA/Adult(AD)/data_processed.csv'
    
    
    output_file = 'DS_Balanced_output.csv'
    target_column = 'target_label'
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-3
    
    try:
        # Load and preprocess data
        (X_minority, X_majority, minority_class, majority_class, 
         preprocessor, feature_names, X_full, y_full) = load_and_preprocess_data(
            input_file, target_column)
        
        # Get number of classes
        unique_classes = np.unique(y_full)
        num_classes = len(unique_classes)
        print(f"Number of classes: {num_classes}")
        print(f"Unique classes: {unique_classes}")
        print(f"Minority class: {minority_class}")
        
        # Convert y_full to numpy array if it's not already
        y_full_array = y_full.to_numpy() if hasattr(y_full, 'to_numpy') else np.array(y_full)
        
        # Train WM-CVAE on all data
        print("\nTraining WM-CVAE...")
        wm_cvae = train_wm_cvae(X_full, y_full_array, num_classes, 
                               batch_size, num_epochs, learning_rate)
        
        # Calculate number of synthetic samples needed
        num_synthetic = len(X_majority) - len(X_minority)
        print(f"\nGenerating {num_synthetic} synthetic samples for minority class...")
        
        # Find the minority class index directly from the class mapping
        class_counts = pd.Series(y_full_array).value_counts()
        minority_class_idx = np.argmin(class_counts.values)
        print(f"Minority class index: {minority_class_idx}")
        
        # Create one-hot encoded condition for minority class
        minority_condition = torch.zeros(num_classes)
        minority_condition[minority_class_idx] = 1
        
        # Generate synthetic samples
        synthetic_minority = generate_synthetic_samples(
            wm_cvae, num_synthetic, X_minority.shape[1], minority_condition)
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_full, synthetic_minority])
        y_synthetic = np.full(num_synthetic, minority_class)
        y_combined = np.concatenate([y_full_array, y_synthetic])
        
        # Create final DataFrame
        final_df = pd.DataFrame(X_combined, columns=feature_names)
        final_df['target_label'] = y_combined
        
        # Print final distribution
        print("\nFinal class distribution:")
        print(final_df['target_label'].value_counts())
        
        # Save results
        final_df.to_csv(output_file, index=False)
        print(f"\nBalanced dataset saved to {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

# Add diagnostic function to help debug class distribution issues
def diagnose_class_distribution(y_data, label):
    """Helper function to diagnose class distribution"""
    print(f"\nDiagnosing {label}:")
    if isinstance(y_data, pd.Series):
        print("Data type: pandas.Series")
        print("Value counts:")
        print(y_data.value_counts())
    elif isinstance(y_data, np.ndarray):
        print("Data type: numpy.ndarray")
        print("Unique values:", np.unique(y_data))
        print("Value counts:")
        values, counts = np.unique(y_data, return_counts=True)
        for val, count in zip(values, counts):
            print(f"{val}: {count}")
    else:
        print(f"Data type: {type(y_data)}")

if __name__ == "__main__":
    main()