import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class StellarFlareTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, max_seq_length=1000):
        super().__init__()
        
        # Input embedding layer to convert 1D brightness to d_model dimensions
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary classification: flare or no flare
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, src_mask=None, src_padding_mask=None):
        # src shape: (batch_size, seq_length, input_dim)
        
        # Embed input
        x = self.input_embedding(src)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform through encoder layers
        x = self.transformer_encoder(x, src_mask, src_padding_mask)
        
        # Global average pooling across sequence length
        x = torch.mean(x, dim=1)
        
        # Final classification layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x

def create_flare_detection_model(config=None):
    if config is None:
        config = {
            'input_dim': 1,          # Single brightness measurement
            'd_model': 256,          # Transformer embedding dimension
            'nhead': 8,              # Number of attention heads
            'num_layers': 6,         # Number of transformer layers
            'dim_feedforward': 1024, # Feedforward network dimension
            'max_seq_length': 1000   # Maximum sequence length
        }
    
    model = StellarFlareTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        max_seq_length=config['max_seq_length']
    )
    
    return model

# Example usage:
def train_step(model, optimizer, criterion, batch_data, batch_labels):
    optimizer.zero_grad()
    
    # Assuming batch_data shape: (batch_size, seq_length, 1)
    predictions = model(batch_data)
    loss = criterion(predictions, batch_labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    # Create model
    model = create_flare_detection_model()
    
    # Example hyperparameters
    learning_rate = 1e-4
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        # Training loop
    def train_model(model, train_loader, val_loader, criterion, optimizer, 
                   num_epochs, device='cuda', patience=5):
        model = model.to(device)
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                predictions = model(batch_data)
                loss = criterion(predictions, batch_labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                predicted = (predictions > 0.5).float()
                train_correct += (predicted == batch_labels).sum().item()
                train_total += batch_labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    predictions = model(batch_data)
                    loss = criterion(predictions, batch_labels)
                    
                    val_loss += loss.item()
                    predicted = (predictions > 0.5).float()
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Store training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_accuracy,
                'val_loss': avg_val_loss,
                'val_acc': val_accuracy
            })
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print('-' * 60)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_flare_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after epoch {epoch+1}')
                    break
        
        return training_history

    # Set training parameters
    num_epochs = 50
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 5

    # Create data loaders (assuming you have your data prepared)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    # history = train_model(model, train_loader, val_loader, criterion, optimizer, 
    #                      num_epochs, device, patience)
    #         loss = train_step(model, optimizer, criterion, batch_data, batch_labels)