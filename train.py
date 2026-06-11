import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# Import the dataset and model architecture you just verified from dataset.py
from dataset import FencingLungeDataset, Fencing1DCNN

def train_model():
    # 1. SET DEVICE (Utilize Mac's Apple Silicon GPU if available for maximum speed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Using Apple Silicon GPU (MPS) for training!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU (CUDA) for training!")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU for training.")

    # 2. LOAD DATASET
    # ⚠️ Double-check that these match your exact filename strings!
    my_files = ["labelled_fencing_data_Kano (2025).csv", "labelled_fencing_data_Limardo (2025).csv"]
    print("\nLoading dataset files...")
    full_dataset = FencingLungeDataset(csv_files=my_files, window_size=15)
    
    # Extract feature count automatically from your validated data pipeline
    sample_x, _ = full_dataset[0]
    num_features = sample_x.shape[0]
    
    # 3. SPLIT DATA (80% Train to learn patterns, 20% Validation to test accuracy)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders group windows into batches so the computer processes them efficiently
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Dataset split completed: {train_size} training windows | {val_size} validation windows.")

    # 4. INITIALIZE MODEL & OPTIMIZATION TOOLS
    model = Fencing1DCNN(num_features=num_features).to(device)

    # 🌟 THE FIX: Tell the AI that a Lunge is roughly 66x more important to catch than Neutral
    weights = torch.tensor([1.0, 66.0], dtype=torch.float32).to(device)
    
    # CrossEntropyLoss handles imbalanced data classifications (lots of 0s, few 1s)
    criterion = nn.CrossEntropyLoss() 
    # Adam optimizer acts as the math steering wheel to adjust network weights smoothly
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. THE MAIN TRAINING LOOP
    epochs = 10  # Number of times the AI reviews the entire dataset
    print("\n⚡ Starting Training Phase...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train() # Put model in training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            # Send data batches to your selected processor (CPU or GPU)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Reset existing gradient calculations
            optimizer.zero_grad()
            
            # Forward Pass: Make predictions
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward Pass: Calculate calculus error rates and step weights forward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        epoch_loss = running_loss / train_size
        epoch_acc = (correct_train / total_train) * 100
        
        # 6. EVALUATION PHASE (Test the AI against unseen data to measure true skills)
        model.eval() # Put model in static testing mode
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Disable gradient updates to save memory
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                _, val_predicted = torch.max(val_outputs, 1)
                total_val += val_y.size(0)
                correct_val += (val_predicted == val_y).sum().item()
                
        val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 7. SAVE THE TRAINED BRAIN FILE
    model_save_path = "fencing_lunge_cnn.pth"
    torch.save(model.state_dict(), model_save_path)
    print("-" * 50)
    print(f"🎉 TRAINING COMPLETE! Neural network weights saved successfully to: '{model_save_path}'")

if __name__ == "__main__":
    train_model()