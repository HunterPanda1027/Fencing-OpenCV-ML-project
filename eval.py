import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FencingLungeDataset, Fencing1DCNN
from sklearn.metrics import confusion_matrix, classification_report

TARGET_WINDOW_SIZE = 12

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load dataset and model
    dataset = FencingLungeDataset(csv_files=["labelled(right)_fencing_data_Limardo (2025).csv"], window_size=TARGET_WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    sample_x, _ = dataset[0]
    model = Fencing1DCNN(num_features=sample_x.shape[0], window_size=TARGET_WINDOW_SIZE).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load("fencing_lunge_cnn.pth", map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            probabilities = torch.softmax(outputs, dim=1)
            lunge_probs = probabilities[:, 1].cpu().numpy()

            predicted = np.where(lunge_probs >= 0.5, 1, 0)
            
            all_preds.extend(predicted)
            all_labels.extend(batch_y.numpy())
            
    print("\n📊 THE REALITY CHECK BREAKDOWN:")
    print("-" * 40)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"True Neutrals spotted: {tn}")
    print(f"Lunges MISSED (False Neutrals): {fn}")      # 🌟 Fixed: actual lunge predicted neutral
    print(f"Neutrals mistaken for Lunges: {fp}")      # 🌟 Fixed: actual neutral predicted lunge
    print(f"True Lunges SUCCESSFULLY caught: {tp}")
    
    # Target precision and recall specifically for class 1 (Lunges)
    print(classification_report(all_labels, all_preds, target_names=["Neutral", "Lunge"]))

if __name__ == "__main__":
    evaluate()