import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

TARGET_WINDOW_SIZE = 12

# ==========================================
# 1. THE REPAIRED DATA PIPELINE
# ==========================================
class FencingLungeDataset(Dataset): # Matched to your exact error name!
    def __init__(self, csv_files, window_size=12, step_size=1):
        self.window_size = window_size
        self.step_size = step_size
        self.X = []
        self.y = []
        
        for file in csv_files:

            df = pd.read_csv(file).dropna(subset=['action_label'])
            features_df = df.drop(columns=['timestamp_ms', 'action_label'], errors='ignore')
            
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df = features_df.fillna(0)
            

            if "left" in file:
                features_df = features_df.iloc[:, :34]

                final_df = pd.DataFrame(index = features_df.index)

                #tracking hip frop and distance between ankles
                final_df["hip_y"] = features_df["lrh_y"] 
                final_df['ankle_distance'] = (features_df['lra_x'] - features_df['lla_x']).abs()

                #tracking velocities
                final_df["shoulder_velocity"] = features_df["lrs_x"].diff().fillna(0) - features_df["cam_shift"]
                final_df["wrist_velocity"] = features_df["lw_x"].diff().fillna(0) - features_df["cam_shift"]
                final_df["front_ankle_velocity"] = features_df["lra_x"].diff().fillna(0) - features_df["cam_shift"]
            else:
                features_df = features_df.iloc[:, np.r_[0, -33:0]]
                x_cols = [col for col in features_df.columns if '_x' in col.lower() or 'x_' in col.lower()]
                features_df[x_cols] = 1.0 - features_df[x_cols] #mirroring

                final_df = pd.DataFrame(index = features_df.index)

                 #tracking hip frop and distance between ankles
                final_df["hip_y"] = features_df["rrh_y"] 
                final_df['ankle_distance'] = (features_df['rra_x'] - features_df['rla_x']).abs()

                #tracking velocities
                final_df["shoulder_velocity"] = features_df["rrs_x"].diff().fillna(0) + features_df["cam_shift"]
                final_df["wrist_velocity"] = features_df["rw_x"].diff().fillna(0) + features_df["cam_shift"]
                final_df["front_ankle_velocity"] = features_df["rra_x"].diff().fillna(0) + features_df["cam_shift"]
            
            feature_matrix = final_df.to_numpy(dtype=np.float32)
            
            # Force labels to integers, converting any accidental text errors to 0
            df['action_label'] = pd.to_numeric(df['action_label'], errors='coerce').fillna(0)
            labels = df['action_label'].to_numpy(dtype=np.int64)
            
            num_rows = len(df)
            
            # Slice continuous data into overlapping windows
            for start_idx in range(0, num_rows - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_features = feature_matrix[start_idx:end_idx].copy()

                window_labels = labels[start_idx : start_idx + TARGET_WINDOW_SIZE]

                # Label assignment: If any frame is a lunge (1), the window is a lunge
                target_label = 1 if np.sum(window_labels == 1) >= (TARGET_WINDOW_SIZE / 2) else 0
                
                self.X.append(window_features)
                self.y.append(target_label)
                
        # CRITICAL FIX: Ensuring self.X and self.y are permanently attached to the class
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32) 
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)    

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Transpose from (Time, Features) to PyTorch-expected (Features, Time)
        return self.X[idx].permute(1, 0), self.y[idx]


# ==========================================
# 2. THE AI ARCHITECTURE
# ==========================================
class Fencing1DCNN(nn.Module):
    def __init__(self, num_features, window_size):
        super(Fencing1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.4)
        flattened_frames = window_size // 2
        self.fc1 = nn.Linear(64 * flattened_frames, 32)
        self.fc2 = nn.Linear(32, 2) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 3. VERIFICATION ENGINE
# ==========================================
if __name__ == "__main__":
    # ⚠️ CHANGE THESE NAMES TO YOUR EXACT LABELED CSV FILENAMES!
    my_files = ["labelled(left)_fencing_data_Kano (2025).csv", "labelled(right)_fencing_data_Limardo (2025).csv"] 
    
    try:
        dataset = FencingLungeDataset(csv_files=my_files, window_size=TARGET_WINDOW_SIZE)
        print("🎉 SUCCESS: Data Pipeline Initialized Safely!")
        print(f"-> Created {len(dataset)} total motion windows.")
        
        sample_x, sample_y = dataset[0]
        num_features = sample_x.shape[0]
        print(f"-> Detected tracking features per frame: {num_features}")
        print(f"-> Total distinct lunge-labeled windows: {int(torch.sum(dataset.y))}")
        
        model = Fencing1DCNN(num_features=num_features, window_size=TARGET_WINDOW_SIZE)
        print("\n🤖 SUCCESS: 1D-CNN Model Structure Built!")
        
        mock_batch = sample_x.unsqueeze(0) 
        mock_output = model(mock_batch)
        print(f"-> Model Output Tensor Shape: {mock_output.shape} (Successfully Processed!)")
        
    except Exception as e:
        print(f"❌ PIPELINE ERROR: {e}")