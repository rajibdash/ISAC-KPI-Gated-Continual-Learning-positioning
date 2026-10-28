# ISAC KPI/Observation Gated Continual Learning about positioning (Outdoor)

## Purpose
* Based on understanding, in a Continual Learning (CL) framework for drone or object positioning using Integrated Sensing and Communications (ISAC), the new learning framework system must continuously update its positioning models using real time radio/channel condition observations and Key Performance Indicators (KPIs) without forgetting previously learned environments (e.g., changes in weather/climate, structural blockages, or new multi-path reflections).

* The core design detail that bridges the gap between raw RF sensing and deep learning model updates is the KPI-Gated Experience Replay Buffer with Dynamic Task Allocation.

## Simulation environment & other requirements
* Trying to deveop this framework using a specific simulation testbed (e.g., Sionna, MATLAB, OpenAirInterface). 
* What type of input data need to be worked with? i.e., positioning KPIs 
* Which specific algorithm need to be considered for the KPI gating logic like Statistical distance checks, clustering, or confidence score drops etc.

## Layout architecture

```
[Raw ISAC Echoes/CSI] ──> [Positioning Neural Net] ──> [Position/Velocity Output]
                                   │                              │
                                   ▼                              ▼
                       [Observation Analytics]   ───>   [KPI Evaluation Engine]
                                   │                              │
                                   └─────────[Gating Logic]───────┘
                                                  │
                                    (Trigger if Accuracy Drops)
                                                  │
                                                  ▼
                                  [Continual Learning Engine]
                             (Fine-tuning / Replay-based Updates)

```

**The KPI & Observation Evaluation Engine**
 * Rationale:
 * Function:

**The Gating Mechanism**
 * Rationale:
 * Function:

**Continual Learning (CL) Engine**
 * Rationale:
 * Function:

**Simulation testbed Sionna layout:**

```

[ Outdoor ISAC Environment (Sionna) ]
          │                      │
          ▼                      ▼
    Sensing Metrics        Comm Metrics (CSI/RSRP)
          │                      │
          └───────────► [ KPI-Gate ] ◄───────────┘
                             │
                    ┌────────┴────────┐
                    ▼ Over Threshold  ▼ Under Threshold
             [Trigger Retraining]  [Freeze Weights / Inference Only]
                    │
                    ▼
          [Experience Replay Buffer]
                    │
                    ▼
          [Continual Learning Model] ──► Predicted Coordinates (X, Y, Z)


```

**Draft code**
  * By introducing a KPI/Observation Gate, we update the positioning network only when sensing/communication performance drops or when novel, high-confidence observations are detected.

```python

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Assuming sionna and mitsuba are configured in the environment
import sionna
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray

# 1. SETUP SIMULATION ENVIRONMENT (NVIDIA SIONNA)
def setup_sionna_scene():
    """Initializes the outdoor 3D scene for ISAC simulation."""
    # Load a preview or standard outdoor scene (e.g., an urban canyon)
    scene = Scene() # In full deployment, load a specific interactive .xml/ .ply scene
    
    # Configure 28 GHz Millimeter Wave Carrier
    scene.frequency = 28e9 
    
    # Add a Base Station (Transmitter & Sensing Receiver)
    tx = Transmitter(name="BS", position=[0.0, 0.0, 25.0], orientation=[0.0, 0.0, 0.0])
    tx.array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5)
    scene.add(tx)
    
    return scene

# 2. KPI/OBSERVATION GATING & MODEL DESIGN
class KPIGatedPositioningNetwork(nn.Module):
    """Deep Neural Network for 3D outdoor positioning."""
    def __init__(self, input_dim=64):
        super(KPIGatedPositioningNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Outputs: Estimated X, Y, Z coordinates
        )
        
    def forward(self, x):
        return self.fc(x)

class ContinualLearningManager:
    """Manages experience replay buffer and KPI-gated model updates."""
    def __init__(self, model, lr=0.001, rsrp_threshold=-95.0):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.rsrp_threshold = rsrp_threshold
        self.buffer = [] # Experience Replay Buffer for Continual Learning
        
    def check_kpi_gate(self, current_rsrp, sensing_snr):
        """
        Gate Decision Logic:
        Trigger training if communications degrade (RSRP drops) 
        BUT sensing remains reliable (High SNR) to validate position labels.
        """
        if current_rsrp < self.rsrp_threshold and sensing_snr > 10.0:
            return True # Trigger Model Update (Gate Open)
        return False # Maintain current weights (Gate Closed)

    def update_model(self, features, true_coordinates):
        """Executes a continual learning step using current data + memory buffer."""
        self.model.train()
        
        # Experience Replay: Blend old data to prevent catastrophic forgetting
        if len(self.buffer) > 0:
            old_feat, old_coord = zip(*self.buffer[-10:]) # Fetch last 10 samples
            features = torch.cat([features, torch.tensor(np.array(old_feat), dtype=torch.float32)], dim=0)
            true_coordinates = torch.cat([true_coordinates, torch.tensor(np.array(old_coord), dtype=torch.float32)], dim=0)
        
        # Optimize
        self.optimizer.zero_grad()
        predictions = self.model(features)
        loss = self.criterion(predictions, true_coordinates)
        loss.backward()
        self.optimizer.step()
        
        # Add current high-confidence sample to replay buffer
        self.buffer.append((features[0].detach().numpy(), true_coordinates[0].detach().numpy()))
        return loss.item()

# 3. EXECUTE SIMULATION PIPELINE
if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize components
    scene = setup_sionna_scene()
    position_net = KPIGatedPositioningNetwork(input_dim=64)
    cl_manager = ContinualLearningManager(position_net)
    
    print("Simulating Outdoor Mobility Trajectory...")
    # Mocking trace steps through a changing environmental clutter block
    for step in range(1, 6):
        # Generate synthetic RSRP (dBm) and Sensing SNR (dB) values
        mock_rsrp = -90.0 - (step * 2.5)   # Simulating degradation over time
        mock_sensing_snr = 15.0 - (step * 0.5) # Sensing stays relatively strong
        
        # Generate random complex-valued channel profile flattened to 64 features
        mock_csi_features = torch.randn(1, 64) 
        true_xyz = torch.tensor([[12.0 * step, 5.0 * step, 1.5]], dtype=torch.float32)
        
        # Evaluate Gate Condition
        gate_triggered = cl_manager.check_kpi_gate(mock_rsrp, mock_sensing_snr)
        
        if gate_triggered:
            loss = cl_manager.update_model(mock_csi_features, true_xyz)
            print(f"Step {step}: [GATE OPENED] RSRP={mock_rsrp:.1f} dBm | Model adapted. Loss: {loss:.4f}")
        else:
            print(f"Step {step}: [GATE CLOSED] RSRP={mock_rsrp:.1f} dBm | Using frozen weights for Inference.")

```

**Implementation Steps for the Drone Pipeline (Draft)**

  * __Feature Extraction__: Convert raw ISAC channel state information (CSI) into an environmental fingerprint vector (AoA + ToA).
  * __ISAC KPI Monitoring__: Maintain a sliding window of the moving average of throughput and BLER.
  * __Buffer Injection__: If BLER surges > 5% while sensing data indicates stationary or predictable velocity, flag the current RF fingerprint as a "New Geometry Blockage" and store it.
  * __Edge Training__: During drone docking/charging cycles, run the EWC loss function using the stored buffer to update the local positioning model without destroying global performance.
