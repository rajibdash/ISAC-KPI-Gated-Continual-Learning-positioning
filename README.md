# ISAC KPI/Observation Gated Continual Learning about positioning (Outdoor)

> The conceptual diagrams and draft below are preserved for motivation.
> The tested Python implementation is described in [Implementation](#implementation)
> and [Quickstart](#quickstart). It supersedes the draft's RSRP-only trigger:
> communication degradation alone does **not** authorize training, and high sensing
> SNR alone does **not** establish label reliability.

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
  * Raw ISAC measurements are noisy and can degrade for reasons that do not require model retraining (e.g., temporary blockage, fading, or link fluctuations). The evaluation engine converts these raw observations into reliable adaptation evidence by combining communication KPIs (e.g., RSRP/BLER), sensing quality indicators, label reliability, and positioning error/drift trends over a rolling window.
* Function:
  * Normalizes incoming feature vectors using frozen calibration statistics from initial training data.
Maintains an eligible rolling window of observations (only when sensing quality and label confidence satisfy minimum requirements).
Computes adaptation signals: mean positioning error, feature-drift score, and communication degradation flags.
Rejects low-trust observations from adaptation evidence and outputs structured evaluation signals for the gate (ready, label_ok, comm_bad, drift_score, position_error_m).

**The KPI Gating Mechanism**

* Rationale:
  * Continuous retraining is expensive and risky: it can overfit transient events and increase catastrophic forgetting. A gate is required to ensure that updates happen only when there is persistent, validated evidence of environment/task change—not just short-term communication drops.
* Function:
  * Applies policy rules to evaluation outputs and decides whether adaptation is allowed.
Requires persistence across consecutive windows (error/drift must remain elevated).
Enforces cooldown between adaptation attempts to prevent update loops.
Requires a minimum number of pending reliable labeled samples before training.
Emits interpretable reason codes (stable, warming_up, communication_only_degradation, waiting_for_persistence, cooldown, collecting_labeled_samples, position_error, feature_drift).

**Continual Learning (CL) Engine**

* Rationale:
  * The system must adapt to genuine distribution shifts while preserving prior positioning performance. Updating the deployed model directly is unsafe; adaptation should first be tested on a candidate and accepted only when it improves recent conditions without unacceptable regression on historical data.
* Function:
  * Stores reliable online samples in a pending buffer and historical samples in a bounded replay memory (reservoir sampling).
Trains an isolated candidate model on a mix of pending + replay data (deployed model remains unchanged during training).
Validates candidate performance on both recent and historical validation sets using 3D RMSE.
Deploys candidate only if recent performance improves beyond threshold and historical regression stays within tolerance; otherwise rejects and keeps current model.
Provides adaptation outcomes (accepted, validation_rejected, insufficient_data, numerical_failure) and clears/commits buffers according to policy.

**Simulation testbed Sionna layout:**

```

    [Outdoor ISAC Environment (Sionna)]
          │                      │
          ▼                      ▼
   Sensing Metrics]        [Comm Metrics (CSI/RSRP)]
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
  * SINR drop was omitted from the gating metrics to prevent the system from getting trapped in an infinite training loop caused by non-line-of-sight (NLOS) blockages.
  * Instead of SINR, the architecture would rely on the decoupling of RSRP and Sensing SNR(Radar Echo Quality):

```

         [Environmental Event]
                 │
        ┌────────┴──────--------──┐
        ▼                         ▼
 [Temporary Blockage]     [True Spatial Layout Change]
 (e.g., Passing Bus)      (e.g., Turning into a new street)
        │                         │
        ├─► RSRP Drops            ├─► RSRP Drops Permanently
        └─► Sensing SNR Drops     └─► Sensing SNR Stays High (New reflections)
        │                         │
   [KPI-GATE CLOSED]           [KPI-GATE OPENS: Trigger CL Model]
  (Protect the CL model)           (Adapt to new environment)

```


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

## Implementation

The `src/isac_cl` package implements a replay-based baseline, not EWC or dynamic
task allocation. Only NumPy and PyTorch are runtime dependencies; Sionna/MATLAB
can supply observations without being imported by the controller.

| Module | Architecture responsibility |
|---|---|
| `config.py` | Validated immutable policy thresholds and training/buffer settings |
| `models.py` | `Observation`, `Evaluation`, shape/finite-value helpers, and `PositionNet` MLP |
| `evaluation.py` | Frozen calibration normalization, eligible rolling window, mean position error, communication flags, and drift |
| `gating.py` | Consecutive-window evidence, minimum pending data, and attempt cooldown |
| `continual.py` | Uniform reservoir replay, pending FIFO, isolated candidate training, and validation-controlled deployment |
| `pipeline.py` | `process_observation`: inference → evaluation → admission → gating → optional adaptation |

Features are finite real vectors `[D]` (split complex CSI into real/imaginary
components upstream); labels/predictions are XYZ in meters. Training/validation
arrays have shapes `[N,D]` and `[N,3]`. Invalid shapes or nonfinite inputs raise
`ValueError`. Reference statistics require at least two calibration samples and
never update online. Drift is the RMS of the rolling mean of standardized
features; position error is mean Euclidean distance over eligible observations.
Candidate selection instead uses **3D positioning RMSE**, not coordinate RMSE.

An eligible label requires coordinates, sufficient sensing SNR, and independently
calibrated label confidence. Unreliable observations clear the evaluation window
and gate streak, but do not discard previously admitted pending data.
Communication degradation uses linear-power-averaged RSRP or the mean of available
BLER measurements (missing BLER is ignored). During unreliable-label outages,
evaluation returns no adaptation evidence, including `comm_bad=False`.

The gate requires persistent error **or** drift; communication alone is monitor-only.
Persistence counts consecutive eligible rolling windows, which can overlap.
Use strictly increasing nonnegative integer observation steps; cooldown counts
steps, not seconds, and starts only when an adaptation is attempted, including
rejections. Missing validation postpones training without starting cooldown.

The candidate is a deep copy trained with equally sized fresh and replay batches,
sampled with replacement. It is deployed only if recent RMSE strictly improves
by more than `min_recent_improvement_m` and old RMSE increases by no more than
`max_old_regression_m`. Nonfinite losses, gradients, model state, or validation
metrics reject it. Failed numeric attempts report `numerical_failure` (unavailable
after-metrics are infinity). Invalid validation arrays raise before training.
Completed attempts, accepted or rejected, insert pending observations into replay
and clear pending; insufficient data leaves both buffers unchanged. Pending
overflow evicts the oldest samples; replay uses bounded uniform reservoir sampling.

Gate reason codes are `unreliable_observation_or_label`, `warming_up`, `stable`,
`communication_only_degradation`, `waiting_for_persistence`, `cooldown`,
`collecting_labeled_samples`, `position_error`, and `feature_drift`. The pipeline
also reports `waiting_for_independent_validation`; adaptation outcomes separately
report `insufficient_data`, `accepted`, `validation_rejected`, or `numerical_failure`.
The returned prediction is always the pre-adaptation deployed-model prediction.

## Quickstart

Python 3.10+ is required. From the repository root, install CPU PyTorch first to
avoid downloading accelerator libraries, then install the package:

```bash
python -m pip install "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

This self-contained synthetic example initializes an offline model and processes
a shifted stream. It is an API demonstration, not an ISAC performance benchmark.

```python
import numpy as np
import torch
from isac_cl import (
    Config, Observation, PositionNet, EvaluationEngine, KPIGate,
    ContinualLearningEngine, process_observation,
)

rng = np.random.default_rng(42)
torch.manual_seed(42)
torch.set_num_threads(1)

def dataset(n, shift=0.0):
    x = rng.normal(shift, 1, (n, 2)).astype(np.float32)
    y = np.column_stack((x, np.zeros(n))).astype(np.float32)
    return x, y

train_x, train_y = dataset(64)
old_validation = dataset(32)
recent_validation = dataset(32, shift=2)
config = Config(window=4, persistence=2, min_new_samples=4, train_steps=40)
evaluator = EvaluationEngine(train_x, config)
model = PositionNet(feature_dim=2)  # CPU by default

# Offline initialization uses only training data and the frozen normalization.
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(
        model(torch.from_numpy(evaluator.normalize(train_x))),
        torch.from_numpy(train_y),
    )
    loss.backward()
    optimizer.step()

learner = ContinualLearningEngine(
    model, evaluator, config, train_x, train_y, seed=42,
)
gate = KPIGate(config)
online_x, online_y = dataset(12, shift=2)
for step, (x, xyz) in enumerate(zip(online_x, online_y)):
    observation = Observation(
        features=x, rsrp_dbm=-100, sensing_snr_db=20,
        label_confidence=1.0, xyz_m=xyz, bler=0.2,
    )  # Synthetic oracle labels; confidence is not inferred from SNR.
    result = process_observation(
        step, observation, evaluator, gate, learner,
        old_validation=old_validation, recent_validation=recent_validation,
    )
    print(step, result["reason"], result["adaptation"])
```

For Sionna or MATLAB integration, replace `dataset` and the observation loop with
extracted communication fingerprints, waveform-measured RSRP/BLER, independently
modeled sensing SNR, and explicitly identified oracle or calibrated position labels.
Do not treat uncalibrated channel gain as RSRP or radar sensing SNR.

### Assumptions and limitations

- Threshold defaults are illustrative research policies, not ISAC standards.
- Validation must be disjoint from initial training, pending, and replay samples.
  Split by trajectory/time/environment to avoid near-duplicate leakage; the
  package cannot infer dataset provenance. In online experiments, do not expose
  future validation samples before they become available. Keep a separate final
  test set untouched by candidate selection.
- No supervised adaptation during label outages. High SNR does not replace an
  independent label source.
- Mean-shift drift detection misses covariance/multimodal changes and can
  repeatedly flag a legitimate new environment after cooldown.
- Full waveform KPI extraction, radar sensing, scene/trajectory simulation,
  task allocation, EWC, checkpointing, asynchronous training, and production
  scheduling are out of scope. Engine instances are synchronous, not thread-safe.
- Seed NumPy/PyTorch for reproducibility. Replay/batch sampling uses local seeded
  RNGs; stochastic custom models also depend on PyTorch's RNG and device behavior.

## Testing

After installing the test extra, run from the repository root:

```bash
python -m pytest -q
# Focused engine tests:
python -m pytest -q tests/test_evaluation.py tests/test_gating.py
```

Tests run on CPU with fixed seeds and controlled synthetic fixtures. They cover
readiness, label outages, RSRP/BLER, drift, reason codes, persistence/cooldown,
sample admission, validation acceptance/rejection, candidate isolation, numerical
failures, buffer bounds, reservoir determinism, invalid inputs, and orchestration.
