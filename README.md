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


**Implementation Steps for the Drone Pipeline (Draft)**

  * __Feature Extraction__: Convert raw ISAC channel state information (CSI) into an environmental fingerprint vector (AoA + ToA).
  * __ISAC KPI Monitoring__: Maintain a sliding window of the moving average of throughput and BLER.
  * __Buffer Injection__: If BLER surges > 5% while sensing data indicates stationary or predictable velocity, flag the current RF fingerprint as a "New Geometry Blockage" and store it.
  * __Edge Training__: During drone docking/charging cycles, run the EWC loss function using the stored buffer to update the local positioning model without destroying global performance.
