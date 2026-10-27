# ISAC-KPI-Gated-Continual-Learning-positioning

* Based on understanding, in a Continual Learning (CL) framework for drone or object positioning using Integrated Sensing and Communications (ISAC), the new learning framework system must continuously update its positioning models using real time radio/channel condition observations and Key Performance Indicators (KPIs) without forgetting previously learned environments (e.g., changes in weather/climate, structural blockages, or new multi-path reflections).

* The core design detail that bridges the gap between raw RF sensing and deep learning model updates is the KPI-Gated Experience Replay Buffer with Dynamic Task Allocation.

* **Implementation Steps for the Drone Pipeline (Draft)**

  * __Feature Extraction__: Convert raw ISAC channel state information (CSI) into an environmental fingerprint vector (AoA + ToA).
  * __ISAC KPI Monitoring__: Maintain a sliding window of the moving average of throughput and BLER.
  * __Buffer Injection__: If BLER surges > 5% while sensing data indicates stationary or predictable velocity, flag the current RF fingerprint as a "New Geometry Blockage" and store it.
  * __Edge Training__: During drone docking/charging cycles, run the EWC loss function using the stored buffer to update the local positioning model without destroying global performance.
