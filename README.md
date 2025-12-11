# Enhanced DCVC-RT: Multi-Frame Feature Fusion

This repository contains an enhanced implementation of the **DCVC-RT** (Deep Contextual Video Compression - Real Time) codec. The key innovation is the **Multi-Frame Deformable Fusion Module**, which improves video compression efficiency by leveraging temporal context from multiple past frames ($F_{t-1}, F_{t-2}, F_{t-3}$) instead of a single reference frame .

## 1. Architecture Overview

Standard DCVC-RT conditions its entropy model only on the immediately preceding frame ($F_{t-1}$). Our proposed method enriches this context by maintaining a history of the last **three** frames. [cite_start]It aligns the older frames to the current viewpoint using deformable convolutions and fuses them into a single, rich context tensor, $F_{tc}$.

## 2. Fusion Module Diagram

The fusion process operates entirely in the latent feature space as illustrated above. The pipeline consists of four distinct stages:

### Step 1: Input Collection
The module retrieves the feature maps of the three most recent reconstructed frames from the Decoded Picture Buffer (DPB):
**Anchor ($F_{t-1}$):** The most recent past frame.
**Support Frames ($F_{t-2}, F_{t-3}$):** Older reference frames used to enrich context.

### Step 2: Joint Offset Prediction
* **Input:** Concatenation of $F_{t-1}, F_{t-2}, F_{t-3}$ along the channel dimension.
* **Operation:** A CNN predicts dense **offset fields** (labeled "Offset Field" in the diagram).
* **Purpose:** These offsets ($K \times K$ vectors) indicate the spatial displacement required to align pixels in the older frames ($F_{t-2}, F_{t-3}$) to the anchor frame ($F_{t-1}$), compensating for motion without explicit optical flow.

### Step 3: Deformable Feature Alignment
* **Operation:** A Deformable Convolution (DCN) is applied to each support frame using the predicted offsets.
* **Visual Reference:** See "Deformable Conv" and "Aligned Features" in the diagram.
* **Result:** This warps $F_{t-2}$ and $F_{t-3}$ to spatially align with the anchor $F_{t-1}$, creating aligned features.

### Step 4: Multi-Frame Feature Fusion
* **Operation:** The anchor $F_{t-1}$ and the aligned supports are concatenated and passed through a learned $1\times1$ convolution ("Conv 1x1").
* **Output ($F_{tc}$):** A fused context tensor that aggregates high-confidence temporal information to replace the single-frame prior.

## 3. Code Implementation (`video_model.py`)

The file `src/models/video_model.py` includes three major updates to support this flow:

1.  **`OffsetEstimator`**: A lightweight CNN to predict spatial offsets from concatenated feature history[cite: 218].
2.  **`DeformableFusion`**: Implements the alignment and fusion logic shown in the diagram.
3.  **`DMC` Class Updates**:
    * **Buffer Size:** `self.max_dpb_size` increased to **3** to store $F_{t-1}$ through $F_{t-3}$.
    * **Compression Loop:** Calls `get_fused_context()` to generate $F_{tc}$ before entropy coding.
Checkpoints: https://drive.google.com/drive/folders/1Y0eSk_HLBnm3el9theUJ6iufyhiev8UB?usp=drive_link




