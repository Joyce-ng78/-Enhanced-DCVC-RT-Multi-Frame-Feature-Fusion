# Enhanced DCVC-RT: Multi-Frame Feature Fusion

This repository contains an enhanced implementation of the **DCVC-RT** (Deep Contextual Video Compression - Real Time) codec. [cite_start]The key innovation is the **Multi-Frame Deformable Fusion Module**, which improves video compression efficiency by leveraging temporal context from multiple past frames ($F_{t-1}, F_{t-2}, F_{t-3}$) instead of a single reference frame.

## 1. Architecture Overview

Standard DCVC-RT conditions its entropy model only on the immediately preceding frame ($F_{t-1}$). Our proposed method enriches this context by maintaining a history of the last **three** frames. [cite_start]It aligns the older frames to the current viewpoint using deformable convolutions and fuses them into a single, rich context tensor, $F_{tc}$.

### Fusion Module Diagram

🏗️ Architecture
System Overview
The enhanced system maintains DCVC-RT's real-time capability while adding multi-frame context:
Input Frames (x_t, x_t-1, x_t-2, x_t-3)
    ↓
Feature Extraction → Feature Buffer [F_t-1, F_t-2, F_t-3]
    ↓
Multi-Frame Fusion Module
    ├── Offset Prediction Network
    ├── Deformable Convolution (Alignment)
    └── 1×1 Conv (Fusion) → F_tc
    ↓
DCVC-RT Encoder (with F_tc)
    ├── Conditional Encoding
    ├── Hyper-Encoder/Decoder
    └── Entropy Model (using F_tc as prior)
    ↓
Compressed Bitstream
    ↓
DCVC-RT Decoder (with F_tc)
    ↓
Reconstructed Frame (x̂_t)**

## 2. Pipeline Flow

The fusion process operates entirely in the latent feature space following this pipeline:

### Step 1: Input Collection
[cite_start]The module retrieves the feature maps of the three most recent reconstructed frames from the Decoded Picture Buffer (DPB)[cite: 192]:
* **Anchor:** $F_{t-1}$ (The most recent past frame).
* **Support Frames:** $F_{t-2}, F_{t-3}$ (Older reference frames).

### Step 2: Joint Offset Prediction
* **Input:** Concatenation of $F_{t-1}, F_{t-2}, F_{t-3}$ along the channel dimension.
* **Operation:** A CNN predicts dense **offset fields** ($\Delta p$) for the support frames.
* **Purpose:** These offsets indicate the spatial displacement required to align pixels in the older frames ($F_{t-2}, F_{t-3}$) to the anchor frame ($F_{t-1}$), effectively compensating for motion without explicit optical flow[cite: 209, 211, 212].

### Step 3: Deformable Feature Alignment
* **Operation:** A Deformable Convolution (DCN) is applied to each support frame using the predicted offsets.
* **Formula:** $F'_{t-k}(p) = \sum w_k \cdot F_{t-k}(p + p_k + \Delta p_k)$.
* [cite_start]**Result:** This warps $F_{t-2}$ and $F_{t-3}$ to spatially align with the anchor $F_{t-1}$, creating aligned features $F'_{t-2}$ and $F'_{t-3}$[cite: 227, 231, 235].

### Step 4: Multi-Frame Feature Fusion
* **Operation:** The anchor $F_{t-1}$ and the aligned supports $F'_{t-2}, F'_{t-3}$ are concatenated and passed through a learned $1\times1$ convolution.
* **Output ($F_{tc}$):** A fused context tensor that aggregates high-confidence temporal information. [cite_start]This tensor replaces the original single-frame prior in the DCVC-RT entropy model[cite: 243, 244, 254].

## 3. Code Implementation (`video_model.py`)

The file `src/models/video_model.py` includes three major updates to support this flow:

1.  **`OffsetEstimator`**: A lightweight CNN to predict spatial offsets from concatenated feature history.
2.  **`DeformableFusion`**: Implements the alignment and fusion logic (Step 2-4 above).
3.  **`DMC` Class Updates**:
    * **Buffer Size:** `self.max_dpb_size` increased to **3** to store $F_{t-1}$ through $F_{t-3}$.
    * **Compression Loop:** Calls `get_fused_context()` to generate $F_{tc}$ before entropy coding.



