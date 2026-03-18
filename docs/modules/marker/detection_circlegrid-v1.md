# Circle Grid Detection and Identification Pipeline

**Version:** 1.0
**Module:** `marker/detection.cpp`, `identification/board_circle/identification_circle.cpp`
**Entry point:** `detection::detect_and_identify_circlegrid()`

## 1. Problem Statement

Detecting and identifying markers on an asymmetric circle grid calibration target from monocular camera frames. The target consists of `R x C` circles arranged in a hex-like pattern where physical positions follow:

$$x = (2c + r \bmod 2) \cdot s, \quad y = r \cdot s$$

where $s$ is the circle spacing, $r$ is the row index, and $c$ is the column index. Each marker has a unique global ID: $\text{gid} = r \cdot C + c$.

### Core Challenges

1. **Orientation ambiguity**: For near-square grids (aspect ratio < 1.3), the grid looks nearly identical under 180-degree rotation. A 10x5 asymmetric grid at 56 mm spacing produces a 504x504 mm board &mdash; effectively square. The homography reprojection cost for both orientations is nearly identical (~2494 each), making cost-based disambiguation impossible.

2. **Row-pair swap**: OpenCV's `findCirclesGrid` with `CALIB_CB_ASYMMETRIC_GRID` may return odd rows before even rows, producing a systematic row-pair permutation.

3. **Adjacent-marker swaps**: When edge markers go off-screen, Hungarian tracking can assign a remaining marker to its disappeared neighbor's ID. On a hex grid with ~30 px inter-marker spacing and ~25-35 px row separation, this is within typical matching thresholds.

4. **Cascading swaps**: A single disappeared marker can trigger a chain: M0 disappears, M1 takes M0's spot, M6 takes M1's spot. Each link must be detected and broken.

## 2. Pipeline Overview

```
Input Image
    |
    v
[1] Brightness Scale Loop
    |   - Multiple brightness levels (0.67x, 0.8x, 1.0x, 1.2x, 1.5x)
    |   - Adaptive binarization + blob detection
    |   - Test findCirclesGrid at each level
    |
    v
[2] Primary Identification (findCirclesGrid)
    |   - Row-pair swap correction (cost-based)
    |   - 180-degree ambiguity resolution (immutable reference)
    |
    v--- if primary failed or weak (<50% identified)
    |
[3] Hungarian Tracking
    |   - Velocity-predicted cost matrix
    |   - Jonker-Volgenant O(n^3) assignment
    |   - RANSAC homography validation
    |   - Bidirectional velocity acceptance check
    |
    v
[4] 180-degree Ambiguity Resolution
    |   - Velocity variance comparison
    |
    v
[5] Local Homography Re-identification
    |   - Global RANSAC homography projection
    |   - Local 4-neighbor homography fallback
    |
    v
[6] Post-Homography Velocity Filter       [near-square only]
    |   - Re-validates homography-added markers
    |     against the similarity motion model
    |
    v
[7] FCG-Reference Row-Pair Swap Correction
    |   - RANSAC homography from board coords
    |   - Adjacent-row cost comparison
    |
    v
[8] Disappeared-Neighbor Swap Detection    [near-square only]
    |   - Hex-grid 6-connectivity
    |   - Velocity-field motion compensation
    |   - Cascading passes (up to 5 levels)
    |
    v
[9] Tracker State Update
    |   - Per-marker track history (3-frame ring buffer)
    |   - Global similarity velocity field (RANSAC)
    |
    v
Output: identified markers, ordering matrix, debug artifacts
```

## 3. Brightness Scale Loop

**Motivation**: Calibration targets are often photographed under uneven lighting. A single binarization threshold fails when parts of the board are in shadow. Testing multiple brightness scalings increases detection robustness.

**Algorithm**:
1. For each scale factor $\beta \in \{1.0, 1.5, 0.67, 0.44, 0.3\}$:
   - Scale image: $I' = \text{clamp}(\beta \cdot I, 0, 255)$
   - Adaptive binarization with Gaussian window
   - Blob detection via `filter_as_objects_simple_descriptors_rings()`
   - Attempt `findCirclesGrid` (see Section 4)
2. Select the brightness level with the most detected coding markers, preferring results that pass `findCirclesGrid`.

## 4. findCirclesGrid and Row-Pair Swap

### 4.1 OpenCV findCirclesGrid

A `PredetectedBlobDetector` wrapper feeds pre-detected keypoints into OpenCV's `findCirclesGrid()`. The function attempts clustering-based detection first (faster for regular grids), falling back to the standard algorithm if clustering fails.

### 4.2 Row-Pair Swap Correction

**Problem**: For asymmetric grids, `findCirclesGrid` may return centers with odd rows preceding even rows within each row pair.

**Detection**: Compute a homography $H$ from board coordinates to detected centers for both orderings (original and swapped). The ordering with lower mean reprojection error wins.

**Swap formula** (for each center index $i$):
```
det_row = i / cols
det_col = i % cols
swapped_row = det_row + 1    if det_row is even
             det_row - 1    if det_row is odd
swapped_idx = swapped_row * cols + det_col
```

**Decision**: Swap if $\text{cost}_{\text{swapped}} < 0.9 \cdot \text{cost}_{\text{original}}$.

### 4.3 180-Degree Ambiguity Resolution

For a near-square grid, the IDENTITY and FLIP_180 orientations produce nearly identical homography costs. A sign-based or cost-based approach is mathematically impossible &mdash; both orientations project to the same dot products.

**Solution**: An immutable static reference from the first successful `findCirclesGrid` frame.

1. **First frame**: Store the detected centers as `immutable_reference`. Lock orientation 0.
2. **Subsequent frames**: Compare both orientations against `immutable_reference` by summing pairwise distances. The orientation with lower total distance wins.

**Why immutable**: The reference must never be updated, because a single wrong update would permanently flip all subsequent frames. The first `findCirclesGrid` frame is trusted because it passed OpenCV's internal consistency checks with full marker visibility.

**Non-square grids** (aspect > 1.3): Use cost-based comparison directly &mdash; the 20% margin ($\text{cost}_{\text{flip}} < 0.8 \cdot \text{cost}_{\text{identity}}$) is sufficient.

## 5. Hungarian Tracking

### 5.1 Motivation

When `findCirclesGrid` fails (partial occlusion, edge markers off-screen), frame-to-frame tracking maintains marker identity. The Hungarian algorithm provides globally optimal assignment.

### 5.2 Velocity-Predicted Cost Matrix

For each tracked marker $i$ with history $(p_0, t_0), (p_1, t_1), \ldots$, predict position at the current frame via least-squares velocity:

$$\hat{p}_i = p_{\text{last}} + v \cdot (t_{\text{curr}} - t_{\text{last}})$$

where $v$ is estimated from 2-3 historical positions.

**Cost matrix**: $C_{ij} = \| \hat{p}_i - q_j \|_2$ where $q_j$ is the $j$-th detected marker position.

### 5.3 Jonker-Volgenant Assignment

The assignment problem is solved via the Jonker-Volgenant algorithm in $O(n^3)$ time. The cost matrix is padded to square dimensions; unmatched entries are filled with `max_distance = 80 px`.

### 5.4 RANSAC Homography Validation

After assignment, a RANSAC homography is fitted from (prev_position, curr_position) pairs with an inlier threshold of 5 px. Only inlier matches are accepted. This rejects gross misassignments where the spatial transformation is inconsistent with the majority.

### 5.5 Bidirectional Velocity Acceptance Check

**Guard**: Only active on near-square grids (aspect < 1.3) where swap ambiguity exists. Non-square grids (5x7, aspect = 1.5) are exempt &mdash; `findCirclesGrid` is unambiguous for these.

For each accepted match $(gid, \text{curr\_idx})$:

1. **Forward check**: $\hat{p}_{\text{curr}} = p_{\text{prev}} + \mathbf{V}(p_{\text{prev}})$, error $= \|\hat{p}_{\text{curr}} - p_{\text{curr}}\|$
2. **Backward check**: $\hat{p}_{\text{prev}} = p_{\text{curr}} - \mathbf{V}(p_{\text{curr}})$, error $= \|\hat{p}_{\text{prev}} - p_{\text{prev}}\|$

**Tolerance** (displacement-proportional):

$$\tau = \min\!\Big(\max\!\big(0.5 \cdot \|\mathbf{V}(p_{\text{prev}})\|,\; 15\big),\; 0.05 \cdot L_{\text{short}}\Big)$$

where $L_{\text{short}}$ is the shorter image edge. The 50% displacement factor accounts for prediction error scaling with motion speed (acceleration effects, lens distortion). The 15 px floor prevents over-rejection at low speeds. The 5% cap prevents over-tolerance at high speeds.

**Reject if**: forward error > $\tau$ OR backward error > $\tau$.

## 6. Velocity Field: 2D Similarity Model

### 6.1 Motivation

All markers lie on a single rigid board. Their apparent velocities in the image are governed by a single rigid-body motion plus perspective-induced scale changes. Fitting a global model &mdash; rather than per-marker or local k-NN models &mdash; ensures physical consistency and provides robustness via RANSAC.

**Why not local k-NN?** A local fit to 5 nearest neighbors gives different $(v_C, \omega)$ for each query point, producing an inconsistent velocity field that violates rigid-body physics. Edge markers have biased neighbor sets. A single outlier track corrupts all nearby predictions.

### 6.2 Similarity Model

The velocity at point $\mathbf{p}$ is:

$$\mathbf{v}(\mathbf{p}) = \mathbf{v}_C + \omega \times \mathbf{r} + \sigma \cdot \mathbf{r}$$

where $\mathbf{r} = \mathbf{p} - \mathbf{C}$ is the position relative to the centroid $\mathbf{C}$, $\omega$ is the angular velocity (scalar in 2D), and $\sigma$ is the isotropic scale rate ($\sigma > 0$ when the board approaches the camera).

In component form:

$$v_x = v_{Cx} - \omega \cdot r_y + \sigma \cdot r_x$$
$$v_y = v_{Cy} + \omega \cdot r_x + \sigma \cdot r_y$$

This is a **2D similarity velocity field** with 4 unknowns: $(v_{Cx}, v_{Cy}, \omega, \sigma)$.

### 6.3 Acceleration Model

For markers with 3+ frame history, acceleration is estimated via central difference:

$$\mathbf{a}(t\!-\!1) = \mathbf{p}(t) - 2\,\mathbf{p}(t\!-\!1) + \mathbf{p}(t\!-\!2)$$

The acceleration field follows the same similarity structure with centripetal correction:

$$\mathbf{a}(\mathbf{p}) = \mathbf{a}_C + \varepsilon \times \mathbf{r} + \dot{\sigma} \cdot \mathbf{r} - (\omega^2 - \sigma^2) \cdot \mathbf{r}$$

where $\varepsilon$ is angular acceleration and $\dot{\sigma}$ is scale acceleration. The $(\omega^2 - \sigma^2)$ term combines centripetal and scale-squared effects. Before fitting, the centripetal+scale term is removed from measured accelerations:

$$\mathbf{a}_{\text{corr}} = \mathbf{a}_{\text{measured}} + (\omega^2 - \sigma^2) \cdot \mathbf{r}$$

### 6.4 RANSAC Fitting

**Why RANSAC?** Velocity measurements come from identified marker tracks. If Hungarian tracking made a wrong assignment on a previous frame, that track's velocity is corrupted. RANSAC rejects these outlier velocities.

**Parameters**:

| Parameter | Velocity | Acceleration |
|-----------|----------|-------------|
| Sample size | 8 markers | 8 markers |
| Equations per sample | 16 (2 per marker) | 16 |
| Unknowns | 4 | 4 |
| Overdetermination | 4x | 4x |
| Iterations | 200 | 100 |
| Inlier threshold | 3.0 px | 5.0 px |

**Why 8-marker samples?** With 4 unknowns, 2 markers (4 equations) suffice for an exact solution, but the resulting hypothesis is noise-sensitive. 8 markers give 4x overdetermination per hypothesis, producing much more stable estimates. The cost is needing more iterations: $P(\text{success}) = 1 - (1 - p^8)^{200}$ where $p$ is the inlier ratio. At 70% inliers: $P > 0.9999$.

**Outlier smoothing**: After RANSAC, outlier marker velocities are replaced with model-predicted values:

$$\mathbf{v}_{\text{outlier}} \leftarrow (v_{Cx} - \omega \cdot r_y + \sigma \cdot r_x,\; v_{Cy} + \omega \cdot r_x + \sigma \cdot r_y)$$

This produces a clean, physically consistent field for all subsequent consumers.

### 6.5 Position Prediction

The predicted displacement over time $\Delta t$ is:

$$\Delta \mathbf{p} = \mathbf{v} \cdot \Delta t + \tfrac{1}{2}\,\mathbf{a} \cdot \Delta t^2$$

### 6.6 Instantaneous Centers

The velocity field has special points where $\mathbf{v} = 0$:

**Combined center** (spiral center): Solving $\mathbf{v}(\mathbf{p}) = 0$ yields:

$$\begin{bmatrix} \sigma & -\omega \\ \omega & \sigma \end{bmatrix} \begin{bmatrix} r_x \\ r_y \end{bmatrix} = \begin{bmatrix} -v_{Cx} \\ -v_{Cy} \end{bmatrix}$$

with determinant $\sigma^2 + \omega^2$. For pure rotation ($\sigma \approx 0$), this reduces to the classical instantaneous center of rotation. For pure scale ($\omega \approx 0$), this is the center of expansion/contraction.

## 7. Local Homography Re-identification

### 7.1 Motivation

After Hungarian tracking identifies a subset of markers, remaining unidentified blobs can be matched via geometric projection. A homography maps board coordinates to image positions; projecting unmatched board positions identifies the nearest unmatched blob.

### 7.2 Two-Level Approach

1. **Global homography**: RANSAC fit from all identified markers ($\geq 4$ required). Projects every missing board position; accepts if projection-to-blob distance < $0.3 \times \text{mean\_spacing}$.

2. **Local homography**: For blobs that the global fit misses (e.g., due to strong lens distortion at edges), fit a homography from the 4 nearest identified neighbors. The local fit adapts to local distortion.

**Why 30% of spacing?** The threshold must be tight enough to avoid cross-matching with adjacent positions, yet loose enough to accommodate projection error from a planar homography on a curved lens. Empirically, 30% of the mean neighbor spacing balances these requirements.

## 8. Post-Homography Velocity Filter

### 8.1 Problem

The local homography step (Section 7) can re-introduce swapped markers that the bidirectional velocity check (Section 5.5) correctly rejected. This happens because the homography fits the majority of markers correctly and projects the swapped position close to a blob &mdash; the spatial fit is good even though the temporal prediction is wrong.

### 8.2 Solution

After homography adds new markers, each newly-added marker is re-checked against the velocity field:

$$\text{error} = \| (p_{\text{prev}} + \mathbf{V}(p_{\text{prev}})) - p_{\text{curr}} \|$$

If $\text{error} > \tau$ (same displacement-proportional tolerance as Section 5.5), the marker is rejected.

**Guard**: Only on near-square grids. Non-square grids don't suffer from swap ambiguity.

## 9. FCG-Reference Row-Pair Swap Correction

### 9.1 Approach

Using the board coordinate system as a reference, compute a RANSAC homography $H$ from board coordinates to current image positions. For each adjacent-row pair $(r, c)$ and $(r+1, c)$:

$$\text{cost}_{\text{current}} = \|H \cdot b_a - p_a\| + \|H \cdot b_b - p_b\|$$
$$\text{cost}_{\text{swapped}} = \|H \cdot b_a - p_b\| + \|H \cdot b_b - p_a\|$$

**Swap if**: $\text{cost}_{\text{swapped}} < 0.5 \cdot \text{cost}_{\text{current}}$. The conservative 50% threshold avoids false swaps from noise.

## 10. Disappeared-Neighbor Swap Detection

### 10.1 Problem

When a marker goes off-screen, its hex-adjacent neighbor may be assigned the disappeared marker's ID by Hungarian tracking. The homography correction (Section 9) doesn't catch this because it only compares present marker pairs.

### 10.2 Hex-Grid Connectivity

For an asymmetric circle grid, each marker has 6 hex neighbors:

```
Even row (r%2 == 0):                 Odd row (r%2 == 1):
  (r-1, c-1)  (r-1, c)                (r-1, c)  (r-1, c+1)
       \       /                            \       /
  (r, c-1) -- (r, c) -- (r, c+1)    (r, c-1) -- (r, c) -- (r, c+1)
       /       \                            /       \
  (r+1, c-1)  (r+1, c)                (r+1, c)  (r+1, c+1)
```

**Why hex and not 4-connected?** On the asymmetric grid, the physical nearest neighbors are the hex-diagonal ones (distance $s\sqrt{2}$), closer than same-row neighbors (distance $2s$). Swaps follow physical proximity, not grid-axis alignment.

### 10.3 Algorithm

For each identified marker with $\text{gid} = X$:
1. Enumerate all hex neighbors of $X$
2. For each neighbor $Y$: if $Y$ is NOT currently assigned AND was seen within the last 5 frames:
   - Predict $Y$'s expected position using the velocity field: $\hat{p}_Y = p_Y^{\text{prev}} + \mathbf{V}(p_Y^{\text{prev}}) \cdot \Delta t$
   - If $\|p_X^{\text{curr}} - \hat{p}_Y\| < \tau_{\text{swap}}$: marker $X$ is at $Y$'s predicted position &rarr; reject $X$

**Threshold**: $\tau_{\text{swap}} = \max(20\;\text{px},\; 0.5 \cdot \|\mathbf{v}_C\|)$

The velocity-proportional term ensures the threshold scales with board speed. At high speeds, prediction error is larger, and a tighter threshold would cause false rejections.

### 10.4 Cascading Passes

A single pass may miss chain swaps (M0 disappears &rarr; M1 takes M0's spot &rarr; M6 takes M1's spot). After rejecting M1, M6 now sees M1 as "missing" and can be caught in the next pass.

The `assigned_gids` set is updated live: when a marker is rejected, it is immediately removed from the set, making it visible as "missing" to subsequent checks within the same pass and across passes. Up to 5 passes are executed; early termination occurs when a pass produces zero rejections.

## 11. Tracking State

### 11.1 Per-Marker Track History

Each identified marker maintains a 3-frame ring buffer:

```
MarkerTrack {
    position_history[3]    // (x, y) positions
    frame_ids[3]           // frame numbers
    history_count          // 0-3 entries
    last_position          // most recent position
    last_seen_frame        // for recency checks
    age                    // frames since first seen
}
```

Tracks not seen for >10 frames are pruned.

### 11.2 Forward and Backward Fields

Two velocity fields are maintained:

- **Forward field**: Anchored at previous-frame positions. Used to predict current-frame positions from previous.
- **Backward field**: Anchored at current-frame positions. Used to predict previous-frame positions from current.

Both fields share the same similarity model parameters $(v_C, \omega, \sigma)$ but are fitted independently (slightly different centroids due to board motion).

## 12. Debug Output

### 12.1 Per-Frame CSV (`<debug_dir>/filter-csv/frame_NNNNNN.csv`)

One row per detected marker, columns:

| Column | Description |
|--------|-------------|
| `marker_idx` | Detection index |
| `pixel_x`, `pixel_y` | Image position |
| `final_gid` | Assigned global ID (-1 if unidentified) |
| `method` | Identification method used |
| `vel_field_valid` | Whether velocity field was available |
| `vel_vCx`, `vel_vCy`, `vel_omega`, `vel_sigma` | Similarity model parameters |
| `vel_rms` | Model fit RMS residual |
| `vel_inliers`, `vel_total` | RANSAC inlier count |
| `vel_fwd_err`, `vel_bwd_err` | Bidirectional prediction errors |
| `vel_tolerance` | Acceptance threshold used |
| `vel_rejected` | 1 if velocity check rejected this marker |
| `disappeared_nbr_gid` | Neighbor gid that triggered swap detection |
| `disappeared_dist` | Distance to expected neighbor position |
| `disappeared_rejected` | 1 if disappeared-neighbor check rejected |
| `homography_added` | 1 if added by local homography (not Hungarian) |
| `track_prev_x`, `track_prev_y` | Previous-frame track position |
| `predicted_x`, `predicted_y` | Velocity-predicted position |

### 12.2 Detection PNGs

- `<debug_dir>/markers-png-NNN/frame_NNNNNN.png` &mdash; Pre-optimization visualization
- `<debug_dir>/markers-png-final-NNN/frame_NNNNNN.png` &mdash; Final ordering sent to calibrator

**Visual layers**:
- Magenta lines: prediction vectors (current position &rarr; predicted position)
- Cyan arrows: per-marker velocity (RANSAC inlier), red arrows: outlier
- Yellow circle + "R": instantaneous center of rotation
- Green circle + "S": instantaneous center of scale
- Text overlay: `v=(vCx,vCy) w=Xdeg/f s=X/f inl=N/M rms=Xpx`
- Green circles: identified markers with ID labels
- Yellow dot-circles: unidentified detected blobs

## Appendix A: Key Thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Near-square guard | aspect < 1.3 | Distinguishes ambiguous (10x5, aspect=1.0) from unambiguous (7x5, aspect=1.5) grids |
| Hungarian max distance | 80 px | Upper bound on inter-frame displacement at typical frame rates |
| RANSAC homography inlier | 5 px | Tight enough to reject swaps, loose enough for subpixel detection noise |
| Velocity RANSAC inlier | 3 px | Below inter-marker spacing (~30 px), above lens distortion effect (~1-2 px) |
| Velocity RANSAC sample size | 8 markers | 4x overdetermination for robust hypothesis; 200 iterations for 99.99% success at 70% inlier rate |
| Velocity acceptance tolerance | max(50% disp, 15 px) capped at 5% edge | Proportional to motion speed; floor prevents over-rejection at rest; cap prevents over-tolerance |
| Homography matching | 30% of mean spacing | Tight enough for unique matching, loose enough for projection error |
| Row-pair swap | swapped cost < 90% original | Conservative to avoid false swaps |
| 180-degree flip (non-square) | flip cost < 80% identity | Requires clear margin |
| Disappeared-neighbor threshold | max(20 px, 50% speed) | Velocity-proportional with absolute floor |
| Track recency | 5 frames | Balance between catching delayed disappearances and avoiding stale data |
| Track pruning | 10 frames unseen | Prevents unbounded memory growth |
| Cascading passes | max 5 | Covers chain swaps of practical length |

## Appendix B: Design Decisions

### B.1 Global vs Local Velocity Model

**Decision**: Single global similarity model for all markers.

**Alternatives considered**:
- Per-marker velocity: No spatial coherence; each marker's estimate is noisy.
- Local k-NN (k=5): Different query points get different $(v_C, \omega)$, violating rigid-body physics. Edge markers have biased neighbor sets.
- Global rigid body (no scale): Misses approach/recede motion, producing systematic residuals.

**Rationale**: All markers are on one rigid board. A single 4-parameter model $(v_{Cx}, v_{Cy}, \omega, \sigma)$ enforces physical consistency while capturing the dominant motions (translation, rotation, perspective-induced scale).

### B.2 Post-Homography Filter Instead of Homography Modification

**Decision**: Re-check homography-added markers against velocity field, rather than modifying the homography algorithm itself.

**Rationale**: The homography re-identification is a general geometric method that works well for filling gaps. The problem is specific to near-square grids where swapped positions are geometrically plausible. A post-filter keeps the homography logic clean and applies the velocity constraint only where needed.

### B.3 Hex Connectivity for Disappeared-Neighbor Check

**Decision**: Use full 6-neighbor hex connectivity for asymmetric grids.

**Alternative**: 4-connected (row and column adjacency only).

**Rationale**: On the asymmetric hex grid, the physical nearest neighbors are the diagonal ones (distance $s\sqrt{2} \approx 1.41s$) which are closer than same-row neighbors (distance $2s$). Swaps follow physical proximity. Using only 4-connectivity misses diagonal swaps like gid 7 (row 1, col 2) &rarr; gid 3 (row 0, col 3).

### B.4 Immutable 180-Degree Reference

**Decision**: Lock orientation from the first `findCirclesGrid` frame and never update.

**Alternative**: Adaptive reference updated each frame.

**Rationale**: A single incorrect update permanently flips all subsequent frames. The first `findCirclesGrid` frame with full visibility provides a reliable anchor. The cost of a slightly stale reference (board has moved) is negligible &mdash; the total pairwise distance metric is robust to moderate translations and rotations.
