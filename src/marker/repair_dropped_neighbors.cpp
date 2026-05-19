#include "repair_dropped_neighbors.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <unordered_set>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace marker::repair
{

uint64_t median_dt_ns(const std::map<int, FrameCacheEntry>& frame_cache)
{
    if (frame_cache.size() < 2)
    {
        return 0;
    }
    std::vector<uint64_t> dts;
    dts.reserve(frame_cache.size() - 1);
    auto prev_it = frame_cache.begin();
    for (auto it = std::next(prev_it); it != frame_cache.end(); ++it, ++prev_it)
    {
        const uint64_t t_curr = it->second.ts_ns;
        const uint64_t t_prev = prev_it->second.ts_ns;
        if (t_curr == 0 || t_prev == 0 || t_curr <= t_prev)
        {
            continue;
        }
        dts.push_back(t_curr - t_prev);
    }
    if (dts.empty())
    {
        return 0;
    }
    std::nth_element(dts.begin(), dts.begin() + dts.size() / 2, dts.end());
    return dts[dts.size() / 2];
}

namespace
{
// Detect whether a timestamp gap lives between two adjacent cache entries.
bool is_gap(const uint64_t t_prev, const uint64_t t_curr, const uint64_t threshold_ns)
{
    if (t_prev == 0 || t_curr == 0 || t_curr <= t_prev)
    {
        return false;
    }
    return (t_curr - t_prev) > threshold_ns;
}

uint64_t gap_threshold_ns(const uint64_t median_dt, const float gap_factor)
{
    return static_cast<uint64_t>(static_cast<double>(median_dt) * static_cast<double>(gap_factor));
}

// Earliest re-sync point strictly after start_idx, classified as either an
// FCG success (a trusted back-prop anchor) or a post-gap frame (a jump that
// breaks the Hungarian chain). The scan is unbounded — the affine-repair
// anti-alias guard inside repair_span() is the only thing that decides how
// far back-propagation actually extends.
struct NextBoundary
{
    enum Kind
    {
        kFcg,    // next_boundary is an FCG success — usable as back-prop anchor
        kGap,    // next_boundary is the first frame after a timestamp jump
        kNone,   // end-of-sequence reached with no further FCG or jump
    };
    Kind kind = kNone;
    int  idx  = -1;   // -1 iff kind == kNone
};

NextBoundary find_next_boundary(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const int start_idx,
    const uint64_t threshold_ns)
{
    const auto start_it = frame_cache.find(start_idx);
    if (start_it == frame_cache.end())
    {
        return {NextBoundary::kNone, -1};
    }
    auto prev_it = start_it;
    for (auto scan = std::next(start_it); scan != frame_cache.end(); ++scan, ++prev_it)
    {
        if (scan->second.fcg_succeeded)
        {
            return {NextBoundary::kFcg, scan->first};
        }
        if (is_gap(prev_it->second.ts_ns, scan->second.ts_ns, threshold_ns))
        {
            return {NextBoundary::kGap, scan->first};
        }
    }
    return {NextBoundary::kNone, -1};
}
}  // namespace

std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const float gap_factor)
{
    std::vector<Span> spans;
    const uint64_t median_dt = median_dt_ns(frame_cache);
    if (median_dt == 0)
    {
        return spans;
    }
    const uint64_t threshold_ns = gap_threshold_ns(median_dt, gap_factor);

    auto prev_it = frame_cache.begin();
    for (auto it = std::next(prev_it); it != frame_cache.end(); ++it, ++prev_it)
    {
        if (!is_gap(prev_it->second.ts_ns, it->second.ts_ns, threshold_ns))
        {
            continue;
        }
        const NextBoundary next = find_next_boundary(frame_cache, it->first, threshold_ns);
        if (next.kind != NextBoundary::kFcg)
        {
            continue;  // no FCG reachable before the next jump — handled as unrecoverable
        }
        Span span;
        span.start_idx    = it->first;
        span.anchor_idx   = next.idx;
        span.dt_at_gap_ns = it->second.ts_ns - prev_it->second.ts_ns;
        spans.push_back(span);
    }
    return spans;
}

std::vector<UnrecoverableSpan> detect_unrecoverable_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const float gap_factor)
{
    std::vector<UnrecoverableSpan> spans;
    const uint64_t median_dt = median_dt_ns(frame_cache);
    if (median_dt == 0)
    {
        return spans;
    }
    const uint64_t threshold_ns = gap_threshold_ns(median_dt, gap_factor);

    const int last_cached_frame = frame_cache.empty() ? -1 : frame_cache.rbegin()->first;

    auto prev_it = frame_cache.begin();
    for (auto it = std::next(prev_it); it != frame_cache.end(); ++it, ++prev_it)
    {
        if (!is_gap(prev_it->second.ts_ns, it->second.ts_ns, threshold_ns))
        {
            continue;
        }
        const NextBoundary next = find_next_boundary(frame_cache, it->first, threshold_ns);
        if (next.kind == NextBoundary::kFcg)
        {
            continue;  // reachable FCG → handled by detect_drop_affected_spans + repair_span
        }
        UnrecoverableSpan span;
        span.start_idx    = it->first;
        span.end_idx      = (next.kind == NextBoundary::kGap) ? (next.idx - 1)
                                                              : last_cached_frame;
        span.dt_at_gap_ns = it->second.ts_ns - prev_it->second.ts_ns;
        spdlog::warn("repair: gap at frame {} (dt={}ms) unrecoverable ({}) — "
                     "invalidating frames [{}..{}]",
                     span.start_idx, span.dt_at_gap_ns / 1'000'000ULL,
                     next.kind == NextBoundary::kGap ? "bounded by next jump"
                                                      : "no further FCG or jump",
                     span.start_idx, span.end_idx);
        spans.push_back(span);
    }
    return spans;
}

namespace
{
// ---------------------------------------------------------------------------
// Affine + matching primitives
// ---------------------------------------------------------------------------

/// Per-span cache of the anchor FCG frame used by the repair loop.
///
/// The anchor is the first trusted FCG success after the dedupe gap; its
/// identified marker positions drive affine propagation for every frame in
/// the recoverable span. We also precompute `median_nn_distance_px` once so
/// threshold computation doesn't repeat the O(N^2) scan per frame.
struct AnchorReference
{
    std::vector<cv::Point2f> positions;               ///< one entry per valid gid
    std::vector<int>         gids;                    ///< parallel to positions
    float                    median_nn_distance_px = 0.f;  ///< image-space inter-marker scale
};

// Median nearest-neighbour distance across the anchor's identified markers.
// For the asymmetric-offset HexRing this equals the s*sqrt(2) diagonal-row
// spacing; row-shift aliasing sits at 2s = sqrt(2)*d_nn in image space.
// Robust to the number of markers and adapts to zoom / board distance without
// needing any external calibration.
float compute_median_nn_distance(const std::vector<cv::Point2f>& positions)
{
    if (positions.size() < 2)
    {
        return 0.f;
    }
    std::vector<float> nearest_distances;
    nearest_distances.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i)
    {
        float best_sq = std::numeric_limits<float>::max();
        for (size_t j = 0; j < positions.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            const float delta_x = positions[i].x - positions[j].x;
            const float delta_y = positions[i].y - positions[j].y;
            const float distance_sq = delta_x * delta_x + delta_y * delta_y;
            if (distance_sq < best_sq)
            {
                best_sq = distance_sq;
            }
        }
        if (best_sq < std::numeric_limits<float>::max())
        {
            nearest_distances.push_back(std::sqrt(best_sq));
        }
    }
    if (nearest_distances.empty())
    {
        return 0.f;
    }
    std::nth_element(nearest_distances.begin(),
                     nearest_distances.begin() + nearest_distances.size() / 2,
                     nearest_distances.end());
    return nearest_distances[nearest_distances.size() / 2];
}

AnchorReference extract_anchor_reference(const FrameCacheEntry& anchor_entry)
{
    AnchorReference reference;
    reference.positions.reserve(anchor_entry.marker_positions.size());
    reference.gids.reserve(anchor_entry.marker_positions.size());
    for (size_t gid = 0; gid < anchor_entry.marker_positions.size(); ++gid)
    {
        const cv::Point2f position = anchor_entry.marker_positions[gid];
        if (position.x < 0.f || position.y < 0.f)
        {
            continue;
        }
        reference.positions.push_back(position);
        reference.gids.push_back(static_cast<int>(gid));
    }
    reference.median_nn_distance_px = compute_median_nn_distance(reference.positions);
    return reference;
}

/// Image-space thresholds for one repair span, all expressed in pixels.
///
/// Populated by thresholds_from_scale() from the anchor's inter-marker scale.
/// Using a single struct keeps the per-frame repair free of hardcoded pixel
/// constants and makes it possible to adapt to any zoom / board distance
/// without retuning.
struct GeometricThresholds
{
    std::vector<float> icp_px;               ///< coarse-to-fine ICP gates
    float              final_gate_px        = 0.f;  ///< final gid->blob snap cap
    float              max_chained_delta_px = 0.f;  ///< alias guard on chained affine
};

/// Build the geometric threshold bag from the anchor's image-space scale.
///
/// `d_nn_px` is the median nearest-neighbour distance between anchor-identified
/// markers. For the asymmetric-offset HexRing this equals `s * sqrt(2)` where
/// `s` is the grid spacing in image pixels, so it adapts to zoom and board
/// distance automatically.
///
/// The coefficients are chosen so that:
///   - the row-shift alias (`sqrt(2) * d_nn ≈ 1.41 * d_nn`) always exceeds
///     `max_chained_delta_px` (1.20 * d_nn) → aliased fits are rejected
///   - legitimate handheld motion (~0.10-0.15 * d_nn per frame) fits under
///     `max_chained_delta_px` for ~8 consecutive missed-frame accumulations
///   - `final_gate_px` (0.20 * d_nn) is below the smallest valid inter-gid
///     image distance, so the final greedy assignment cannot snap a marker
///     onto a neighbour's blob
GeometricThresholds thresholds_from_scale(const float d_nn_px)
{
    GeometricThresholds thresholds;
    thresholds.icp_px                = {1.20f * d_nn_px, 0.60f * d_nn_px,
                                        0.30f * d_nn_px, 0.20f * d_nn_px};
    thresholds.final_gate_px         = 0.20f * d_nn_px;
    thresholds.max_chained_delta_px  = 1.20f * d_nn_px;
    return thresholds;
}

std::vector<cv::Point2f> blob_positions_of(const std::vector<base::MarkerCoding>& coding)
{
    std::vector<cv::Point2f> positions;
    positions.reserve(coding.size());
    for (const base::MarkerCoding& coding_marker : coding)
    {
        positions.emplace_back(coding_marker.col_, coding_marker.row_);
    }
    return positions;
}

cv::Point2f apply_affine(const cv::Mat& affine, const cv::Point2f& point)
{
    const double m00 = affine.at<double>(0, 0);
    const double m01 = affine.at<double>(0, 1);
    const double m02 = affine.at<double>(0, 2);
    const double m10 = affine.at<double>(1, 0);
    const double m11 = affine.at<double>(1, 1);
    const double m12 = affine.at<double>(1, 2);
    const double output_x = m00 * static_cast<double>(point.x) + m01 * static_cast<double>(point.y) + m02;
    const double output_y = m10 * static_cast<double>(point.x) + m11 * static_cast<double>(point.y) + m12;
    return cv::Point2f(static_cast<float>(output_x), static_cast<float>(output_y));
}

/// One (anchor-gid, current-frame-blob) candidate pair for greedy matching.
struct Pair
{
    float dist_px;      ///< Euclidean distance between predicted and blob positions
    int   anchor_index; ///< index into AnchorReference::gids / positions
    int   blob_index;   ///< index into the current frame's blob_positions
};

/// Enumerate (anchor-gid, blob) pairs within `threshold_px` of each other
/// after applying the candidate affine, sorted ascending by distance.
///
/// The sorted list is consumed by resolve_pairs_greedy() so that the shortest
/// matches take precedence when both endpoints are unused.
std::vector<Pair> build_candidate_pairs(
    const AnchorReference&          anchor,
    const std::vector<cv::Point2f>& blob_positions,
    const cv::Mat&                  affine,
    const float                     threshold_px)
{
    std::vector<Pair> pairs;
    pairs.reserve(anchor.gids.size() * 4);
    for (size_t anchor_index = 0; anchor_index < anchor.gids.size(); ++anchor_index)
    {
        const cv::Point2f predicted = apply_affine(affine, anchor.positions[anchor_index]);
        for (size_t blob_index = 0; blob_index < blob_positions.size(); ++blob_index)
        {
            const float delta_x = predicted.x - blob_positions[blob_index].x;
            const float delta_y = predicted.y - blob_positions[blob_index].y;
            const float distance_px = std::sqrt(delta_x * delta_x + delta_y * delta_y);
            if (distance_px <= threshold_px)
            {
                pairs.push_back(Pair{distance_px, static_cast<int>(anchor_index), static_cast<int>(blob_index)});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& lhs, const Pair& rhs) { return lhs.dist_px < rhs.dist_px; });
    return pairs;
}

/// Result of a single round of greedy (anchor-gid, blob) matching.
///
/// `blob_to_gid` is parallel to the current frame's blob list — it is what
/// build_repaired_decoding() turns into the repaired `ImageDecoding`.
/// `src_anchor` / `dst_blob` are the accepted coordinate pairs, formatted
/// the way cv::estimateAffinePartial2D() expects them.
struct Assignment
{
    std::vector<int>         blob_to_gid;    ///< gid assigned to each blob, -1 if none
    std::vector<cv::Point2f> src_anchor;     ///< anchor-frame positions of matched pairs
    std::vector<cv::Point2f> dst_blob;       ///< current-frame positions of matched pairs
    int                      matched_count = 0;
};

/// Translation-unit-local disposition of one back-propagated frame.
///
/// Exposed only through `SpanRepairResult` at the public API boundary.
///   kCured              — affine cure passed all checks; decoding replaced.
///   kAntialiasRejected  — early-out (too few blobs, chained-delta jump, or
///                         final matched_count < 4). The caller's gap sweep
///                         will invalidate the frame under kGap.
///   kPhantomRejected    — cure produced IDs but the virtual-row homography
///                         check flagged them as row/column aliased; the
///                         caller must invalidate the frame under kPhantom.
enum class RepairOutcome
{
    kCured,
    kAntialiasRejected,
    kPhantomRejected,
};

/// Greedy one-to-one assignment: shortest pair wins, conflicts drop to the
/// next pair in the sorted candidate list. O(N) in the number of candidates.
Assignment resolve_pairs_greedy(
    const std::vector<Pair>&        pairs,
    const AnchorReference&          anchor,
    const std::vector<cv::Point2f>& blob_positions)
{
    Assignment result;
    result.blob_to_gid.assign(blob_positions.size(), -1);
    std::vector<char> anchor_taken(anchor.gids.size(), 0);
    std::vector<char> blob_taken(blob_positions.size(), 0);
    for (const Pair& pair : pairs)
    {
        if (anchor_taken[pair.anchor_index] || blob_taken[pair.blob_index])
        {
            continue;
        }
        anchor_taken[pair.anchor_index] = 1;
        blob_taken[pair.blob_index] = 1;
        result.blob_to_gid[pair.blob_index] = anchor.gids[pair.anchor_index];
        result.src_anchor.push_back(anchor.positions[pair.anchor_index]);
        result.dst_blob.push_back(blob_positions[pair.blob_index]);
        ++result.matched_count;
    }
    return result;
}

/// One round of gid->blob matching: enumerate candidates under the given
/// threshold and resolve them greedily. Convenience wrapper used by both the
/// ICP inner loop and the final gating pass.
Assignment assign_greedy(
    const AnchorReference&          anchor,
    const std::vector<cv::Point2f>& blob_positions,
    const cv::Mat&                  affine,
    const float                     threshold_px)
{
    if (anchor.gids.empty() || blob_positions.empty())
    {
        Assignment empty;
        empty.blob_to_gid.assign(blob_positions.size(), -1);
        return empty;
    }
    const std::vector<Pair> pairs = build_candidate_pairs(anchor, blob_positions, affine, threshold_px);
    return resolve_pairs_greedy(pairs, anchor, blob_positions);
}

// ---------------------------------------------------------------------------
// Virtual-row alias detection (post-repair structural check)
// ---------------------------------------------------------------------------
//
// After the affine chain has produced a candidate (blob -> gid) mapping, we
// fit a homography from the board's intrinsic (row, col) lattice into the
// current image and project a wide ring of *phantom* grid cells — cells that
// sit outside the real 0..rows-1 / 0..cols-1 rectangle. If several phantom
// cells land within one `final_gate_px` of *unidentified* blobs in the same
// frame, those blobs form a coherent off-board lattice, which is the
// signature of a row- or column-shift alias that slipped past the chained-
// delta guard. See the diary
// `docs/diary/2026-04-22-1030-virtual-row-alias-invalidation-wip.md` for the
// failure mode this was introduced to catch (eposN_1 F556..F625).

// The board uses the asymmetric-offset HexRing layout:
//   x = 2*col + (row mod 2),  y = row
// Canonical source: `GridCalibrationTargetCirclegridHexRing::createGridPoints()`
// in `aslam_cv/aslam_cameras_circlegrid_hexring/src/GridCalibrationTargetCirclegridHexRing.cpp`,
// and the workspace CLAUDE.md "HexRing target grid geometry" section.
// We use the unitless form here — the homography absorbs the scale — and
// compute parity with `((row % 2) + 2) % 2` so negative phantom rows
// produce the same alternation as the real grid.
int board_parity_for_row(const int row)
{
    return ((row % 2) + 2) % 2;
}

cv::Point2f board_lattice_point(const int row, const int col)
{
    const int parity = board_parity_for_row(row);
    return cv::Point2f(static_cast<float>(2 * col + parity), static_cast<float>(row));
}

// Collect the (board_pts, image_pts) pairs usable to fit the homography.
struct MatchedLatticePairs
{
    std::vector<cv::Point2f> board_pts;
    std::vector<cv::Point2f> image_pts;
};

MatchedLatticePairs collect_matched_lattice_pairs(
    const Assignment&               final_assignment,
    const std::vector<cv::Point2f>& blob_positions,
    const BoardCircleGrid&          board)
{
    MatchedLatticePairs pairs;
    pairs.board_pts.reserve(final_assignment.matched_count);
    pairs.image_pts.reserve(final_assignment.matched_count);
    for (size_t blob_index = 0; blob_index < blob_positions.size(); ++blob_index)
    {
        const int gid = final_assignment.blob_to_gid[blob_index];
        if (gid < 0)
        {
            continue;
        }
        const Eigen::Vector2i row_col = board.id_to_row_and_col(gid);
        pairs.board_pts.push_back(board_lattice_point(row_col(0), row_col(1)));
        pairs.image_pts.push_back(blob_positions[blob_index]);
    }
    return pairs;
}

// Enumerate the phantom cells that surround the real grid. For a rows x cols
// board the full candidate box is
//     r in [-(rows-1), 2*rows-1) = [-(rows-1), 2*rows-2]
//     c in [-(cols-1), 2*cols-1) = [-(cols-1), 2*cols-2]
// and we subtract the real 0..rows-1 / 0..cols-1 rectangle. That yields
// rows=7, cols=5 -> 20*14 - 35 = 245 phantom cells, which matches the plan.
std::vector<cv::Point2f> enumerate_phantom_board_points(
    const int rows,
    const int cols)
{
    std::vector<cv::Point2f> phantoms;
    const int row_lo = -(rows - 1);
    const int row_hi =  2 * rows - 1;   // exclusive
    const int col_lo = -(cols - 1);
    const int col_hi =  2 * cols - 1;   // exclusive
    phantoms.reserve(static_cast<size_t>((row_hi - row_lo) * (col_hi - col_lo)));
    for (int row = row_lo; row < row_hi; ++row)
    {
        for (int col = col_lo; col < col_hi; ++col)
        {
            const bool inside_real_grid =
                (row >= 0) && (row < rows) && (col >= 0) && (col < cols);
            if (inside_real_grid)
            {
                continue;
            }
            phantoms.push_back(board_lattice_point(row, col));
        }
    }
    return phantoms;
}

// Count phantom cells whose projected image position has at least one
// *unidentified* blob within `gate_px`. Multiple phantom cells may share the
// same nearest blob — each phantom cell contributes at most one to the sum,
// which matches the plan's "sum hits across ALL phantom cells" wording. A
// single unidentified blob sitting inside a lattice-coherent cluster of
// phantom cells (the bijective row-shift case) therefore produces one hit
// per phantom cell, which is exactly the structural evidence we want to
// flag.
int count_phantom_hits(
    const std::vector<cv::Point2f>& projected_phantoms,
    const std::vector<cv::Point2f>& blob_positions,
    const std::vector<int>&         blob_to_gid,
    const float                     gate_px)
{
    const float gate_sq = gate_px * gate_px;
    int hit_count = 0;
    for (const cv::Point2f& projected : projected_phantoms)
    {
        float best_sq = std::numeric_limits<float>::max();
        for (size_t blob_index = 0; blob_index < blob_positions.size(); ++blob_index)
        {
            if (blob_to_gid[blob_index] >= 0)
            {
                continue;
            }
            const float delta_x = projected.x - blob_positions[blob_index].x;
            const float delta_y = projected.y - blob_positions[blob_index].y;
            const float distance_sq = delta_x * delta_x + delta_y * delta_y;
            if (distance_sq < best_sq)
            {
                best_sq = distance_sq;
            }
        }
        if (best_sq <= gate_sq)
        {
            ++hit_count;
        }
    }
    return hit_count;
}

// Return true iff the post-repair identification is row/column aliased.
//
// The check fits a homography board->image on the accepted matches, projects
// every phantom lattice cell through it, and asks how many of those land on
// currently unidentified blobs. A legitimate frame has few unidentified
// blobs and they don't cluster on a shifted lattice, so random false
// positives are unlikely; a row-shifted frame puts the entire orphaned row
// onto phantom cells and therefore lights up many hits.
//
// Returns false on insufficient data (matched_count < 6) or degenerate
// homography — we never invalidate on inability to decide.
bool has_virtual_row_alias(
    const int                       frame_idx,
    const Assignment&               final_assignment,
    const std::vector<cv::Point2f>& blob_positions,
    const BoardCircleGrid&          board,
    const GeometricThresholds&      thresholds)
{
    if (final_assignment.matched_count < 6)
    {
        return false;
    }
    const MatchedLatticePairs lattice =
        collect_matched_lattice_pairs(final_assignment, blob_positions, board);
    if (lattice.board_pts.size() < 6)
    {
        return false;
    }
    const cv::Mat homography = cv::findHomography(
        lattice.board_pts, lattice.image_pts, cv::RANSAC,
        static_cast<double>(thresholds.final_gate_px));
    if (homography.empty())
    {
        return false;
    }
    const std::vector<cv::Point2f> phantom_board_pts =
        enumerate_phantom_board_points(board.rows_, board.cols_);
    if (phantom_board_pts.empty())
    {
        return false;
    }
    std::vector<cv::Point2f> projected_phantoms;
    cv::perspectiveTransform(phantom_board_pts, projected_phantoms, homography);
    const int hit_count = count_phantom_hits(
        projected_phantoms, blob_positions, final_assignment.blob_to_gid,
        thresholds.final_gate_px);
    const bool aliased = hit_count > 3;
    if (aliased)
    {
        spdlog::warn("repair_span: frame {} virtual-row aliased "
                     "(phantom_hits={} on {} cells, matched={})",
                     frame_idx, hit_count,
                     static_cast<int>(phantom_board_pts.size()),
                     final_assignment.matched_count);
    }
    return aliased;
}

// Attempt to cure a row-/column-aliased labeling by shifting every matched
// gid by (dx, dy) in *physical* lattice coordinates. Returns the corrected
// blob_to_gid vector if the post-shift phantom check comes back clean,
// nullopt otherwise.
//
// Board geometry reference — canonical:
//   - `GridCalibrationTargetCirclegridHexRing::createGridPoints()` in
//     `aslam_cv/aslam_cameras_circlegrid_hexring/src/GridCalibrationTargetCirclegridHexRing.cpp`
//   - Workspace CLAUDE.md section "HexRing target grid geometry".
// Asymmetric layout (the only mode used in production): each grid point at
// index (row, col) has physical coords (x, y) = (2*col + row%2, row) times
// spacing_meters.  Neighbor distances are d_nn = s*sqrt(2) diagonally and
// 2s horizontally.
//
// Why physical coords, not (dr, dc): a uniform (dr, dc) in board-index
// space does NOT correspond to a consistent image translation when row
// parity changes — even-row labels at dc=0 land correctly on odd-target
// rows, but odd-row labels at dc=0 miss by one column because the odd-row
// parity offset flips. Using physical (dx, dy) and recomputing
// (new_r, new_c) per-blob from the new lattice position handles parity
// correctly: the same image translation maps cleanly across even and odd
// source rows.
//
// Candidate range: dx in [-(2*cols-1), 2*cols-1], dy in [-(rows-1), rows-1]
// (the full physical bounding box that any shift can cover before all blobs
// fall off the board), sorted by Manhattan distance so the smallest shifts
// are tried first.
struct PhysicalShift
{
    int dx;  // in units of spacing (lattice x = 2*c + r%2)
    int dy;  // in units of spacing (lattice y = r)
};

// Translate a board-index (row, col) by a physical shift and return the new
// (row, col) on the lattice, or false if the shifted position falls off the
// board or between lattice sites (which happens when the x-parity of the
// shifted y doesn't match the destination x).
struct ShiftedRowCol
{
    int  row   = 0;
    int  col   = 0;
    bool valid = false;
};

ShiftedRowCol apply_physical_shift(
    const int            old_row,
    const int            old_col,
    const PhysicalShift& shift,
    const int            rows,
    const int            cols)
{
    const int new_row = old_row + shift.dy;
    if (new_row < 0 || new_row >= rows)
    {
        return ShiftedRowCol{0, 0, false};
    }
    const int old_x_phys = 2 * old_col + (old_row % 2);
    const int new_x_phys = old_x_phys + shift.dx;
    const int new_parity = new_row % 2;
    // x = 2*c + parity → c = (x - parity) / 2, and that must be a non-negative
    // integer for the new position to fall on a lattice site.
    const int c_numerator = new_x_phys - new_parity;
    if (c_numerator < 0 || (c_numerator % 2) != 0)
    {
        return ShiftedRowCol{0, 0, false};
    }
    const int new_col = c_numerator / 2;
    if (new_col < 0 || new_col >= cols)
    {
        return ShiftedRowCol{0, 0, false};
    }
    return ShiftedRowCol{new_row, new_col, true};
}

std::vector<PhysicalShift> build_physical_shift_candidates(const BoardCircleGrid& board)
{
    std::vector<PhysicalShift> candidates;
    const int dx_extent = 2 * board.cols_ - 1;
    const int dy_extent = board.rows_ - 1;
    candidates.reserve(static_cast<size_t>((2 * dx_extent + 1) * (2 * dy_extent + 1) - 1));
    for (int dy = -dy_extent; dy <= dy_extent; ++dy)
    {
        for (int dx = -dx_extent; dx <= dx_extent; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            candidates.push_back(PhysicalShift{dx, dy});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const PhysicalShift& lhs, const PhysicalShift& rhs)
              {
                  const int manhattan_lhs = std::abs(lhs.dx) + std::abs(lhs.dy);
                  const int manhattan_rhs = std::abs(rhs.dx) + std::abs(rhs.dy);
                  if (manhattan_lhs != manhattan_rhs)
                  {
                      return manhattan_lhs < manhattan_rhs;
                  }
                  // Within a shell, prefer row-dominated shifts (|dy| large).
                  const int abs_dy_lhs = std::abs(lhs.dy);
                  const int abs_dy_rhs = std::abs(rhs.dy);
                  if (abs_dy_lhs != abs_dy_rhs)
                  {
                      return abs_dy_lhs > abs_dy_rhs;
                  }
                  return std::tie(lhs.dx, lhs.dy) < std::tie(rhs.dx, rhs.dy);
              });
    return candidates;
}

struct CureResult
{
    std::vector<int> blob_to_gid;
    int              matched_count = 0;
    int              dx            = 0;
    int              dy            = 0;
};

std::optional<CureResult> attempt_shift_cure(
    const int                       frame_idx,
    const Assignment&               final_assignment,
    const std::vector<cv::Point2f>& blob_positions,
    const BoardCircleGrid&          board,
    const GeometricThresholds&      thresholds)
{
    const std::vector<PhysicalShift> shift_candidates = build_physical_shift_candidates(board);
    for (const PhysicalShift& shift : shift_candidates)
    {
        std::vector<int> candidate_blob_to_gid(blob_positions.size(), -1);
        int              match_count = 0;
        for (size_t blob_index = 0; blob_index < blob_positions.size(); ++blob_index)
        {
            const int old_gid = final_assignment.blob_to_gid[blob_index];
            if (old_gid < 0)
            {
                continue;
            }
            const Eigen::Vector2i row_col = board.id_to_row_and_col(old_gid);
            const ShiftedRowCol shifted = apply_physical_shift(
                row_col(0), row_col(1), shift, board.rows_, board.cols_);
            if (!shifted.valid)
            {
                continue;
            }
            candidate_blob_to_gid[blob_index] = shifted.row * board.cols_ + shifted.col;
            ++match_count;
        }
        if (match_count < 6)
        {
            continue;
        }
        Assignment candidate;
        candidate.blob_to_gid   = std::move(candidate_blob_to_gid);
        candidate.matched_count = match_count;
        const bool still_aliased = has_virtual_row_alias(
            frame_idx, candidate, blob_positions, board, thresholds);
        if (still_aliased)
        {
            continue;
        }
        CureResult cure;
        cure.blob_to_gid   = std::move(candidate.blob_to_gid);
        cure.matched_count = match_count;
        cure.dx            = shift.dx;
        cure.dy            = shift.dy;
        return cure;
    }
    return std::nullopt;
}

/// Coarse-to-fine ICP: on each pass, build candidate pairs within the current
/// threshold, re-fit an affine-partial-2D model (translation + rotation +
/// uniform scale) on the matched endpoints, and shrink the threshold.
///
/// Starts from `initial_affine` and returns the most refined estimate. If any
/// pass cannot find at least 3 pairs we skip its refit and carry the previous
/// estimate into the next (tighter) pass.
cv::Mat refine_affine(
    const AnchorReference&          anchor,
    const std::vector<cv::Point2f>& blob_positions,
    const cv::Mat&                  initial_affine,
    const std::vector<float>&       thresholds_px)
{
    cv::Mat affine = initial_affine.clone();
    for (const float threshold_px : thresholds_px)
    {
        const Assignment assignment = assign_greedy(anchor, blob_positions, affine, threshold_px);
        if (assignment.matched_count < 3)
        {
            continue;
        }
        std::vector<uchar> inliers;
        const cv::Mat refined = cv::estimateAffinePartial2D(
            assignment.src_anchor, assignment.dst_blob, inliers,
            cv::RANSAC, std::max(2.0, threshold_px * 0.5));
        if (!refined.empty())
        {
            affine = refined;
        }
    }
    return affine;
}

// ---------------------------------------------------------------------------
// Repaired ImageDecoding construction
// ---------------------------------------------------------------------------

base::ImageDecoding build_repaired_decoding(
    const std::vector<base::MarkerCoding>& coding,
    const std::vector<int>&                blob_to_gid,
    const BoardCircleGrid&                 board,
    const cv::Mat1b&                       image)
{
    std::vector<base::MarkerRing> rings;
    rings.reserve(coding.size());
    Eigen::Matrix<std::optional<int>, -1, -1> ordering =
        Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board.rows_, board.cols_, std::nullopt);

    for (size_t blob_index = 0; blob_index < coding.size(); ++blob_index)
    {
        base::MarkerRing ring(coding[blob_index]);
        ring.global_id_ = blob_to_gid[blob_index];
        rings.push_back(ring);
        if (ring.global_id_ >= 0)
        {
            const Eigen::Vector2i row_col = board.id_to_row_and_col(ring.global_id_);
            ordering(row_col(0), row_col(1)) = static_cast<int>(rings.size() - 1);
        }
    }

    const cv::Mat1b empty;
    return base::ImageDecoding(
        /*success=*/ true,
        /*linear_input=*/ image.empty() ? cv::Mat1b() : image,
        /*binary=*/ empty,
        /*inverted_binary=*/ empty,
        /*ordering=*/ ordering,
        /*markers=*/ rings,
        /*markers_location=*/ empty,
        /*calibrated_area=*/ empty,
        /*all_detected_markers=*/ coding);
}

// Construct a "failed" ImageDecoding that downstream rebuildStoredObservations
// skips, so the frame drops out of calibration without taking its bad pass-1
// identifications with it.
base::ImageDecoding build_failed_decoding(const int board_rows, const int board_cols)
{
    const cv::Mat1b empty;
    const Eigen::Matrix<std::optional<int>, -1, -1> empty_ordering =
        Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board_rows, board_cols, std::nullopt);
    const std::vector<base::MarkerRing>   no_rings;
    const std::vector<base::MarkerCoding> no_coding;
    return base::ImageDecoding(
        /*success=*/ false,
        /*linear_input=*/ empty,
        /*binary=*/ empty,
        /*inverted_binary=*/ empty,
        /*ordering=*/ empty_ordering,
        /*markers=*/ no_rings,
        /*markers_location=*/ empty,
        /*calibrated_area=*/ empty,
        /*all_detected_markers=*/ no_coding);
}

// ---------------------------------------------------------------------------
// Per-frame repair — fit affine, assign, and publish the decoding.
// Returns true on success, false if no repair was possible.
// `running_affine` is updated in place so long spans chain cleanly.
// All geometric thresholds scale with the anchor's d_nn (median
// nearest-neighbour image-space spacing); see thresholds_from_scale().
// ---------------------------------------------------------------------------

float translation_delta_px(const cv::Mat& affine_a, const cv::Mat& affine_b)
{
    const double delta_x = affine_a.at<double>(0, 2) - affine_b.at<double>(0, 2);
    const double delta_y = affine_a.at<double>(1, 2) - affine_b.at<double>(1, 2);
    return static_cast<float>(std::sqrt(delta_x * delta_x + delta_y * delta_y));
}

RepairOutcome repair_single_frame(
    const int                           frame_idx,
    const Span&                         span,
    const FrameCacheEntry&              frame_entry,
    const AnchorReference&              anchor,
    const GeometricThresholds&          thresholds,
    const BoardCircleGrid&              board,
    cv::Mat&                            running_affine,
    std::map<int, base::ImageDecoding>& decoded)
{
    const std::vector<cv::Point2f> blob_positions = blob_positions_of(frame_entry.coding_markers);
    if (blob_positions.size() < 4)
    {
        spdlog::warn("repair_span: frame {} has only {} coding blobs — skipping",
                     frame_idx, blob_positions.size());
        return RepairOutcome::kAntialiasRejected;
    }

    const cv::Mat trial_affine = refine_affine(anchor, blob_positions, running_affine, thresholds.icp_px);
    const float   jump_px      = translation_delta_px(trial_affine, running_affine);
    if (jump_px > thresholds.max_chained_delta_px)
    {
        spdlog::warn("repair_span: frame {} affine translation jumped {:.1f}px (>{:.1f}px = "
                     "1.2*d_nn) — likely row-alias, skipping",
                     frame_idx, jump_px, thresholds.max_chained_delta_px);
        return RepairOutcome::kAntialiasRejected;
    }
    running_affine = trial_affine;

    // Non-const: row-shift cure (below) may rewrite blob_to_gid / matched_count
    // in place when the virtual-row check fires and a single-row relabel
    // restores a clean homography fit.
    Assignment final_assignment =
        assign_greedy(anchor, blob_positions, running_affine, thresholds.final_gate_px);

    if (final_assignment.matched_count < 4)
    {
        spdlog::warn("repair_span: frame {} matched only {} markers — leaving as-is",
                     frame_idx, final_assignment.matched_count);
        return RepairOutcome::kAntialiasRejected;
    }

    // Post-repair structural check. The chained-delta guard only catches
    // column-shift aliases (magnitude 2*d_nn); row shifts sit at d_nn/sqrt(2)
    // ≈ 0.7*d_nn, under the 1.2*d_nn cap, so they slip through the first
    // gate. The virtual-row homography check catches them from the other
    // side by looking for unidentified blobs that fit an off-board lattice.
    if (has_virtual_row_alias(frame_idx, final_assignment, blob_positions, board, thresholds))
    {
        const std::optional<CureResult> cure = attempt_shift_cure(
            frame_idx, final_assignment, blob_positions, board, thresholds);
        if (!cure.has_value())
        {
            spdlog::warn("repair_span: frame {} row-aliased, no shift cure — invalidating",
                         frame_idx);
            return RepairOutcome::kPhantomRejected;
        }
        spdlog::info("repair_span: frame {} row-shift-cured via physical (dx={}, dy={}) — "
                     "re-labeled {} matches after phantom trip",
                     frame_idx, cure->dx, cure->dy, cure->matched_count);
        final_assignment.blob_to_gid   = cure->blob_to_gid;
        final_assignment.matched_count = cure->matched_count;
    }

    const cv::Mat1b image = load_frame_image(frame_entry);
    base::ImageDecoding repaired = build_repaired_decoding(
        frame_entry.coding_markers, final_assignment.blob_to_gid, board, image);

    decoded.erase(frame_idx);
    decoded.emplace(frame_idx, std::move(repaired));

    spdlog::info("repair_span: frame {} repaired via affine (anchor={}, matched={}/{}, dt_gap_ms={})",
                 frame_idx, span.anchor_idx, final_assignment.matched_count, anchor.gids.size(),
                 span.dt_at_gap_ns / 1'000'000ULL);
    return RepairOutcome::kCured;
}
}  // namespace

// Decode the per-frame backup image stored in a FrameCacheEntry. Lives at
// marker::repair namespace scope (not in the file-local anonymous block above)
// because CircleGridCalibInterface::rerenderRepairedDebugImages reaches into
// it from the parent kalibr package — see repair_dropped_neighbors.hpp for
// the public declaration.
cv::Mat1b load_frame_image(const FrameCacheEntry& entry)
{
    if (!entry.cached_image_png.empty())
    {
        // PNG is lossless so decoded pixels match the original gray frame.
        return cv::imdecode(entry.cached_image_png, cv::IMREAD_GRAYSCALE);
    }
    if (!entry.cached_image.empty())
    {
        return entry.cached_image;
    }
    if (!entry.image_path.empty())
    {
        return cv::imread(entry.image_path.string(), cv::IMREAD_GRAYSCALE);
    }
    return cv::Mat1b();
}

SpanRepairResult repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& /*base_params*/,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& /*output_path*/)
{
    SpanRepairResult result;
    const auto anchor_it = frame_cache.find(span.anchor_idx);
    if (anchor_it == frame_cache.end() || !anchor_it->second.fcg_succeeded)
    {
        spdlog::warn("repair_span: anchor {} missing or not FCG — skipping", span.anchor_idx);
        return result;
    }
    const AnchorReference anchor = extract_anchor_reference(anchor_it->second);
    if (anchor.gids.size() < 4)
    {
        spdlog::warn("repair_span: anchor {} has only {} identified markers — skipping span",
                     span.anchor_idx, anchor.gids.size());
        return result;
    }
    if (anchor.median_nn_distance_px <= 0.f)
    {
        spdlog::warn("repair_span: anchor {} has degenerate geometry (d_nn=0) — skipping span",
                     span.anchor_idx);
        return result;
    }
    const GeometricThresholds thresholds = thresholds_from_scale(anchor.median_nn_distance_px);
    spdlog::info("repair_span: anchor {} d_nn={:.1f}px -> icp[{:.1f},{:.1f},{:.1f},{:.1f}] "
                 "final_gate={:.1f}px chained_cap={:.1f}px span=[{},{}] length={}",
                 span.anchor_idx, anchor.median_nn_distance_px,
                 thresholds.icp_px[0], thresholds.icp_px[1], thresholds.icp_px[2], thresholds.icp_px[3],
                 thresholds.final_gate_px, thresholds.max_chained_delta_px,
                 span.start_idx, span.anchor_idx - 1,
                 span.anchor_idx - span.start_idx);

    cv::Mat running_affine = cv::Mat::eye(2, 3, CV_64F);
    for (int idx = span.anchor_idx - 1; idx >= span.start_idx; --idx)
    {
        const auto cache_it = frame_cache.find(idx);
        if (cache_it == frame_cache.end())
        {
            spdlog::warn("repair_span: frame {} missing from cache", idx);
            break;
        }
        const RepairOutcome outcome = repair_single_frame(
            idx, span, cache_it->second, anchor, thresholds, board,
            running_affine, decoded);
        switch (outcome)
        {
            case RepairOutcome::kCured:
            {
                result.cured.push_back(idx);
                break;
            }
            case RepairOutcome::kPhantomRejected:
            {
                result.phantom_rejected.push_back(idx);
                break;
            }
            case RepairOutcome::kAntialiasRejected:
            {
                // Fall through — left for the caller's gap sweep to invalidate.
                break;
            }
        }
    }
    std::sort(result.cured.begin(), result.cured.end());
    std::sort(result.phantom_rejected.begin(), result.phantom_rejected.end());
    spdlog::info("repair_span: span [{},{}] cured={} phantom={} / {} frames (dt_gap_ms={})",
                 span.start_idx, span.anchor_idx - 1,
                 result.cured.size(), result.phantom_rejected.size(),
                 span.anchor_idx - span.start_idx,
                 span.dt_at_gap_ns / 1'000'000ULL);
    return result;
}

void invalidate_span(
    const UnrecoverableSpan&            span,
    std::map<int, base::ImageDecoding>& decoded,
    const InvalidationReason            reason,
    const std::vector<int>&             skip_cured)
{
    const std::unordered_set<int> skip(skip_cured.begin(), skip_cured.end());
    int invalidated_count = 0;
    for (int idx = span.start_idx; idx <= span.end_idx; ++idx)
    {
        if (skip.count(idx))
        {
            continue;
        }
        const auto decoded_it = decoded.find(idx);
        if (decoded_it == decoded.end())
        {
            continue;
        }
        // Row/col from the existing ordering matrix so downstream consumers that
        // read ordering_.rows()/cols() still get the right board shape.
        const int board_rows = static_cast<int>(decoded_it->second.coding_markers_.ordering_.rows());
        const int board_cols = static_cast<int>(decoded_it->second.coding_markers_.ordering_.cols());
        decoded.erase(idx);
        decoded.emplace(idx, build_failed_decoding(board_rows, board_cols));
        ++invalidated_count;
    }
    const char* const reason_str =
        (reason == InvalidationReason::kPhantom) ? "phantom" : "gap";
    spdlog::info("invalidate_span: frames [{}..{}] dropped from calibration "
                 "({} entries, {} cured skipped, reason={}, dt_gap_ms={})",
                 span.start_idx, span.end_idx, invalidated_count,
                 static_cast<int>(skip.size()),
                 reason_str,
                 span.dt_at_gap_ns / 1'000'000ULL);
}

namespace
{
struct SpanOutcome
{
    Span              span;
    std::vector<int>  cured;              // frame indices cured by back-propagation
    std::vector<int>  phantom_rejected;   // frames flagged by the virtual-row check
};

void write_repair_report(
    const std::filesystem::path&               output_path,
    const uint64_t                             median_dt,
    const float                                gap_factor,
    const std::vector<SpanOutcome>&            span_outcomes,
    const std::vector<UnrecoverableSpan>&      unrecoverable_spans)
{
    if (output_path.empty())
    {
        return;
    }
    std::filesystem::create_directories(output_path);
    std::ofstream report(output_path / "repair_report.txt");
    if (!report.is_open())
    {
        return;
    }
    report << "# median_dt_ms: "   << (median_dt / 1'000'000ULL) << "\n";
    report << "# gap_factor: "     << gap_factor << "\n";
    report << "# spans: "          << span_outcomes.size() << "\n";
    report << "# unrecoverable: "  << unrecoverable_spans.size() << "\n";
    for (size_t span_idx = 0; span_idx < span_outcomes.size(); ++span_idx)
    {
        const Span& span = span_outcomes[span_idx].span;
        const int   span_length     = span.anchor_idx - span.start_idx;
        const int   cured_count     = static_cast<int>(span_outcomes[span_idx].cured.size());
        const int   phantom_count   = static_cast<int>(span_outcomes[span_idx].phantom_rejected.size());
        const int   antialias_count = span_length - cured_count - phantom_count;
        report << "span " << span_idx
               << ": start="              << span.start_idx
               << " anchor="              << span.anchor_idx
               << " dt_gap_ms="           << (span.dt_at_gap_ns / 1'000'000ULL)
               << " span_frames="         << span_length
               << " cured="               << cured_count
               << " phantom="             << phantom_count
               << " antialias="           << antialias_count
               << "\n";
    }
    for (size_t span_idx = 0; span_idx < unrecoverable_spans.size(); ++span_idx)
    {
        const auto& span = unrecoverable_spans[span_idx];
        report << "invalidated " << span_idx
               << ": start="             << span.start_idx
               << " end="                << span.end_idx
               << " dt_gap_ms="          << (span.dt_at_gap_ns / 1'000'000ULL)
               << " invalidated_frames=" << (span.end_idx - span.start_idx + 1)
               << "\n";
    }
}
}  // namespace

void run_repair_pass(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path)
{
    if (!base_params.repair_dropped_neighbors_)
    {
        spdlog::info("repair: disabled by repair_dropped_neighbors_=false");
        return;
    }

    const uint64_t median_dt = median_dt_ns(frame_cache);
    const std::vector<Span> spans = detect_drop_affected_spans(
        frame_cache, base_params.repair_gap_factor_);
    const std::vector<UnrecoverableSpan> unrecoverable_spans = detect_unrecoverable_spans(
        frame_cache, base_params.repair_gap_factor_);

    spdlog::info("repair: median_dt_ms={} gap_factor={} spans={} unrecoverable={}",
                 median_dt / 1'000'000ULL, base_params.repair_gap_factor_,
                 spans.size(), unrecoverable_spans.size());

    // Pass 1: back-propagate from each recoverable span's FCG anchor. The
    // anti-alias guard inside repair_span decides which frames cure; the
    // virtual-row homography check flags row-aliased ones as phantom; the
    // rest are invalidated below so they don't carry Hungarian IDs into
    // calibration.
    std::vector<SpanOutcome> span_outcomes;
    span_outcomes.reserve(spans.size());
    for (auto iter = spans.rbegin(); iter != spans.rend(); ++iter)
    {
        SpanOutcome outcome;
        outcome.span = *iter;
        SpanRepairResult repair_result = repair_span(
            *iter, frame_cache, base_params, board, decoded, output_path);
        outcome.cured            = std::move(repair_result.cured);
        outcome.phantom_rejected = std::move(repair_result.phantom_rejected);
        span_outcomes.push_back(std::move(outcome));
    }
    // Put span_outcomes back in start-index order for the report.
    std::sort(span_outcomes.begin(), span_outcomes.end(),
              [](const SpanOutcome& lhs, const SpanOutcome& rhs)
              { return lhs.span.start_idx < rhs.span.start_idx; });

    // Pass 2a: invalidate phantom-rejected frames explicitly under
    // reason=kPhantom. Each phantom-rejected frame is its own 1-frame
    // "span" for reporting purposes so the gap sweep below can't double-log
    // it. The phantom list is sparse and non-contiguous, so we process the
    // frames one at a time instead of constructing one big UnrecoverableSpan.
    for (const SpanOutcome& outcome : span_outcomes)
    {
        for (const int phantom_idx : outcome.phantom_rejected)
        {
            UnrecoverableSpan phantom_span;
            phantom_span.start_idx    = phantom_idx;
            phantom_span.end_idx      = phantom_idx;
            phantom_span.dt_at_gap_ns = outcome.span.dt_at_gap_ns;
            invalidate_span(phantom_span, decoded, InvalidationReason::kPhantom);
        }
    }

    // Pass 2b: invalidate the uncured, non-phantom frames of each recoverable
    // span. Their pass-1 Hungarian IDs are untrusted: the chain broke on the
    // anti-alias guard, so we can't vouch for them. Skip both the cured and
    // the phantom-rejected frames so we neither re-stomp their decodings nor
    // mislabel them as reason=kGap in the log.
    for (const SpanOutcome& outcome : span_outcomes)
    {
        UnrecoverableSpan virtual_span;
        virtual_span.start_idx    = outcome.span.start_idx;
        virtual_span.end_idx      = outcome.span.anchor_idx - 1;
        virtual_span.dt_at_gap_ns = outcome.span.dt_at_gap_ns;
        std::vector<int> skip;
        skip.reserve(outcome.cured.size() + outcome.phantom_rejected.size());
        skip.insert(skip.end(), outcome.cured.begin(), outcome.cured.end());
        skip.insert(skip.end(), outcome.phantom_rejected.begin(),
                    outcome.phantom_rejected.end());
        invalidate_span(virtual_span, decoded, InvalidationReason::kGap, skip);
    }

    // Pass 3: spans with no reachable FCG (hit another jump first, or EOF).
    for (const UnrecoverableSpan& unrecoverable : unrecoverable_spans)
    {
        invalidate_span(unrecoverable, decoded, InvalidationReason::kGap);
    }

    write_repair_report(output_path, median_dt, base_params.repair_gap_factor_,
                        span_outcomes, unrecoverable_spans);
}

}  // namespace marker::repair
