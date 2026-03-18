#pragma once

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "board.hpp"
#include "calibration.hpp"

class BoardCircleGrid;

namespace identification::circlegrid
{

struct MarkerTrack
{
    std::array<cv::Point2f, 3> position_history{};
    std::array<int, 3> frame_ids{};
    int history_count = 0;  // 0-3 entries stored
    cv::Point2f last_position;
    int global_id = -1;
    int last_seen_frame = 0;
    int age = 0;
};

struct TrackingStatistics
{
    int frame_id = -1;
    int detected_count = 0;
    int identified_count = 0;
    std::string method;
    float avg_assignment_cost = 0.f;
    float max_assignment_cost = 0.f;
    bool was_180_flipped = false;
    float mean_velocity_magnitude = 0.f;
};

struct HungarianTrackingResult
{
    std::vector<int> global_ids;
    float avg_cost = 0.f;
    float max_cost = 0.f;
    int matched_count = 0;
};

/// ORB-based motion field for velocity interpolation
struct ORBMotionField
{
    std::vector<cv::Point2f> locations;      // matched feature positions in current frame
    std::vector<cv::Point2f> displacements;  // motion vectors (curr - prev)
    bool valid = false;
};

/// Blob-level velocity+acceleration field — anchored at blob positions, ID-independent.
/// Uses identified marker tracks for exact velocity/acceleration computation.
/// A single global similarity model (vCx, vCy, ω, σ) is fitted via RANSAC from all markers.
/// The similarity model captures translation + rotation + isotropic scale (approach/recede).
struct BlobVelocityField
{
    // Per-marker data (kept for residuals and debug visualization)
    std::vector<cv::Point2f> positions;        // anchor positions
    std::vector<cv::Point2f> velocities;       // displacement vectors per frame (px/frame)
    std::vector<cv::Point2f> accelerations;    // central-diff acceleration (px/frame²), zero if unavailable

    // Global similarity velocity model: v(p) = v_C + ω × r + σ · r
    //   vx = vCx - ω*ry + σ*rx
    //   vy = vCy + ω*rx + σ*ry
    cv::Point2f centroid{0.f, 0.f};
    float vCx = 0.f, vCy = 0.f;               // translation velocity at centroid (px/frame)
    float omega = 0.f;                          // angular velocity (rad/frame)
    float sigma = 0.f;                          // isotropic scale rate (1/frame), >0 = expanding

    // Global acceleration model: a(p) = a_C + ε × r + σ_dot · r + centripetal terms
    float aCx = 0.f, aCy = 0.f;               // translation acceleration at centroid (px/frame²)
    float epsilon = 0.f;                        // angular acceleration (rad/frame²)
    float sigma_dot = 0.f;                      // scale acceleration (1/frame²)
    bool has_acceleration = false;

    // Fit quality metrics
    float velocity_residual_rms = 0.f;
    int velocity_inlier_count = 0;
    std::vector<bool> velocity_inliers;         // per-marker inlier flag (for debug viz)

    bool valid = false;

    /// Evaluate the global similarity model at a query point.
    /// Returns predicted POSITION offset: Δpos = v*dt + 0.5*a*dt²
    cv::Point2f transport_predict(const cv::Point2f& query, float dt = 1.f) const;
};

struct TrackingState
{
    std::vector<base::MarkerCoding> prev_markers_;
    std::vector<int> prev_global_ids_;
    cv::Mat1b prev_image_;
    bool has_previous_ = false;

    std::string last_method_;

    // Per-global-id velocity tracks
    std::unordered_map<int, MarkerTrack> tracks_;
    bool orientation_locked_ = false;
    int frame_counter_ = 0;

    // For 180° ambiguity resolution across findCirclesGrid calls
    std::vector<cv::Point2f> prev_findcircles_centers_;

    // 3-frame history of ALL detected marker positions (for velocity interpolation)
    std::array<std::vector<cv::Point2f>, 3> detected_positions_history_;
    int history_write_idx_ = 0;

    // Last findCirclesGrid frame's marker positions (trusted reference for swap detection)
    // Indexed by global_id: last_fcg_positions_[gid] = image position
    std::vector<cv::Point2f> last_fcg_positions_;
    int last_fcg_frame_ = -1;

    // ORB motion field data
    std::vector<cv::KeyPoint> prev_orb_keypoints_;
    cv::Mat prev_orb_descriptors_;

    // Blob velocity fields for bidirectional acceptance check
    BlobVelocityField forward_blob_field_;       // anchored at prev positions, vel+acc forward
    BlobVelocityField backward_blob_field_;      // anchored at curr positions, vel+acc backward
    std::vector<cv::Point2f> prev_blob_positions_;  // raw positions from previous frame

    // Debug: per-marker prediction vectors (from→to) for visualization
    // Populated during Hungarian tracking, drawn by showExtractionVisualization
    std::vector<std::pair<cv::Point2f, cv::Point2f>> debug_prediction_vectors_;  // (current_pos, predicted_pos)

    void update(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids,
                const cv::Mat1b& image);
    void update_blob_velocity_fields(const std::vector<base::MarkerCoding>& curr_markers);
    void clear();

    /// Predict position for a tracked marker using up to 3 frames of history
    cv::Point2f predict_position(int global_id, int current_frame) const;

    /// Predict position with ORB-interpolated velocity for stale tracks
    cv::Point2f predict_position_with_orb(int global_id, int current_frame,
                                           const ORBMotionField& motion_field) const;

    /// Estimate motion field from ORB features between prev and current image
    ORBMotionField estimate_motion_field(const cv::Mat1b& current_image) const;

    /// Update ORB keypoints after processing a frame
    void update_orb(const cv::Mat1b& image);

    /// Update velocity tracks after identification
    void update_tracks(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids);
};

// --- Existing functions ---

std::optional<std::vector<int>> identify_with_tracking(const std::vector<base::MarkerCoding>& prev_markers,
                                                       const std::vector<base::MarkerCoding>& curr_markers,
                                                       const std::vector<int>& prev_ids, float distance_threshold,
                                                       float ransac_threshold);

void identify_new_markers_by_row_lines(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board);

bool test_find_circles_grid(std::vector<int>& indices, const std::vector<base::MarkerCoding>& coding_markers,
                            const BoardCircleGrid& board, TrackingState& tracker_state);

bool validate_tracking_with_ecc(const std::vector<base::MarkerCoding>& prev_markers,
                                const std::vector<base::MarkerCoding>& curr_markers, const cv::Mat1b& prev_image,
                                const cv::Mat1b& curr_image, float distance_threshold = 50.0f,
                                float ransac_threshold = 5.0f, float ecc_threshold = 0.7f);

// --- New functions ---

/// Jonker-Volgenant O(n^3) assignment for rectangular cost matrices
std::vector<int> hungarian_assignment(const std::vector<std::vector<float>>& cost_matrix, float max_cost);

/// Velocity-predicted Hungarian tracking
HungarianTrackingResult identify_with_hungarian_tracking(const TrackingState& state,
                                                         const std::vector<base::MarkerCoding>& curr_markers,
                                                         const BoardCircleGrid& board, float max_distance = 80.0f,
                                                         float ransac_threshold = 5.0f,
                                                         const ORBMotionField* motion_field = nullptr);

/// Check if board has 180-degree rotational ambiguity
bool board_has_180_ambiguity(const BoardCircleGrid& board);

/// Compute 180-degree flipped global IDs
std::vector<int> flip_ids_180(const std::vector<int>& global_ids, const BoardCircleGrid& board);

/// Resolve 180-degree ambiguity using velocity consistency
std::vector<int> resolve_180_ambiguity(const std::vector<int>& global_ids,
                                       const std::vector<base::MarkerCoding>& curr_markers,
                                       const TrackingState& state, const BoardCircleGrid& board);

/// Re-identify unmatched markers using local homography (distortion-invariant)
void identify_unmatched_by_local_homography(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board);

/// Validate marker topology using RANSAC homography and correct row swaps
void validate_and_correct_topology(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board);

}  // namespace identification::circlegrid
