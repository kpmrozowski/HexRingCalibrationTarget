#include "repair_dropped_neighbors.hpp"

#include <algorithm>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "marker/detection.hpp"

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

std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const float gap_factor,
    const int   max_span_len)
{
    std::vector<Span> spans;
    const uint64_t median_dt = median_dt_ns(frame_cache);
    if (median_dt == 0)
    {
        return spans;
    }
    const uint64_t threshold_ns =
        static_cast<uint64_t>(static_cast<double>(median_dt) * static_cast<double>(gap_factor));

    auto prev_it = frame_cache.begin();
    for (auto it = std::next(prev_it); it != frame_cache.end(); ++it, ++prev_it)
    {
        const uint64_t t_curr = it->second.ts_ns;
        const uint64_t t_prev = prev_it->second.ts_ns;
        if (t_curr == 0 || t_prev == 0 || t_curr <= t_prev)
        {
            continue;
        }
        const uint64_t dt = t_curr - t_prev;
        if (dt <= threshold_ns)
        {
            continue;
        }

        Span span;
        span.start_idx    = it->first;
        span.dt_at_gap_ns = dt;

        for (auto scan = it; scan != frame_cache.end(); ++scan)
        {
            if (scan->first - span.start_idx > max_span_len)
            {
                break;
            }
            if (scan->second.fcg_succeeded && scan->first > span.start_idx)
            {
                span.anchor_idx = scan->first;
                break;
            }
        }
        if (span.anchor_idx < 0)
        {
            spdlog::warn("repair: gap at frame {} (dt={}ms) has no FCG anchor within {} frames; skipping",
                         span.start_idx, dt / 1'000'000ULL, max_span_len);
            continue;
        }
        spans.push_back(span);
    }
    return spans;
}

namespace
{
// Build a fresh TrackingState seeded from a findCirclesGrid anchor frame so
// the detector's H2 FCG pixel-to-pixel affine recovery can run against it.
// has_previous_ is left false so Hungarian tracking is skipped — H2 alone
// drives identification, avoiding stale/reverse-direction velocity artefacts.
identification::circlegrid::TrackingState make_seeded_tracker(
    const FrameCacheEntry& anchor_entry, const int anchor_idx, const int total_markers)
{
    identification::circlegrid::TrackingState fresh;
    fresh.fcg_ever_succeeded_ = true;
    fresh.last_fcg_frame_     = anchor_idx;
    fresh.last_fcg_positions_.assign(total_markers, cv::Point2f(-1.f, -1.f));
    for (int gid = 0;
         gid < total_markers && gid < static_cast<int>(anchor_entry.marker_positions.size());
         ++gid)
    {
        fresh.last_fcg_positions_[gid] = anchor_entry.marker_positions[gid];
    }
    fresh.frame_counter_ = anchor_idx;
    return fresh;
}
}  // namespace

void repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path)
{
    const auto anchor_it = frame_cache.find(span.anchor_idx);
    if (anchor_it == frame_cache.end() || !anchor_it->second.fcg_succeeded)
    {
        spdlog::warn("repair_span: anchor {} missing or not FCG — skipping", span.anchor_idx);
        return;
    }
    const int total_markers = static_cast<int>(board.rows_ * board.cols_);

    auto tracker = make_seeded_tracker(anchor_it->second, span.anchor_idx, total_markers);

    DetectionParameters params = base_params;
    params.repair_dropped_neighbors_ = false;

    for (int idx = span.anchor_idx - 1; idx >= span.start_idx; --idx)
    {
        const auto cache_it = frame_cache.find(idx);
        if (cache_it == frame_cache.end())
        {
            spdlog::warn("repair_span: frame {} missing from cache", idx);
            return;
        }

        cv::Mat1b image;
        if (!cache_it->second.cached_image.empty())
        {
            image = cache_it->second.cached_image;
        }
        else if (!cache_it->second.image_path.empty())
        {
            image = cv::imread(cache_it->second.image_path.string(), cv::IMREAD_GRAYSCALE);
        }
        if (image.empty())
        {
            spdlog::warn("repair_span: no image available for frame {} (neither cached nor path)", idx);
            return;
        }

        base::ImageDecoding repaired = marker::detection::detect_and_identify_circlegrid(
            image, params, board, tracker, idx, output_path, cache_it->second.coding_markers);

        decoded.erase(idx);
        decoded.emplace(idx, std::move(repaired));
        spdlog::info("repair_span: frame {} repaired (anchor={}, dt_gap_ms={})",
                     idx, span.anchor_idx, span.dt_at_gap_ns / 1'000'000ULL);
    }
}

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
        frame_cache, base_params.repair_gap_factor_, base_params.repair_max_span_len_);

    spdlog::info("repair: median_dt_ms={} gap_factor={} spans={}",
                 median_dt / 1'000'000ULL, base_params.repair_gap_factor_, spans.size());

    for (auto iter = spans.rbegin(); iter != spans.rend(); ++iter)
    {
        repair_span(*iter, frame_cache, base_params, board, decoded, output_path);
    }

    if (!output_path.empty())
    {
        std::filesystem::create_directories(output_path);
        std::ofstream report(output_path / "repair_report.txt");
        if (report.is_open())
        {
            report << "# median_dt_ms: " << (median_dt / 1'000'000ULL) << "\n";
            report << "# gap_factor: "   << base_params.repair_gap_factor_ << "\n";
            report << "# spans: "        << spans.size() << "\n";
            for (size_t span_idx = 0; span_idx < spans.size(); ++span_idx)
            {
                const auto& span = spans[span_idx];
                report << "span " << span_idx
                       << ": start="           << span.start_idx
                       << " anchor="           << span.anchor_idx
                       << " dt_gap_ms="        << (span.dt_at_gap_ns / 1'000'000ULL)
                       << " repaired_frames="  << (span.anchor_idx - span.start_idx)
                       << "\n";
            }
        }
    }
}

}  // namespace marker::repair
