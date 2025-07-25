#include "calibration.hpp"

base::MarkerUnidentified::MarkerUnidentified(const int label, const float row, const float col, const Type type,
                                             const int width, const int height, float black_value, float white_value)
    : label_(label),
      row_(row),
      col_(col),
      type_(type),
      width_(width),
      height_(height),
      black_value_(black_value),
      white_value_(white_value)
{
}

base::MarkerCoding::MarkerCoding(const int label_ring, const float row, const float col, const int width_ring,
                                 const int height_ring, float black_value, const float white_value)
    : label_ring_(label_ring),
      row_(row),
      col_(col),
      width_ring_(width_ring),
      height_ring_(height_ring),
      black_value_(black_value),
      white_value_(white_value)
{
}

base::MarkerRing::MarkerRing(const MarkerUnidentified &inner, const MarkerUnidentified &ring)
    : lalbe_inner_(inner.label_),
      label_ring_(ring.label_),
      row_((inner.row_ + ring.row_) / 2.0f),
      col_((inner.col_ + ring.col_) / 2.0f),
      width_inner_(inner.width_),
      height_inner_(inner.height_),
      width_ring_(ring.width_),
      height_ring_(ring.height_),
      black_value_((inner.black_value_ + ring.black_value_) / 2.0f),
      white_value_((inner.white_value_ + ring.white_value_) / 2.0f)
{
}

base::MarkerRing::MarkerRing(const MarkerCoding &coding)
    : lalbe_inner_(-1),
      label_ring_(coding.label_ring_),
      row_(coding.row_),
      col_(coding.col_),
      width_inner_(coding.width_ring_ * 0.5f),
      height_inner_(coding.height_ring_ * 0.5f),
      width_ring_(coding.width_ring_),
      height_ring_(coding.height_ring_),
      black_value_(coding.black_value_),
      white_value_(coding.white_value_)
{
}

base::CodingMarkers base::CodingMarkers::without_coding_markers() const
{
    base::CodingMarkers without_coding(ordering_, markers_);

    std::vector<bool> used(markers_.size(), false);

    for (int row = 0; row < without_coding.ordering_.rows(); ++row)
    {
        for (int col = 0; col < without_coding.ordering_.cols(); ++col)
        {
            if (without_coding.ordering_(row, col).has_value())
            {
                const int idx = without_coding.ordering_(row, col).value();
                if (markers_[idx].lalbe_inner_ == -1)
                {
                    without_coding.ordering_(row, col) = std::nullopt;
                }
                else
                {
                    used[without_coding.ordering_(row, col).value()] = true;
                }
            }
        }
    }

    // position is old idx, value is new idx
    std::vector<int> to_new_idx(without_coding.markers_.size(), -1);
    std::vector<base::MarkerRing> rings_shrink;

    for (size_t idx = 0; idx < used.size(); ++idx)
    {
        if (used[idx])
        {
            to_new_idx[idx] = rings_shrink.size();
            rings_shrink.emplace_back(without_coding.markers_[idx]);
        }
    }

    for (int row = 0; row < without_coding.ordering_.rows(); ++row)
    {
        for (int col = 0; col < without_coding.ordering_.cols(); ++col)
        {
            if (without_coding.ordering_(row, col).has_value())
            {
                without_coding.ordering_(row, col) = to_new_idx[without_coding.ordering_(row, col).value()];
            }
        }
    }
    without_coding.markers_ = rings_shrink;

    return without_coding;
}

base::CodingMarkers::CodingMarkers(const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                                   const std::vector<MarkerRing> &markers)
    : ordering_(ordering), markers_(markers)
{
}

base::ValidityLocation::ValidityLocation(const cv::Mat1b &markers_location, const cv::Mat1b &calibrated_area)
    : markers_location_(markers_location), calibrated_area_(calibrated_area)
{
}

base::ImageDecoding::ImageDecoding(const cv::Mat1b &linear_input, const cv::Mat1b &binary,
                                   const cv::Mat1b &inverted_binary,
                                   const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                                   const std::vector<base::MarkerRing> &markers, const cv::Mat1b &markers_location,
                                   const cv::Mat1b &calibrated_area)
    : linear_input_(linear_input),
      binary_(binary),
      inverted_binary_(inverted_binary),
      coding_markers_(ordering, markers),
      validity_(markers_location, calibrated_area)
{
}
