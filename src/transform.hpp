#pragma once

#include <Eigen/Dense>

struct TransformData
{
    std::vector<float> rotation_;
    std::vector<float> translation_;
};

template <typename Floating>
class Transform
{
   public:
    Eigen::Quaternion<Floating> rotation_;
    Eigen::Matrix<Floating, 3, 1> translation_;

    static Transform identity()
    {
        Transform<Floating> transform;
        transform.rotation_ = Eigen::Quaternion<Floating>::Identity();
        transform.translation_ = Eigen::Matrix<Floating, 3, 1>::Zero();
        return transform;
    }

    Transform<Floating> inverse() const
    {
        Transform<Floating> temp;
        temp.rotation_ = rotation_.conjugate();
        temp.translation_ = temp.rotation_ * translation_ * Floating(-1.0);
        return temp;
    }

    Eigen::Matrix<Floating, 3, 1> transform_point(const Eigen::Matrix<Floating, 3, 1>& point) const
    {
        return rotation_ * point + translation_;
    }
    void transform_point_in_place(Eigen::Matrix<Floating, 3, 1>& point) const
    {
        point = rotation_ * point + translation_;
    }

    Transform<Floating> operator*(const Transform<Floating>& other) const
    {
        return Transform<Floating>(rotation_ * other.rotation_, translation_ + rotation_ * other.translation_);
    }

    Transform() = default;
    Transform(const Eigen::Quaternion<Floating>& rotation, const Eigen::Matrix<Floating, 3, 1>& translation)
        : rotation_(rotation), translation_(translation)
    {
    }
    Transform(const TransformData& data) { from_transform_data(data); }

    TransformData to_transform_data() const
    {
        TransformData data;
        data.rotation_ = {rotation_.coeffs()[0], rotation_.coeffs()[1], rotation_.coeffs()[2], rotation_.coeffs()[3]};
        data.translation_ = {translation_(0), translation_(1), translation_(2)};

        return data;
    }

    void from_transform_data(const TransformData& data)
    {
        rotation_.coeffs()[0] = data.rotation_[0];
        rotation_.coeffs()[1] = data.rotation_[1];
        rotation_.coeffs()[2] = data.rotation_[2];
        rotation_.coeffs()[3] = data.rotation_[3];

        translation_[0] = data.translation_[0];
        translation_[1] = data.translation_[1];
        translation_[2] = data.translation_[2];
    }

    template <typename OtherFloating>
    Transform<OtherFloating> cast() const
    {
        if constexpr (std::is_same<Floating, OtherFloating>::value)
        {
            return *this;
        }

        return Transform<OtherFloating>(rotation_.template cast<OtherFloating>(),
                                        translation_.template cast<OtherFloating>());
    }
};

typedef Transform<double> Transformd;
typedef Transform<float> Transformf;
