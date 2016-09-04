#ifndef RTC_ARRAY_LIKE_H
#define RTC_ARRAY_LIKE_H

#include "dimension.h"

enum DEVICE_T {
    DEVICE_T_CPU,
    DEVICE_T_GPU
};

std::ostream& operator<<(std::ostream& stream, const DEVICE_T& device);

template<typename Container, typename T>
struct ArrayLike {
    Dimension sizes_;
    T* ptr_;
    int offset_;
    const DEVICE_T device_;

    XINLINE int numel() const {
        return sizes_.numel();
    }

    XINLINE int offset() const {
        return offset_;
    }

    XINLINE const Dimension& size() const {
        return sizes_;
    }

    XINLINE T& operator[](const int& index) {
        return *(ptr_ + offset_ + index);
    }

    XINLINE const T& operator[](const int& index) const {
        return *(ptr_ + offset_ + index);
    }

    XINLINE T& operator[](const Dimension& indices) {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }

    XINLINE const T& operator[](const Dimension& indices) const {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }

    XINLINE int ndim() const {
        return sizes_.ndim();
    }

    ArrayLike(const Dimension& sizes, const DEVICE_T& dev, T* ptr, const int& offset)
        : sizes_(sizes),
          device_(dev),
          ptr_(ptr),
          offset_(offset) {}
};

#endif
