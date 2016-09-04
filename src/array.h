#ifndef RTC_ARRAY_H
#define RTC_ARRAY_H

#include <memory>
#include <iostream>

#include "device.h"
#include "dimension.h"
#include "array_like.h"
#include "array_reference.h"

template<typename T>
struct ArrayGather;

template<typename T>
struct Array : public ArrayLike<Array<T>, T> {
    std::shared_ptr<int> owners_;
    Array();
    void allocate_data();
    void allocate_data_cpu();
    void allocate_data_gpu();
    Array(const Dimension& sizes, DEVICE_T dev);
    Array<T> subtensor(int idx) const;
    Array<T> slice(int start, int end) const;
    ArrayGather<T> operator[](const Array<int>& indices);
    Array<T> to_device(DEVICE_T dev) const;
    Array(const Array<T>& other);
    void free_data();
    void free_data_cpu();
    void free_data_gpu();
    ~Array();
    void print() const;
    Array& operator=(const double& other);
    Array& operator+=(const double& other);
    Array& operator-=(const double& other);
    void print(int indent, bool print_comma) const;
    ArrayReference<T> view();
    const ArrayReference<T> const_view() const;

    T& XINLINE operator[](const int& index) {
        return *(this->ptr_ + this->offset_ + index);
    }

    const T& XINLINE operator[](const int& index) const {
        return *(this->ptr_ + this->offset_ + index);
    }

    T& XINLINE operator[](const Dimension& indices) {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }

    const T& XINLINE operator[](const Dimension& indices) const {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }
};

#endif
