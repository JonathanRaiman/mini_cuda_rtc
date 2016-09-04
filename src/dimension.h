#ifndef RTC_DIMENSION_H
#define RTC_DIMENSION_H

#include <vector>
#include <initializer_list>

#define MAX_DIM 10
#ifdef __CUDACC__
#define XINLINE __device__ __host__
#else
#define XINLINE inline
#endif

struct Dimension {
    int sizes_[MAX_DIM];
    int ndim_;

    Dimension(const std::vector<int>& sizes);

    XINLINE Dimension(std::initializer_list<int> sizes) : ndim_(sizes.size()) {
        int i = 0;
        for (auto iter = sizes.begin(); iter != sizes.end(); iter++) {
            sizes_[i] = *iter;
            i++;
        }
    }

    XINLINE ~Dimension() {}

    int XINLINE ndim() const {
        return ndim_;
    }

    int XINLINE operator[](int dim) const {
        return sizes_[dim];
    }

    void XINLINE set_dim(int dim, int value) {
        sizes_[dim] = value;
    }

    Dimension(const Dimension& other);

    XINLINE Dimension& operator=(const Dimension& other) {
        ndim_ = other.ndim();
        for (int i = 0; i < other.ndim(); i++) {
            sizes_[i] = other[i];
        }
        return *this;
    }

    int XINLINE numel() const {
        int volume = 1;
        for (int i = 0; i < ndim_; i++) {
            volume *= sizes_[i];
        }
        return volume;
    }

    Dimension XINLINE subsize() const {
        Dimension subdim({});
        subdim.ndim_ = ndim_ - 1;
        for (int i = 1; i < ndim_; i++) {
            subdim.set_dim(i - 1, sizes_[i]);
        }
        return subdim;
    }

    static int XINLINE index2offset(const Dimension& sizes, const Dimension& indices) {
        int offset = 0;
        int volume = 1;
        for (int i = indices.ndim() - 1; i >= 0; i--) {
            offset += indices[i] * volume;
            volume *= sizes[i];
        }
        return offset;
    }
};

std::ostream& operator<<(std::ostream& stream, const Dimension& dims);

#endif
