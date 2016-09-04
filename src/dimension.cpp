#include "dimension.h"
#include <iostream>

Dimension::Dimension(const std::vector<int>& sizes) : ndim_(sizes.size()) {
    for (int i = 0; i < sizes.size();i++) {
        sizes_[i] = sizes[i];
    }
}

Dimension::Dimension(const Dimension& other) : ndim_(other.ndim_) {
    for (int i = 0; i < ndim_;i++) {
        sizes_[i] = other.sizes_[i];
    }
}

std::ostream& operator<<(std::ostream& stream, const Dimension& dims) {
    stream << "(";
    for (int i = 0; i < dims.ndim();i++) {
        stream << dims[i];
        if (i != dims.ndim() - 1) {
            stream << ", ";
        } else {
            stream << ")";
        }
    }
    return stream;
}
