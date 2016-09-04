#ifndef RTC_ARRAY_GATHER_H
#define RTC_ARRAY_GATHER_H

#include "array.h"

template<typename T>
struct ArrayGather {
    Array<T> array_;
    Array<int> indices_;
    ArrayGather(const Array<T>& array, const Array<int>& indices);
    ArrayGather<T>& operator+=(const Array<T>& updates);
    ArrayGather<T>& operator=(const Array<T>& updates);
    ArrayGather<T>& operator-=(const Array<T>& updates);
};

#endif
