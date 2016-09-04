#ifndef RTC_ARRAY_REFERENCE_H
#define RTC_ARRAY_REFERENCE_H
#include "array_like.h"
template<typename T>
struct ArrayReference : public ArrayLike<ArrayReference<T>, T> {
    using ArrayLike<ArrayReference<T>, T>::ArrayLike;
};
#endif
