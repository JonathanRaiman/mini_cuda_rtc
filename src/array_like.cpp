#include "array_like.h"
#include <iostream>

std::ostream& operator<<(std::ostream& stream, const DEVICE_T& device) {
    return stream << ((device == DEVICE_T_CPU) ? "cpu" : "gpu");
}
