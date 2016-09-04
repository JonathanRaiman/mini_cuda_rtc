#ifndef RTC_FUNCTOR_H
#define RTC_FUNCTOR_H

#define XINLINE __device__ __host__

namespace functor {
    struct Fill {
        double fill_;
        XINLINE Fill(double fill) : fill_(fill) {}
        template<typename T>
        T XINLINE operator()(const T& val) {
            return fill_;
        }
    };
    struct Increment {
        double inc_;
        XINLINE Increment(double inc) : inc_(inc) {}
        template<typename T>
        T XINLINE operator()(const T& val) {
            return val + inc_;
        }
    };

    struct Decrement {
        double dec_;
        XINLINE Decrement(double dec) : dec_(dec) {}
        template<typename T>
        T XINLINE operator()(const T& val) {
            return val + dec_;
        }
    };
}

namespace saver {
    struct Increment {
        template<typename T>
        static void XINLINE save(T& left, const T& right) {
            left += right;
        }
    };
    struct Decrement {
        template<typename T>
        static void XINLINE save(T& left, const T& right) {
            left -= right;
        }
    };
    struct Assign {
        template<typename T>
        static void XINLINE save(T& left, const T& right) {
            left = right;
        }
    };
}

#endif
