#define INDENT_INCREMENT 2

#include <string>
#include <iostream>

#include "kernels.h"
#include "functor.h"
#include "array_gather.h"
#include "make_message.h"

template<typename T>
Array<T>::Array() : ArrayLike<Array<T>, T>({}, DEVICE_T_CPU, nullptr, 0),
          owners_(std::make_shared<int>(1)) {}

template<typename T>
void Array<T>::allocate_data() {
    if (this->device_ == DEVICE_T_CPU) {
        allocate_data_cpu();
    } else {
        allocate_data_gpu();
    }
}

template<typename T>
void Array<T>::allocate_data_cpu() {
    auto ptr = malloc(this->sizes_.numel() * sizeof(T));
    if (ptr == NULL) {
        throw std::runtime_error(
            make_message(
                "could not allocate ",
                this->sizes_.numel() * sizeof(T),
                " bytes on the cpu"
            )
        );
    } else {
        this->ptr_ = (T*)ptr;
    }
}

template<typename T>
void Array<T>::allocate_data_gpu() {
    Device::set_gpu_device(0);
    auto status = cudaMalloc(&this->ptr_, this->sizes_.numel() * sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error(
            make_message(
                "could not allocate ",
                this->sizes_.numel() * sizeof(T),
                " bytes on the gpu, reason = ",
                cudaGetErrorString(status)
            )
        );
    }
}

template<typename T>
Array<T>::Array(const Dimension& sizes, DEVICE_T dev)
        : ArrayLike<Array<T>, T>(sizes, dev, nullptr, 0),
          owners_(std::make_shared<int>(1)) {
    allocate_data();
}

template<typename T>
Array<T> Array<T>::subtensor(int idx) const {
    auto arr = slice(idx, idx + 1);
    arr.sizes_ = arr.sizes_.subsize();
    return arr;
}

template<typename T>
Array<T> Array<T>::slice(int start, int end) const {
    Array<T> arr = *this;
    int subvolume = 1;
    for (int i = 1; i < this->ndim(); i++) {
        subvolume *= this->sizes_[i];
    }
    arr.offset_ = this->offset_ + start * subvolume;
    arr.sizes_.set_dim(0, end - start);
    return arr;
}

template<typename T>
Array<T> Array<T>::to_device(DEVICE_T dev) const {
    if (this->device_ == dev) {
        return *this;
    } else {
        Array<T> arr(this->sizes_, dev);

        auto copy_type = this->device_ == DEVICE_T_CPU ?
            cudaMemcpyHostToDevice :
            cudaMemcpyDeviceToHost;

        // gpu -> cpu
        auto status = cudaMemcpy(
            arr.ptr_ + arr.offset_,
            this->ptr_ + this->offset_,
            this->sizes_.numel() * sizeof(T),
            copy_type
        );

        if (status != cudaSuccess) {
            std::string reason;

            throw std::runtime_error(
                make_message(
                    "could not copy ",
                    this->sizes_.numel() * sizeof(T),
                    " bytes from ",
                    this->device_ == DEVICE_T_CPU ?
                    "cpu to gpu" : "gpu to cpu",
                    ", reason = ",
                    cudaGetErrorString(status)
                )
            );
        }
        return arr;
    }
}

template<typename T>
Array<T>::Array(const Array<T>& other)
    : ArrayLike<Array<T>, T>(other.sizes_, other.device_, other.ptr_, other.offset_),
      owners_(other.owners_) {
    (*owners_) += 1;
}

template<typename T>
void Array<T>::free_data() {
    if (this->device_ == DEVICE_T_CPU) {
        free_data_cpu();
    } else {
        free_data_gpu();
    }
}

template<typename T>
void Array<T>::free_data_cpu() {
    free((void*)this->ptr_);
}

template<typename T>
void Array<T>::free_data_gpu() {
    auto status = cudaFree((void*)this->ptr_);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            make_message(
                "could not free memory on device : ",
                status == cudaErrorInvalidDevicePointer ?
                    "cudaErrorInvalidDevicePointer" :
                    "cudaErrorInitializationError"
            )
        );
    }
}

template<typename T>
Array<T>::~Array() {
    (*owners_) -= 1;
    if (this->ptr_ != nullptr && (*owners_) == 0) {
        free_data();
    }
}

template<typename T>
void Array<T>::print() const {
    return print(0, false);
}

template<typename T>
void Array<T>::print(int indent, bool print_comma) const {
    if (this->device_ == DEVICE_T_CPU) {
        if (this->ndim() == 1) {
            std::cout << std::string(indent, ' ') << "[";
            for (int i = 0; i < this->sizes_[0]; i++) {
                std::cout << (*this)[i];
                if (i != this->sizes_[0] - 1) {
                    std::cout << " ";
                }
            }
            if (print_comma) {
                std::cout << "],\n";
            } else {
                std::cout << "]\n";
            }
        } else if (this->ndim() > 1) {
            std::cout << std::string(indent, ' ') << "[\n";
            for (int i = 0; i < this->sizes_[0]; i++) {
                auto child = subtensor(i);
                child.print(indent + INDENT_INCREMENT, i != this->sizes_[0] - 1);
            }
            if (print_comma) {
                std::cout << std::string(indent, ' ') << "],\n";
            } else {
                std::cout << std::string(indent, ' ') << "]\n";
            }
        } else {
            std::cout << std::string(indent, ' ') << "()\n" << std::endl;
        }
    } else {
        auto arr = to_device(DEVICE_T_CPU);
        arr.print();
    }
}

template<typename T>
ArrayReference<T> Array<T>::view() {
    return ArrayReference<T>(
        this->sizes_,
        this->device_,
        this->ptr_,
        this->offset_
    );
}

template<typename T>
const ArrayReference<T> Array<T>::const_view() const {
    return ArrayReference<T>(
        this->sizes_,
        this->device_,
        this->ptr_,
        this->offset_
    );
}

template<typename T>
Array<T>& Array<T>::operator=(const double& other) {
    if (this->device_ == DEVICE_T_CPU) {
        for (int i = 0; i < this->numel();i++) {
            (*this)[i] = other;
        }
    } else {
        launch_unary_kernel(functor::Fill(other), *this, *this);
    }
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator+=(const double& other) {
    if (this->device_ == DEVICE_T_CPU) {
        for (int i = 0; i < this->numel();i++) {
            (*this)[i] += other;
        }
    } else {
        launch_unary_kernel(functor::Increment(other), *this, *this);
    }
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator-=(const double& other) {
    if (this->device_ == DEVICE_T_CPU) {
        for (int i = 0; i < this->numel();i++) {
            (*this)[i] -= other;
        }
    } else {
        launch_unary_kernel(functor::Decrement(other), *this, *this);
    }
    return *this;
}

template<typename T>
ArrayGather<T> Array<T>::operator[](const Array<int>& indices) {
    return ArrayGather<T>(*this, indices);
}
