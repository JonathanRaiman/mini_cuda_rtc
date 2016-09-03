#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <iostream>

#define XINLINE __device__ __host__
#define MAX_DIM 10
#define INDENT_INCREMENT 2

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

int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

void make_message(std::stringstream* ss) {}

template<typename Arg, typename... Args>
void make_message(std::stringstream* ss, const Arg& arg, const Args&... args) {
    (*ss) << arg;
    make_message(ss, args...);
}

template<typename... Args>
std::string make_message(const Args&... args) {
    std::stringstream ss;
    make_message(&ss, args...);
    return ss.str();
}

struct Dimension {
    int sizes_[MAX_DIM];
    int ndim_;

    Dimension(const std::vector<int>& sizes) : ndim_(sizes.size()) {
        for (int i = 0; i < sizes.size();i++) {
            sizes_[i] = sizes[i];
        }
    }

    XINLINE Dimension(std::initializer_list<int> sizes) : ndim_(sizes.size()) {
        int i = 0;
        for (auto iter = sizes.begin(); iter != sizes.end(); iter++) {
            sizes_[i] = *iter;
            i++;
        }
    }

    XINLINE ~Dimension() {
    }

    int XINLINE ndim() const {
        return ndim_;
    }

    int XINLINE operator[](int dim) const {
        return sizes_[dim];
    }

    void XINLINE set_dim(int dim, int value) {
        sizes_[dim] = value;
    }

    Dimension(const Dimension& other) : ndim_(other.ndim_) {
        for (int i = 0; i < ndim_;i++) {
            sizes_[i] = other.sizes_[i];
        }
    }

    Dimension& XINLINE operator=(const Dimension& other) {
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

enum DEVICE_T {
    DEVICE_T_CPU,
    DEVICE_T_GPU
};

std::ostream& operator<<(std::ostream& stream, const DEVICE_T& device) {
    return stream << ((device == DEVICE_T_CPU) ? "cpu" : "gpu");
}

template<typename Container, typename T>
struct ArrayLike {
    Dimension sizes_;
    T* ptr_;
    int offset_;
    const DEVICE_T device_;

    int XINLINE numel() const {
        return sizes_.numel();
    }

    int XINLINE offset() const {
        return offset_;
    }

    const Dimension& XINLINE size() const {
        return sizes_;
    }

    T& XINLINE operator[](const int& index) {
        return *(ptr_ + offset_ + index);
    }

    const T& XINLINE operator[](const int& index) const {
        return *(ptr_ + offset_ + index);
    }

    T& XINLINE operator[](const Dimension& indices) {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }

    const T& XINLINE operator[](const Dimension& indices) const {
        int idx_offset = Dimension::index2offset(this->sizes_, indices);
        return *(this->ptr_ + this->offset_ + idx_offset);
    }

    int XINLINE ndim() const {
        return sizes_.ndim();
    }

    ArrayLike(const Dimension& sizes, const DEVICE_T& dev, T* ptr, const int& offset)
        : sizes_(sizes),
          device_(dev),
          ptr_(ptr),
          offset_(offset) {}
};

template<typename T>
struct ArrayReference : public ArrayLike<ArrayReference<T>, T> {
    using ArrayLike<ArrayReference<T>, T>::ArrayLike;
};

template<typename Functor, typename T>
void __global__
unary_kernel(Functor func,
             const ArrayReference<T> arr,
             ArrayReference<T> dest) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = arr.numel();
    for (int i = idx; i < size; i += stride) {
        dest[i] = func(arr[i]);
    }
}

template<typename Functor, typename T>
void __global__
binary_kernel(Functor func,
              const ArrayReference<T> arr,
              const ArrayReference<T> arr2,
              ArrayReference<T> dest) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = arr.numel();
    for (int i = idx; i < size; i += stride) {
        dest[i] = func(arr[i], arr2[i]);
    }
}

template<typename Saver, typename SrcT, typename IndexT>
void scatter_saver_cpu(ArrayReference<SrcT> source,
                           const ArrayReference<IndexT> indices,
                           const ArrayReference<SrcT> updates) {
    int source_cols = source.size()[1];
    int size = indices.numel();
    for (int j = 0; j < size; ++j) {
        auto scatter_index = indices[j];
        for (int col_idx = 0; col_idx < source_cols; ++col_idx) {
            Saver::save(source[{scatter_index, col_idx}], updates[{j, col_idx}]);
        }
    }
}

template<typename Saver, typename SrcT, typename IndexT>
__global__
void scatter_saver_reduction_kernel(
        ArrayReference<SrcT> source,
        const ArrayReference<IndexT> indices,
        const ArrayReference<SrcT> updates) {

    int source_cols = source.size()[1];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = indices.numel();

    for (int col_idx = idx; col_idx < source_cols; col_idx += stride) {
        for (int j = 0; j < size; j++) {
            auto scatter_index = indices[j];
            Saver::save(source[{scatter_index, col_idx}], updates[{j, col_idx}]);
        }
    }
}

// __global__ scatter_saver_atomic(real_t* source, real_t* indices, real_t* updates) {

// }

template<typename Functor, typename Container, typename DestContainer>
void launch_unary_kernel(const Functor& func,
                         const Container& arr,
                         DestContainer dest) {
    const int NT = 128;
    const int max_blocks = 40960;
    // divide up by each embedding dimension (to avoid data-races in accumulation)
    int grid_size = div_ceil(arr.numel(), NT);
    grid_size = std::min(grid_size, max_blocks);

    auto view = arr.const_view();
    auto dest_view = dest.view();

    unary_kernel<<<grid_size, NT, 0, NULL>>>(
        func, view, dest_view
    );
}

template<typename Functor, typename Container1, typename Container2, typename DestContainer>
void launch_binary_kernel(const Functor& func,
                          const Container1& arr1,
                          const Container2& arr2,
                          DestContainer dest) {
    const int NT = 128;
    const int max_blocks = 40960;
    // divide up by each embedding dimension (to avoid data-races in accumulation)

    if (arr1.numel() != arr2.numel()) {
        throw std::runtime_error(
            make_message(
                "error: inputs to binary elementwise operation"
                " must have the same number of elements (got "
                "left.numel() = ", arr1.numel(),
                ", right.numel() = ", arr2.numel(), ")."
            )
        );
    }

    int grid_size = div_ceil(arr1.numel(), NT);
    grid_size = std::min(grid_size, max_blocks);

    auto view1 = arr1.const_view();
    auto view2 = arr2.const_view();
    auto dest_view = dest.view();

    binary_kernel<<<grid_size, NT, 0, NULL>>>(
        func, view1, view2, dest_view
    );
}

struct Device {
    static int gpu_device;

    static void set_gpu_device(int device) {
        if (gpu_device != device) {
            int count;
            cudaGetDeviceCount(&count);
            if (count == 0) {
                throw std::runtime_error(
                    "no gpu devices found"
                );
            } else {
                std::cout << count << " gpu device" << (count == 1 ? " " : "s ") << "found" << std::endl;
            }

            auto status = cudaSetDevice(device);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    make_message(
                        "could not set the gpu device to ",
                        device, ", reason = ", cudaGetErrorString(status)
                    )
                );
            }
            gpu_device = device;
        }
    }
};
int Device::gpu_device = -1;

template<typename T>
struct ArrayGather;

template<typename T>
struct Array : public ArrayLike<Array<T>, T> {
    std::shared_ptr<int> owners_;

    Array() : ArrayLike<Array<T>, T>({}, DEVICE_T_CPU, nullptr, 0),
              owners_(std::make_shared<int>(1)) {}

    void allocate_data() {
        if (this->device_ == DEVICE_T_CPU) {
            allocate_data_cpu();
        } else {
            allocate_data_gpu();
        }
    }

    void allocate_data_cpu() {
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

    void allocate_data_gpu() {
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

    Array(const Dimension& sizes, DEVICE_T dev)
            : ArrayLike<Array<T>, T>(sizes, dev, nullptr, 0),
              owners_(std::make_shared<int>(1)) {
        allocate_data();
    }

    Array<T> subtensor(int idx) const {
        auto arr = slice(idx, idx + 1);
        arr.sizes_ = arr.sizes_.subsize();
        return arr;
    }

    Array<T> slice(int start, int end) const {
        Array<T> arr = *this;
        int subvolume = 1;
        for (int i = 1; i < this->ndim(); i++) {
            subvolume *= this->sizes_[i];
        }
        arr.offset_ = this->offset_ + start * subvolume;
        arr.sizes_.set_dim(0, end - start);
        return arr;
    }

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

    ArrayGather<T> operator[](const Array<int>& indices);

    Array<T> to_device(DEVICE_T dev) const {
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

    Array(const Array<T>& other)
        : ArrayLike<Array<T>, T>(other.sizes_, other.device_, other.ptr_, other.offset_),
          owners_(other.owners_) {
        (*owners_) += 1;
    }

    void free_data() {
        if (this->device_ == DEVICE_T_CPU) {
            free_data_cpu();
        } else {
            free_data_gpu();
        }
    }

    void free_data_cpu() {
        free((void*)this->ptr_);
    }

    void free_data_gpu() {
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

    ~Array() {
        (*owners_) -= 1;
        if (this->ptr_ != nullptr && (*owners_) == 0) {
            free_data();
        }
    }

    void print() const {
        return print(0, false);
    }

    Array& operator=(const double& other);
    Array& operator+=(const double& other);
    Array& operator-=(const double& other);

    void print(int indent, bool print_comma) const {
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

    ArrayReference<T> view() {
        return ArrayReference<T>(
            this->sizes_,
            this->device_,
            this->ptr_,
            this->offset_
        );
    }

    const ArrayReference<T> const_view() const {
        return ArrayReference<T>(
            this->sizes_,
            this->device_,
            this->ptr_,
            this->offset_
        );
    }
};

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


template<typename Saver, typename SrcT, typename IndexT>
void scatter_saver(Array<SrcT> source,
                   const Array<IndexT>& indices,
                   const Array<SrcT>& updates,
                   cudaStream_t stream=NULL) {

    if (source.device_ == DEVICE_T_GPU) {
        const int NT = 128;
        const int max_blocks = 40960;
        // divide up by each embedding dimension (to avoid data-races in accumulation)
        int grid_size = div_ceil(source.size()[1], NT);
        grid_size = std::min(grid_size, max_blocks);

        auto source_view = source.view();
        auto indices_view = indices.const_view();
        auto updated_view = updates.const_view();

        scatter_saver_reduction_kernel<Saver><<<grid_size, NT, 0, stream>>>(
            source_view,
            indices_view,
            updated_view
        );
    } else {
        auto source_view = source.view();
        auto indices_view = indices.const_view();
        auto updated_view = updates.const_view();

        scatter_saver_cpu<Saver>(
            source_view,
            indices_view,
            updated_view
        );
    }
}

template<typename T>
struct ArrayGather {
    Array<T> array_;
    Array<int> indices_;

    ArrayGather(const Array<T>& array, const Array<int>& indices)
        : array_(array), indices_(indices) {}

    ArrayGather<T>& operator+=(const Array<T>& updates) {
        scatter_saver<saver::Increment>(array_, indices_, updates);
        return *this;
    }
    ArrayGather<T>& operator=(const Array<T>& updates) {
        scatter_saver<saver::Assign>(array_, indices_, updates);
        return *this;
    }
    ArrayGather<T>& operator-=(const Array<T>& updates) {
        scatter_saver<saver::Decrement>(array_, indices_, updates);
        return *this;
    }
};

template<typename T>
ArrayGather<T> Array<T>::operator[](const Array<int>& indices) {
    return ArrayGather<T>(*this, indices);
}

struct RunnableOp {
    CUmodule mod_;

    template<typename T>
    ArrayGather<T>& operator(ArrayGather<T> gather, const Array<T>& updates) {

        typedef (void (Array<T>, const Array<int>&, const Array<T>&, cudaStream_t))* func_t;

        func_t generated_functor;
        cuModuleGetFunction(generated_functor, &mod_, "scatter_with_functor");

        return gather;
    }
};

RunnableOp compile_and_run_scatter_op(std::string operator_symbol) {
    RunnableOp op;
    op.mod_ = cuModuleLoad("generated_code.o");
    return op;
}

int main() {
    int dim = 5;
    int cols = 3;

    Array<float> source({dim, cols}, DEVICE_T_GPU);
    source = 0;

    Array<float> updates({dim, cols}, DEVICE_T_GPU);
    updates = 2;

    Array<int> indices({dim}, DEVICE_T_CPU);
    int i = 0;
    for (auto index : {0, 0, 2, 1, 2}) {
        indices[i++] = index;
    }
    indices.print();
    auto indices_gpu = indices.to_device(DEVICE_T_GPU);
    source.print();
    // increment repeatedly at this location:
    source[indices_gpu] += updates;
    source.print();
    // decrement repeatedly at this location:
    source[indices_gpu] -= updates;
    source.print();
    // not well defined in many to one setup
    source[indices_gpu] = updates;
    source.print();

    auto op = compile_and_run_scatter_op("*");
    op(source[indices_gpu], updates);
}
