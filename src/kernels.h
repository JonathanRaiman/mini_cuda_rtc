#ifndef RTC_KERNELS_H
#define RTC_KERNELS_H

#include "array.h"
#include "array_reference.h"
#include "make_message.h"

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

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

#endif
