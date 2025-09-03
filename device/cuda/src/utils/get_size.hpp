/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../utils/cuda_error_handling.hpp"

namespace traccc::cuda {

namespace detail {
template <typename T>
concept container_has_size_ptr = requires(const T& t) {
    { t.size_ptr() };
};

template <typename T>
concept container_has_size = requires(const T& t) {
    { t.size().ptr() };
};
}  // namespace detail

/**
 * @brief Helper function to efficiently get the size of a vecmem container,
 * as the default vecmem copy logic goes through pageable memory which can
 * greatly slow down the application.
 */
template <typename T>
typename T::size_type get_size(const T& data, typename T::size_type* staging,
                               cudaStream_t& stream) {
    static_assert(detail::container_has_size_ptr<T> ||
                  detail::container_has_size<T>);

    if constexpr (detail::container_has_size_ptr<T>) {
        if (data.size_ptr() == nullptr) {
            return data.capacity();
        } else {
            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                staging, data.size_ptr(), sizeof(typename T::size_type),
                cudaMemcpyDeviceToHost, stream));
            TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            return *staging;
        }
    } else {
        if (data.size().ptr() == nullptr) {
            return data.capacity();
        } else {
            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                staging, data.size().ptr(), sizeof(typename T::size_type),
                cudaMemcpyDeviceToHost, stream));
            TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            return *staging;
        }
    }
}

}  // namespace traccc::cuda
