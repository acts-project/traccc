/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "detray/definitions/detail/cuda_definitions.hpp"

// Detray test include(s)
#include "container_cuda_kernel.hpp"

namespace detray {

__global__ void test_single_store_kernel(
    single_store_t::view_type store_view,
    vecmem::data::vector_view<double> sum_data) {

    single_store_dev_t store(store_view);
    vecmem::device_vector<double> sum(sum_data);

    for (const double e : store) {
        sum[0] += e;
    }
}

__global__ void test_tuple_cont_kernel(
    typename tuple_cont_t::view_type store_view,
    vecmem::data::vector_view<double> sum_data) {

    tuple_cont_dev_t store(store_view);
    vecmem::device_vector<double> sum(sum_data);

    const auto& vec0 = store.get<0>();
    const auto& vec1 = store.get<1>();

    for (const auto e : vec0) {
        sum[0] += e;
    }
    for (const auto e : vec1) {
        sum[0] += e;
    }
}

__global__ void test_reg_multi_store_kernel(
    reg_multi_store_t::view_type store_view,
    vecmem::data::vector_view<double> sum_data) {

    reg_multi_store_dev_t store(store_view);
    vecmem::device_vector<double> sum(sum_data);

    const auto& vec0 = store.get<0>();
    const auto& vec1 = store.get<1>();
    const auto& vec2 = store.get<2>();

    for (const auto e : vec0) {
        sum[0] += e;
    }
    for (const auto e : vec1) {
        sum[0] += e;
    }
    for (const auto e : vec2) {
        sum[0] += e;
    }
}

__global__ void test_multi_store_kernel(
    multi_store_t::view_type store_view,
    vecmem::data::vector_view<double> sum_data) {

    multi_store_dev_t store(store_view);
    vecmem::device_vector<double> sum(sum_data);

    const auto& vec0 = store.get<0>();
    const auto& vec1 = store.get<1>().first;
    const auto& vec2 = store.get<1>().second;

    for (const auto e : vec0) {
        sum[0] += e;
    }
    for (const auto e : vec1) {
        sum[0] += e;
    }
    for (const auto e : vec2) {
        sum[0] += e;
    }
}

void test_single_store(single_store_t::view_type store_view,
                       vecmem::data::vector_view<double> sum_data) {

    // run the test kernel
    test_single_store_kernel<<<1, 1>>>(store_view, sum_data);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void test_tuple_container(tuple_cont_t::view_type store_view,
                          vecmem::data::vector_view<double> sum_data) {

    // run the test kernel
    test_tuple_cont_kernel<<<1, 1>>>(store_view, sum_data);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void test_reg_multi_store(reg_multi_store_t::view_type store_view,
                          vecmem::data::vector_view<double> sum_data) {

    // run the test kernel
    test_reg_multi_store_kernel<<<1, 1>>>(store_view, sum_data);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void test_multi_store(typename multi_store_t::view_type store_view,
                      vecmem::data::vector_view<double> sum_data) {

    // run the test kernel
    test_multi_store_kernel<<<1, 1>>>(store_view, sum_data);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

}  // namespace detray
