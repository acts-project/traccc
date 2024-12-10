/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

#define CUDA_ASSERT_TRUE(o, v) \
    do {                       \
        *out &= (!!v);         \
    } while (0)
#define CUDA_ASSERT_FALSE(o, v) \
    do {                        \
        *out &= (!v);           \
    } while (0)
#define CUDA_ASSERT_EQ(o, i, j) \
    do {                        \
        *out &= (i == j);       \
    } while (0)

#define CUDA_TEST(s, n, k)                                              \
    TEST(s, n) {                                                        \
        bool *b, out;                                                   \
        int* i;                                                         \
        VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&b, sizeof(bool)));          \
        VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&i, sizeof(int)));           \
        VECMEM_CUDA_ERROR_CHECK(cudaMemset(b, 1, sizeof(bool)));        \
        k<<<1, 1>>>(b, i);                                              \
        VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());                    \
        VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());               \
        VECMEM_CUDA_ERROR_CHECK(                                        \
            cudaMemcpy(&out, b, sizeof(bool), cudaMemcpyDeviceToHost)); \
        ASSERT_TRUE(out);                                               \
        VECMEM_CUDA_ERROR_CHECK(cudaFree(b));                           \
        VECMEM_CUDA_ERROR_CHECK(cudaFree(i));                           \
    }

__global__ void atomic_ref_operator_equals_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    a = 5;
    CUDA_ASSERT_EQ(out, a.load(), 5);
}

CUDA_TEST(cuda_device_atomic_test, atomic_operator_equals,
          atomic_ref_operator_equals_kernel);

__global__ void atomic_ref_load_kernel(bool* out, int* i) {
    *i = 17;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 17);
}

CUDA_TEST(cuda_device_atomic_test, atomic_load, atomic_ref_load_kernel);

__global__ void atomic_ref_compare_exchange_strong_kernel(bool* out, int* i) {
    *i = 0;
    int zero = 0, five = 5;
    vecmem::device_atomic_ref<int> a(*i);
    // This passes, as i == 0, zero == 0. This will set zero = i = 0 and i = 5;
    CUDA_ASSERT_TRUE(out, a.compare_exchange_strong(zero, 5));
    CUDA_ASSERT_EQ(out, zero, 0);
    // This fails, as i == 5, zero == 0. This will set zero = i = 5;
    CUDA_ASSERT_FALSE(out, a.compare_exchange_strong(zero, 5));
    CUDA_ASSERT_EQ(out, zero, 5);
    // This succeeds, as i == 5, five == 5. This will set five = i = 5 and i =
    // 0;
    CUDA_ASSERT_TRUE(out, a.compare_exchange_strong(five, 0));
    CUDA_ASSERT_EQ(out, five, 5);
    // This fails, as i == 0, five == 5. This will set five = i = 0;
    CUDA_ASSERT_FALSE(out, a.compare_exchange_strong(five, 0));
    CUDA_ASSERT_EQ(out, five, 0);
}

CUDA_TEST(cuda_device_atomic_test, atomic_compare_exchange_strong,
          atomic_ref_compare_exchange_strong_kernel);

__global__ void atomic_ref_store_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0);
    a.store(5);
    CUDA_ASSERT_EQ(out, a.load(), 5);
}

CUDA_TEST(core_device_atomic_test, atomic_store, atomic_ref_store_kernel);

__global__ void atomic_ref_exchange_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0);
    CUDA_ASSERT_EQ(out, a.exchange(5), 0);
    CUDA_ASSERT_EQ(out, a.load(), 5);
    CUDA_ASSERT_EQ(out, a.exchange(3), 5);
    CUDA_ASSERT_EQ(out, a.load(), 3);
}

CUDA_TEST(core_device_atomic_test, atomic_exchange, atomic_ref_exchange_kernel);

__global__ void atomic_ref_fetch_add_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0);
    CUDA_ASSERT_EQ(out, a.fetch_add(5), 0);
    CUDA_ASSERT_EQ(out, a.load(), 5);
    CUDA_ASSERT_EQ(out, a.fetch_add(3), 5);
    CUDA_ASSERT_EQ(out, a.load(), 8);
}

CUDA_TEST(core_device_atomic_test, atomic_fetch_add,
          atomic_ref_fetch_add_kernel);

__global__ void atomic_ref_fetch_sub_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0);
    CUDA_ASSERT_EQ(out, a.fetch_sub(5), 0);
    CUDA_ASSERT_EQ(out, a.load(), -5);
    CUDA_ASSERT_EQ(out, a.fetch_sub(3), -5);
    CUDA_ASSERT_EQ(out, a.load(), -8);
}

CUDA_TEST(core_device_atomic_test, atomic_fetch_sub,
          atomic_ref_fetch_sub_kernel);

__global__ void atomic_ref_fetch_and_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0b0101);
    CUDA_ASSERT_EQ(out, a.fetch_and(0b1100), 0b0101);
    CUDA_ASSERT_EQ(out, a.load(), 0b0100);
    CUDA_ASSERT_EQ(out, a.fetch_and(0b0000), 0b0100);
    CUDA_ASSERT_EQ(out, a.load(), 0b0000);
}

CUDA_TEST(core_device_atomic_test, atomic_fetch_and,
          atomic_ref_fetch_and_kernel);

__global__ void atomic_ref_fetch_or_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0b0101);
    CUDA_ASSERT_EQ(out, a.fetch_or(0b1100), 0b0101);
    CUDA_ASSERT_EQ(out, a.load(), 0b1101);
    CUDA_ASSERT_EQ(out, a.fetch_or(0b0000), 0b1101);
    CUDA_ASSERT_EQ(out, a.load(), 0b1101);
}

CUDA_TEST(core_device_atomic_test, atomic_fetch_or, atomic_ref_fetch_or_kernel);

__global__ void atomic_ref_fetch_xor_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    CUDA_ASSERT_EQ(out, a.load(), 0b0101);
    CUDA_ASSERT_EQ(out, a.fetch_xor(0b1100), 0b0101);
    CUDA_ASSERT_EQ(out, a.load(), 0b1001);
    CUDA_ASSERT_EQ(out, a.fetch_xor(0b0000), 0b1001);
    CUDA_ASSERT_EQ(out, a.load(), 0b1001);
}

CUDA_TEST(core_device_atomic_test, atomic_fetch_xor,
          atomic_ref_fetch_xor_kernel);
