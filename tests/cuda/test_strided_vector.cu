/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <vecmem/memory/cuda/device_memory_resource.hpp>

#include "traccc/device/strided_vector.hpp"

struct MyStruct {
    int a;
    bool b;
    float c;
    unsigned long d;
    float e[4];

    bool operator==(const MyStruct &) const = default;
};

template <typename T>
struct InitHelper {};

template <>
struct InitHelper<char> {
    TRACCC_HOST_DEVICE static char init(unsigned int i) {
        return static_cast<char>(i % 211);
    }
};

template <>
struct InitHelper<short> {
    TRACCC_HOST_DEVICE static short init(unsigned int i) {
        return static_cast<short>(
            (((i % 2 == 0) ? static_cast<short>(-1) : static_cast<short>(1)) *
             (991 * i)) %
            52119);
    }
};

template <>
struct InitHelper<unsigned int> {
    TRACCC_HOST_DEVICE static unsigned int init(unsigned int i) {
        return static_cast<unsigned int>((5019291 * i) % 12452119);
    }
};

template <>
struct InitHelper<long> {
    TRACCC_HOST_DEVICE static long init(unsigned int i) {
        return static_cast<long>((((i % 2 == 0) ? -1L : 1L) * (991215 * i)) %
                                 52119215);
    }
};

template <>
struct InitHelper<MyStruct> {
    TRACCC_HOST_DEVICE static MyStruct init(unsigned int i) {
        return {
            static_cast<int>(i * 4),
            i % 2 == 0,
            static_cast<float>(i) * 2.3f,
            i * 410,
            {static_cast<float>(i), 6.0f, static_cast<float>(i) * 4.4f, 19.4f}};
    }
};

template <>
struct InitHelper<std::array<char, 7>> {
    TRACCC_HOST_DEVICE static std::array<char, 7> init(unsigned int i) {
        return {
            static_cast<char>(i % 21),       static_cast<char>((2 * i) % 61),
            static_cast<char>((3 * i) % 93), static_cast<char>((4 * i) % 151),
            static_cast<char>((5 * i) % 51), static_cast<char>((6 * i) % 101),
            static_cast<char>((7 * i) % 9)};
    }
};

template <>
struct InitHelper<std::array<short, 5>> {
    TRACCC_HOST_DEVICE static std::array<short, 5> init(unsigned int i) {
        return {static_cast<char>(i % 21), static_cast<char>((2 * i) % 61),
                static_cast<char>((3 * i) % 93),
                static_cast<char>((4 * i) % 151),
                static_cast<char>((7 * i) % 9)};
    }
};

template <typename T, std::size_t MaxStride>
__global__ void store_kernel_ptr(void *out, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        traccc::device::strided_vector_device<T, MaxStride> acc(out, n);

        acc.store(InitHelper<T>::init(tid), tid);
    }
}

template <typename T, std::size_t MaxStride>
__global__ void load_kernel_ptr(void *in, T *out, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        traccc::device::strided_vector_device<T, MaxStride> acc(in, n);

        out[tid] = acc.load(tid);
    }
}

template <typename T, std::size_t MaxStride>
__global__ void store_kernel_view(
    traccc::device::strided_vector_view<T, MaxStride> v, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        traccc::device::strided_vector_device<T, MaxStride> acc(v);

        acc.store(InitHelper<T>::init(tid), tid);
    }
}

template <typename T, std::size_t MaxStride>
__global__ void load_kernel_view(
    traccc::device::strided_vector_view<T, MaxStride> v, T *out,
    unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        traccc::device::strided_vector_device<T, MaxStride> acc(v);

        out[tid] = acc.load(tid);
    }
}

template <typename T>
class CUDAStridedVector : public testing::Test {};

template <typename T, std::size_t MaxStride>
struct Config {
    using type = T;
    static constexpr std::size_t stride = MaxStride;
};

using TestTypes = ::testing::Types<
    Config<char, 1>, Config<char, 2>, Config<char, 4>, Config<char, 8>,
    Config<short, 1>, Config<short, 2>, Config<short, 4>, Config<short, 8>,
    Config<unsigned int, 1>, Config<unsigned int, 2>, Config<unsigned int, 4>,
    Config<unsigned int, 8>, Config<long, 1>, Config<long, 2>, Config<long, 4>,
    Config<long, 8>, Config<MyStruct, 1>, Config<MyStruct, 2>,
    Config<MyStruct, 4>, Config<MyStruct, 8>, Config<std::array<char, 7>, 1>,
    Config<std::array<char, 7>, 2>, Config<std::array<char, 7>, 4>,
    Config<std::array<char, 7>, 8>, Config<std::array<short, 5>, 1>,
    Config<std::array<short, 5>, 2>, Config<std::array<short, 5>, 4>,
    Config<std::array<short, 5>, 8>>;
TYPED_TEST_SUITE(CUDAStridedVector, TestTypes);

static_assert(sizeof(char) == 1);
static_assert(sizeof(short) == 2);
static_assert(sizeof(unsigned int) == 4);
static_assert(sizeof(long) == 8);
static_assert(sizeof(MyStruct) == 40);
static_assert(sizeof(std::array<char, 7>) == 7);
static_assert(sizeof(std::array<short, 5>) == 10);

TYPED_TEST(CUDAStridedVector, FromPointer1024) {
    unsigned int n = 1024;

    void *tmp;
    ASSERT_EQ(cudaMallocManaged(&tmp, n * sizeof(typename TypeParam::type)),
              cudaSuccess);

    typename TypeParam::type *out;
    cudaMallocManaged(&out, n * sizeof(typename TypeParam::type));

    unsigned int nThreads = 256;
    unsigned int nBlocks = (n + nThreads - 1) / nThreads;

    store_kernel_ptr<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    load_kernel_ptr<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, out, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (unsigned int i = 0; i < n; ++i) {
        EXPECT_EQ(InitHelper<typename TypeParam::type>::init(i), out[i])
            << " index " << i;
    }
}

TYPED_TEST(CUDAStridedVector, FromPointer531) {
    unsigned int n = 531;

    void *tmp;
    ASSERT_EQ(cudaMallocManaged(&tmp, n * sizeof(typename TypeParam::type)),
              cudaSuccess);

    typename TypeParam::type *out;
    cudaMallocManaged(&out, n * sizeof(typename TypeParam::type));

    unsigned int nThreads = 256;
    unsigned int nBlocks = (n + nThreads - 1) / nThreads;

    store_kernel_ptr<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    load_kernel_ptr<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, out, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (unsigned int i = 0; i < n; ++i) {
        EXPECT_EQ(InitHelper<typename TypeParam::type>::init(i), out[i])
            << " index " << i;
    }
}

TYPED_TEST(CUDAStridedVector, FromVecmem1024) {
    unsigned int n = 1024;

    vecmem::cuda::device_memory_resource device_resource;

    vecmem::data::vector_buffer<typename TypeParam::type> tmp_buff(
        n, device_resource);
    traccc::device::strided_vector_buffer<typename TypeParam::type,
                                          TypeParam::stride>
        tmp(std::move(tmp_buff));

    typename TypeParam::type *out;
    cudaMallocManaged(&out, n * sizeof(typename TypeParam::type));

    unsigned int nThreads = 256;
    unsigned int nBlocks = (n + nThreads - 1) / nThreads;

    store_kernel_view<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    load_kernel_view<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, out, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (unsigned int i = 0; i < n; ++i) {
        EXPECT_EQ(InitHelper<typename TypeParam::type>::init(i), out[i])
            << " index " << i;
    }
}

TYPED_TEST(CUDAStridedVector, FromVecmem531) {
    unsigned int n = 531;

    vecmem::cuda::device_memory_resource device_resource;

    vecmem::data::vector_buffer<typename TypeParam::type> tmp_buff(
        n, device_resource);
    traccc::device::strided_vector_buffer<typename TypeParam::type,
                                          TypeParam::stride>
        tmp(std::move(tmp_buff));

    typename TypeParam::type *out;
    cudaMallocManaged(&out, n * sizeof(typename TypeParam::type));

    unsigned int nThreads = 256;
    unsigned int nBlocks = (n + nThreads - 1) / nThreads;

    store_kernel_view<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    load_kernel_view<typename TypeParam::type, TypeParam::stride>
        <<<nBlocks, nThreads>>>(tmp, out, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (unsigned int i = 0; i < n; ++i) {
        EXPECT_EQ(InitHelper<typename TypeParam::type>::init(i), out[i])
            << " index " << i;
    }
}
