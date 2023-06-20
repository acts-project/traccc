#include "traccc/cuda/seed_merging/seed_merging.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/edm/nseed.hpp"
#include "traccc/cuda/utils/definitions.hpp"

#include <iostream>

namespace traccc::cuda {
namespace kernels {
template<std::size_t N>
__global__ void convert_to_nseeds(seed_collection_types::view vf, nseed<N> * out, unsigned long long * out_n) {
    seed_collection_types::device vfd(vf);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vf.size(); i += gridDim.x * blockDim.x) {
        out[i] = vfd[i];
        atomicAdd(out_n, 1ULL);
    }
}

template<std::size_t N, std::size_t M>
__global__ void merge_nseeds(const nseed<N> * in, const unsigned long long * in_c, nseed<M> * out, unsigned long long * out_c) {
    __shared__ nseed<M> out_seeds[32];
    __shared__ uint32_t num_seeds;
    __shared__ uint32_t num_consm;
    __shared__ uint32_t out_index;

    if (threadIdx.x == 0) {
        num_seeds = 0;
        num_consm = 0;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < *in_c; i += blockDim.x) {
        if (i == blockIdx.x) {
            continue;
        }

        bool compat, consumed;

        if (in[blockIdx.x].size() == in[i].size()) {
            compat = true;
            consumed = true;

            for (int j = 0; j < in[i].size() - 1; ++j) {
                if (in[blockIdx.x]._sps[j+1] != in[i]._sps[j]) {
                    compat = false;
                }

                if (in[i]._sps[j+1] != in[blockIdx.x]._sps[j]) {
                    consumed = false;
                }
            }
        } else {
            if (in[i].size() > in[blockIdx.x].size()) {
                consumed = false;

                for (int j = 0; j < in[i].size(); ++j) {
                    for (int k = 0; k < in[blockIdx.x].size(); ++k) {
                        if (in[i]._sps[j] == in[blockIdx.x]._sps[k]) {
                            consumed = true;
                        }
                    }
                }
            } else {
                consumed = false;
            }

            compat = false;
        }

        if (compat) {
            nseed<M> new_seed;

            new_seed._size = in[blockIdx.x]._size + 1;

            int j = 0;

            for (; j < in[blockIdx.x].size(); ++j) {
                new_seed._sps[j] = in[blockIdx.x]._sps[j];
            }

            new_seed._sps[j] = in[i]._sps[in[i].size() - 1];

            uint32_t idx = atomicAdd(&num_seeds, 1);

            if (idx < 32) {
                out_seeds[idx] = new_seed;
            }
        }

        if (consumed) {
            atomicAdd(&num_consm, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        if (num_seeds == 0 && num_consm == 0) {
            out_index = atomicAdd(out_c, 1U);
            out[out_index] = in[blockIdx.x];
        } else {
            out_index = atomicAdd(out_c, num_seeds);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_seeds; i += blockDim.x) {
        out[out_index + i] = out_seeds[i];
    }
}
}

seed_merging::seed_merging(const traccc::memory_resource& mr, stream& str) : m_mr(mr), m_stream(str) {
}

seed_merging::output_type seed_merging::operator()(const seed_collection_types::buffer&i) const {
    vecmem::unique_alloc_ptr<nseed<20>[]>
        arr1 = vecmem::make_unique_alloc<nseed<20>[]>(m_mr.main, 1000000),
        arr2 = vecmem::make_unique_alloc<nseed<20>[]>(m_mr.main, 1000000);

    vecmem::unique_alloc_ptr<unsigned long long>
        siz1 = vecmem::make_unique_alloc<unsigned long long>(m_mr.main),
        siz2 = vecmem::make_unique_alloc<unsigned long long>(m_mr.main);

    kernels::convert_to_nseeds<20><<<2048, 256>>>(i, arr1.get(), siz1.get());

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    unsigned long long rc;

    CUDA_ERROR_CHECK(cudaMemcpy(&rc, siz1.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::cout << "Step 0 has " << rc << " seeds." << std::endl;

    for (std::size_t i = 0; i < 5; ++i) {
        CUDA_ERROR_CHECK(cudaMemset(siz2.get(), 0, sizeof(unsigned long long)));
        kernels::merge_nseeds<20, 20><<<rc, 256>>>(arr1.get(), siz1.get(), arr2.get(), siz2.get());

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        std::swap(arr1, arr2);
        std::swap(siz1, siz2);

        CUDA_ERROR_CHECK(cudaMemcpy(&rc, siz1.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        std::cout << "Step " << (i + 1) << " has " << rc << " seeds." << std::endl;
    }


    return {std::move(arr1), rc};
}
}
