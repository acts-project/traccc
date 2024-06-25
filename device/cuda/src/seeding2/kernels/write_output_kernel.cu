/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cooperative_groups.h>

#include <cstdint>
#include <traccc/cuda/seeding2/kernels/write_output_kernel.hpp>
#include <traccc/cuda/seeding2/types/internal_sp.hpp>
#include <traccc/cuda/utils/definitions.hpp>
#include <traccc/edm/alt_seed.hpp>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc::cuda {
/**
 * @brief Main output writing kernel.
 *
 * @param[in] n_seeds The number of seeds to write.
 * @param[in] spacepoints Spacepoint array into which seeds are indexed.
 * @param[in] in_seeds Input internal seeds to read from.
 * @param[out] out_seeds Output seeds to write to.
 */
__global__ void write_output_kernel(uint32_t n_seeds,
                                    const internal_sp_t spacepoints,
                                    const alt_seed* const in_seeds,
                                    alt_seed_collection_types::view out_seeds) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    /*
     * Create a writable device vector for the seeds.
     */
    vecmem::device_vector<alt_seed> dvec(out_seeds);

    int i = grid.thread_rank();

    if (i < n_seeds) {
        /*
         * Get the output seed for the current thread.
         */
        dvec.at(i) = in_seeds[i];
    }
}

alt_seed_collection_types::buffer write_output(
    const traccc::memory_resource& mr, uint32_t n_seeds,
    const internal_sp_t spacepoints, const alt_seed* const seeds) {
    static constexpr std::size_t threads_per_block = 256;

    /*
     * Create a "vector buffer" that we can write our output to.
     */
    alt_seed_collection_types::buffer out(n_seeds, mr.main);

    /*
     * Fetch the current CUDA device number.
     */
    int device;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));

    /*
     * Execute the main output kernel.
     */
    write_output_kernel<<<(n_seeds / threads_per_block) +
                              (n_seeds % threads_per_block == 0 ? 0 : 1),
                          threads_per_block>>>(n_seeds, spacepoints, seeds,
                                               out);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return out;
}
}  // namespace traccc::cuda
