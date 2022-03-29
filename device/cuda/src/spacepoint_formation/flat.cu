/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <vecmem/memory/unique_ptr.hpp>

#include "traccc/cuda/geometry/module_map.hpp"
#include "traccc/cuda/spacepoint_formation/flat.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::cuda {
namespace {
/*
 * Declare a product type containing geometry identifiers and a single
 * measurement matching that identifier.
 */
typedef struct {
    geometry_id geom_id;
    measurement meas;
} pair_t;

/**
 * @brief The main GPU kernel for our algorithm.
 *
 * @param[in] mm The module map view to use for lookups.
 * @param[in] input The input array of surface-measurement pairs.
 * @param[out] output The array to write our spacepoints to.
 */
__global__ void spacepoint_formation_flat_kernel(
    module_map_view<geometry_id, transform3> mm, pair_t* input,
    spacepoint* output, std::size_t n) {
    /*
     * Calculate the index of the current thread in the grid, which corresponds
     * to the index of the measurement we are transforming.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Bounds check to ensure we do not execute any code for halo threads which
     * could cause illegal memory accesses.
     */
    if (n < tid) {
        /*
         * Retrieve the transformation that matches the given geometry
         * identifier.
         */
        const transform3& t = *mm[input[n].geom_id];
        const measurement& m = input[n].meas;

        /*
         * Compute the global position from our local position.
         */
        point3 global = t.point_to_global({m.local[0], m.local[1], 0.});

        /*
         * Write the output spacepoint to memory.
         */
        output[n] = {global, {0, 0, 0}, m};
    }
}
}  // namespace

spacepoint_formation_flat::spacepoint_formation_flat(
    vecmem::memory_resource& mr, const module_map<geometry_id, transform3>& mm)
    : m_mr(mr), m_mm(mm) {}

host_spacepoint_collection spacepoint_formation_flat::operator()(
    const host_measurement_container& m) const {
    /*
     * Define the number of threads per CUDA block. In this particular case we
     * use so few registers or shared memory that there is unlikely to be any
     * occupancy issues, so the number of threads most likely doesn't matter
     * too much.
     **/
    static constexpr std::size_t THREADS_PER_BLOCK = 1024;

    /*
     * Declare vectors for the input and output of the algorithm.
     */
    vecmem::vector<pair_t> input_vec(&m_mr);
    vecmem::vector<spacepoint> output_vec(&m_mr);

    /*
     * Loop over the jagged array of measurements, with the different surfaces
     * at the top level.
     */
    for (std::size_t i = 0; i < m.size(); ++i) {
        geometry_id gid = m.at(i).header.module;

        /*
         * Loop over the individual measurements belonging to a given surface
         * and collect them in the input vector, together with the geometry ID.
         */
        for (std::size_t j = 0; j < m.at(i).items.size(); ++j) {
            input_vec.push_back({gid, m.at(i).items.at(j)});
        }
    }

    /*
     * Ensure that there is sufficient space to hold all of our output
     * spacepoints.
     */
    output_vec.reserve(input_vec.size());

    /*
     * Call the kernel.
     */
    spacepoint_formation_flat_kernel<<<
        (THREADS_PER_BLOCK / 1024) + (THREADS_PER_BLOCK % 1024 == 0 ? 0 : 1),
        THREADS_PER_BLOCK>>>(m_mm, input_vec.data(), output_vec.data(),
                             input_vec.size());

    /*
     * Ensure that no launch issues occurred, then make sure the kernel is
     * finished before we return.
     */
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return output_vec;
}
}  // namespace traccc::cuda
