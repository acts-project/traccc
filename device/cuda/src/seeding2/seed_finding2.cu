/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <limits>
#include <memory>
#include <traccc/cuda/seeding2/kernels/kd_tree_kernel.hpp>
#include <traccc/cuda/seeding2/kernels/seed_finding_kernel.hpp>
#include <traccc/cuda/seeding2/kernels/write_output_kernel.hpp>
#include <traccc/cuda/seeding2/seed_finding.hpp>
#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/cuda/utils/definitions.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/lin_circle.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#include <vector>

namespace {
/// Helper function that would produce a default seed-finder configuration
__host__ traccc::seedfinder_config default_seedfinder2_config() {
    traccc::seedfinder_config config;
    traccc::seedfinder_config config_copy = config.toInternalUnits();
    config.highland = 13.6f * std::sqrt(config_copy.radLengthPerSeed) *
                      (1.f + 0.038f * std::log(config_copy.radLengthPerSeed));
    float maxScatteringAngle = config.highland / config_copy.minPt;
    config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
    // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
    // and millimeter
    config.pTPerHelixRadius = 300.f * config_copy.bFieldInZ;
    config.minHelixDiameter2 =
        std::pow(config_copy.minPt * 2.f / config.pTPerHelixRadius, 2.f);
    config.pT2perRadius =
        std::pow(config.highland / config.pTPerHelixRadius, 2.f);
    return config;
}
}  // namespace

namespace traccc::cuda {
__global__ void construct_internal_sp_kernel(internal_sp_t out,
                                             const spacepoint* in,
                                             std::size_t n_sp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_sp) {
        out[tid].x = in[tid].x();
        out[tid].y = in[tid].y();
        out[tid].z = in[tid].z();
        out[tid].phi = atan2f(in[tid].y(), in[tid].x());
        out[tid].radius = in[tid].radius();
        out[tid].link = tid;
    }
}

seed_finding2::seed_finding2(const traccc::memory_resource& mr)
    : m_output_mr(mr), m_finder_conf(default_seedfinder2_config()) {}

seed_finding2::output_type seed_finding2::operator()(
    const spacepoint_collection_types::const_view& sps) const {
    vecmem::cuda::copy copy;
    vecmem::cuda::device_memory_resource mr;
    std::size_t n_sp = copy.get_size(sps);

    internal_sp_owning_t internal_sp_device(mr, n_sp);

    static constexpr std::size_t copy_kernel_threads_per_block = 256;
    construct_internal_sp_kernel<<<
        (n_sp / copy_kernel_threads_per_block) +
            (n_sp % copy_kernel_threads_per_block == 0 ? 0 : 1),
        copy_kernel_threads_per_block>>>(internal_sp_t(internal_sp_device),
                                         sps.ptr(), n_sp);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    auto [kd_tree_device, kd_tree_size, internal_sp_device_new] =
        create_kd_tree(mr, std::move(internal_sp_device), n_sp);

    vecmem::unique_alloc_ptr<alt_seed[]> seeds_device;
    uint32_t seed_count;

    std::tie(seeds_device, seed_count) = run_seeding(
        m_finder_conf, m_filter_conf, mr, internal_sp_t(internal_sp_device_new),
        kd_tree_t(kd_tree_device));

    return write_output(m_output_mr, seed_count,
                        internal_sp_t(internal_sp_device_new),
                        seeds_device.get());
}
}  // namespace traccc::cuda
