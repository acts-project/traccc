/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */


// Spacepoint formation include(s).
#include "spacepoint_formation.hpp"



namespace traccc::cuda {
namespace kernel{
__global__ void spacepoint_formation(spacepoint_container_types::view spacepoints_view,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view){
    
    // Get the global idx
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize device containers
    measurement_container_types::const_device
        measurements_device(measurements_view);
    spacepoint_container_types::device spacepoints_device(
        spacepoints_view);
    vecmem::device_vector<const device::prefix_sum_element_t>
        measurements_prefix_sum(measurements_prefix_sum_view);

    // Ignore if idx is out of range
    if (idx >= measurements_prefix_sum.size())
        return;

    // Get the indices from the prefix sum vector
    const auto module_idx = measurements_prefix_sum[idx].first;
    const auto measurement_idx =
        measurements_prefix_sum[idx].second;

    // Get the measurement for this idx
    const auto& m = measurements_device[module_idx].items.at(
        measurement_idx);

    // Get the current cell module
    const auto& module = measurements_device[module_idx].header;

    // Form a spacepoint based on this measurement
    point3 local_3d = {m.local[0], m.local[1], 0.};
    point3 global = module.placement.point_to_global(local_3d);
    spacepoint s({global, m});

    // Push the speacpoint into the container at the appropriate
    // module idx
    spacepoints_device[module_idx].header = module.module;
    spacepoints_device[module_idx].items.push_back(s);

}
}// namespace kernel

void spacepoint_formation(
    spacepoint_container_types::view spacepoints_view,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view) {

    // The execution range of the kernel
    auto n_measurements = measurements_prefix_sum_view.size();

    // Calculate the execution NDrange for the kernel
    auto nSpaceptFormThreads = 64;
    auto nSpaceptFormBlocks = (n_measurements + nSpaceptFormThreads - 1) / nSpaceptFormThreads;
    kernel::spacepoint_formation<<<nSpaceptFormBlocks,nSpaceptFormThreads>>>(
        spacepoints_view, measurements_view, measurements_prefix_sum_view
    );
    cudaDeviceSynchronize();  


}



}  // namespace traccc::cuda