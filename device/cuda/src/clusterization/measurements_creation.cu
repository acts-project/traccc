/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "measurements_creation.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {

namespace kernel{
__global__ void measurement_creation(cluster_container_types::const_view clusters_view,
                        measurement_container_types::view measurements_view,
                        const cell_container_types::const_view& cells_view)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize device vectors
    const cluster_container_types::const_device clusters_device(
        clusters_view);
    measurement_container_types::device measurements_device(
        measurements_view);
    cell_container_types::const_device cells_device(cells_view);

    // Ignore if idx is out of range
    if (idx >= clusters_device.size())
        return;

    // items: cluster of cells at current idx
    // header: cluster_id object with the information about the
    // cell module
    const auto& cluster = clusters_device[idx].items;
    const auto& module_link = clusters_device[idx].header;
    const auto& module = cells_device.at(module_link).header;

    // Should not happen
    assert(cluster.empty() == false);

    // Fill measurement from cluster
    detail::fill_measurement(measurements_device, cluster,
                                module, module_link, idx);
}
}
void measurement_creation(measurement_container_types::view measurements_view,
                          cluster_container_types::const_view clusters_view,
                          const cell_container_types::const_view& cells_view)
{

    // The kernel execution range
    auto n_clusters = clusters_view.headers.size();
    // Calculate the execution NDrange for the kernel
    auto nMeasurementCreationThreads = 64;
    auto nMeasurementCreationBlocks = (n_clusters + nMeasurementCreationThreads - 1)
                                        / nMeasurementCreationThreads;
    printf("n_clusters:%d\n",n_clusters);                                        
    // Run the kernel
    kernel::measurement_creation<<<nMeasurementCreationBlocks,nMeasurementCreationThreads>>>(
        clusters_view,measurements_view,cells_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    printf("measurements view headers size %d\n",measurements_view.headers.size());
}

}  // namespace traccc::cuda