/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <vecmem/memory/cuda/device_memory_resource.hpp>

#include "traccc/cuda/geometry/module_map.hpp"
#include "traccc/geometry/module_map.hpp"
#include "traccc/io/reader.hpp"

TEST(DeviceModuleMap, ViewSize) {
    ASSERT_LE(sizeof(traccc::cuda::module_map_view<traccc::geometry_id,
                                                   traccc::transform3>),
              64);
}

__global__ void readMapKernel(
    traccc::cuda::module_map_view<traccc::geometry_id, traccc::transform3> m,
    traccc::geometry_id i, traccc::transform3 *o) {
    *o = *(m[i]);
}

TEST(DeviceModuleMap, TrackML) {
    vecmem::cuda::device_memory_resource mr;

    std::string file = traccc::data_directory() +
                       std::string("tml_detector/trackml-detector.csv");

    traccc::surface_reader sreader(
        file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw",
               "rot_zu", "rot_zv", "rot_zw"});
    std::map<traccc::geometry_id, traccc::transform3> inp =
        traccc::read_surfaces(sreader);

    traccc::module_map tmp_map(inp);

    traccc::cuda::module_map map(tmp_map, mr);

    traccc::transform3 *ptr;

    ASSERT_EQ(
        cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(traccc::transform3)),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (const std::pair<const traccc::geometry_id, traccc::transform3> &i :
         inp) {
        traccc::transform3 res;

        readMapKernel<<<1, 1>>>(map, i.first, ptr);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&res, ptr, sizeof(traccc::transform3),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);

        EXPECT_EQ(res, i.second);
    }
}
