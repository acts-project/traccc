/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray test include(s)
#include "detector_cuda_kernel.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Google Test include(s)
#include <gtest/gtest.h>

// System include(s)
#include <limits>

using namespace detray;

TEST(detector_cuda, detector) {
    // memory resource
    vecmem::cuda::managed_memory_resource mng_mr;

    // create toy geometry
    auto [toy_det, names] = build_toy_detector(mng_mr);

    auto ctx0 = typename detector_host_t::geometry_context();

    // host objects
    auto& volumes_host = toy_det.volumes();
    auto& surfaces_host = toy_det.surfaces();
    auto& transforms_host = toy_det.transform_store();
    auto& masks_host = toy_det.mask_store();
    auto& discs_host = masks_host.get<disc_id>();
    auto& cylinders_host = masks_host.get<cylinder_id>();
    auto& rectangles_host = masks_host.get<rectangle_id>();

    // copied outpus from device side
    vecmem::vector<det_volume_t> volumes_device(volumes_host.size(), &mng_mr);
    vecmem::vector<det_surface_t> surfaces_device(surfaces_host.size(),
                                                  &mng_mr);
    vecmem::vector<transform_t> transforms_device(transforms_host.size(),
                                                  &mng_mr);
    vecmem::vector<rectangle_t> rectangles_device(rectangles_host.size(),
                                                  &mng_mr);
    vecmem::vector<disc_t> discs_device(discs_host.size(), &mng_mr);
    vecmem::vector<cylinder_t> cylinders_device(cylinders_host.size(), &mng_mr);

    // get data object for toy detector
    auto toy_det_data = detray::get_data(toy_det);

    // get data object for device outputs
    auto volumes_data = vecmem::get_data(volumes_device);
    auto surfaces_data = vecmem::get_data(surfaces_device);
    auto transforms_data = vecmem::get_data(transforms_device);
    auto rectangles_data = vecmem::get_data(rectangles_device);
    auto discs_data = vecmem::get_data(discs_device);
    auto cylinders_data = vecmem::get_data(cylinders_device);

    // run the test code to copy the objects
    detector_test(toy_det_data, volumes_data, surfaces_data, transforms_data,
                  rectangles_data, discs_data, cylinders_data);

    // check if the same volume objects are copied
    for (unsigned int i = 0u; i < volumes_host.size(); i++) {
        EXPECT_EQ(volumes_host[i] == volumes_device[i], true);
    }

    // check if the same surface objects are copied
    for (unsigned int i = 0u; i < surfaces_host.size(); i++) {
        EXPECT_EQ(surfaces_device[i] == surfaces_host[i], true);
    }

    // check if the same transform objects are copied
    for (unsigned int i = 0u; i < transforms_host.size(ctx0); i++) {
        EXPECT_EQ(transforms_host.at(i, ctx0) == transforms_device[i], true);
    }

    // check if the same masks are copied
    for (unsigned int i = 0u; i < rectangles_host.size(); i++) {
        EXPECT_EQ(rectangles_host[i] == rectangles_device[i], true);
    }

    for (unsigned int i = 0u; i < discs_host.size(); i++) {
        EXPECT_EQ(discs_host[i] == discs_device[i], true);
    }

    for (unsigned int i = 0u; i < cylinders_host.size(); i++) {
        EXPECT_EQ(cylinders_host[i] == cylinders_device[i], true);
    }
}
