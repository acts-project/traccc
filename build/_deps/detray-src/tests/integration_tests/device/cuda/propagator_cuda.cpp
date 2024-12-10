/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/detectors/bfield.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "propagator_cuda_kernel.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// GTest include
#include <gtest/gtest.h>

using namespace detray;

class CudaPropConstBFieldMng
    : public ::testing::TestWithParam<std::tuple<float, float, vector3_t>> {};

/// Propagation test using unified memory
TEST_P(CudaPropConstBFieldMng, propagator) {

    // VecMem memory resource(s)
    vecmem::cuda::managed_memory_resource mng_mr;

    // Test configuration
    propagator_test_config cfg{};
    cfg.track_generator.phi_steps(20).theta_steps(20);
    cfg.track_generator.p_tot(10.f * unit<scalar_t>::GeV);
    cfg.track_generator.eta_range(-3.f, 3.f);
    cfg.propagation.navigation.search_window = {3u, 3u};
    // Configuration for non-z-aligned B-fields
    cfg.propagation.navigation.overstep_tolerance = std::get<0>(GetParam());
    cfg.propagation.stepping.step_constraint = std::get<1>(GetParam());

    // Get the magnetic field
    const vector3_t B = std::get<2>(GetParam());
    auto field = bfield::create_const_field(B);

    // Create the toy geometry
    auto [det, names] = build_toy_detector(mng_mr);

    run_propagation_test<bfield::const_bknd_t>(
        &mng_mr, det, cfg, detray::get_data(det), std::move(field));
}

class CudaPropConstBFieldCpy
    : public ::testing::TestWithParam<std::tuple<float, float, vector3_t>> {};

/// Propagation test using vecmem copy
TEST_P(CudaPropConstBFieldCpy, propagator) {

    // VecMem memory resource(s)
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource dev_mr;

    vecmem::cuda::copy cuda_cpy;

    // Test configuration
    propagator_test_config cfg{};
    cfg.track_generator.phi_steps(20u).theta_steps(20u);
    cfg.track_generator.p_tot(10.f * unit<scalar_t>::GeV);
    cfg.track_generator.eta_range(-3.f, 3.f);
    cfg.propagation.navigation.search_window = {3u, 3u};
    // Configuration for non-z-aligned B-fields
    cfg.propagation.navigation.overstep_tolerance = std::get<0>(GetParam());
    cfg.propagation.stepping.step_constraint = std::get<1>(GetParam());

    // Get the magnetic field
    const vector3_t B = std::get<2>(GetParam());
    auto field = bfield::create_const_field(B);

    // Create the toy geometry
    auto [det, names] = build_toy_detector(host_mr);

    auto det_buff = detray::get_buffer(det, dev_mr, cuda_cpy);

    run_propagation_test<bfield::const_bknd_t>(
        &mng_mr, det, cfg, detray::get_data(det_buff), std::move(field));
}

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation1, CudaPropConstBFieldMng,
    ::testing::Values(std::make_tuple(-100.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{0.f * unit<scalar_t>::T,
                                                0.f * unit<scalar_t>::T,
                                                2.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation2, CudaPropConstBFieldMng,
    ::testing::Values(std::make_tuple(-400.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{0.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation3, CudaPropConstBFieldMng,
    ::testing::Values(std::make_tuple(-400.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{1.f * unit<scalar_t>::T,
                                                0.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation4, CudaPropConstBFieldMng,
    ::testing::Values(std::make_tuple(-600.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation5, CudaPropConstBFieldCpy,
    ::testing::Values(std::make_tuple(-100.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{0.f * unit<scalar_t>::T,
                                                0.f * unit<scalar_t>::T,
                                                2.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation6, CudaPropConstBFieldCpy,
    ::testing::Values(std::make_tuple(-400.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{0.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation7, CudaPropConstBFieldCpy,
    ::testing::Values(std::make_tuple(-400.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{1.f * unit<scalar_t>::T,
                                                0.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

INSTANTIATE_TEST_SUITE_P(
    CudaPropagatorValidation8, CudaPropConstBFieldCpy,
    ::testing::Values(std::make_tuple(-600.f * unit<float>::um,
                                      std::numeric_limits<float>::max(),
                                      vector3_t{1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T,
                                                1.f * unit<scalar_t>::T})));

/// This tests the device propagation in an inhomogenepus magnetic field
TEST(CudaPropagatorValidation9, inhomogeneous_bfield_cpy) {

    // VecMem memory resource(s)
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource dev_mr;

    vecmem::cuda::copy cuda_cpy;

    // Test configuration
    propagator_test_config cfg{};
    cfg.track_generator.phi_steps(10u).theta_steps(10u);
    cfg.track_generator.p_tot(10.f * unit<scalar_t>::GeV);
    cfg.track_generator.eta_range(-3.f, 3.f);
    cfg.propagation.navigation.search_window = {3u, 3u};

    // Get the magnetic field
    auto field = bfield::create_inhom_field();

    // Create the toy geometry with inhomogeneous bfield from file
    auto [det, names] = build_toy_detector(host_mr);

    auto det_buff = detray::get_buffer(det, dev_mr, cuda_cpy);

    run_propagation_test<bfield::cuda::inhom_bknd_t>(
        &mng_mr, det, cfg, detray::get_data(det_buff), std::move(field));
}
