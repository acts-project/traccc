/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// Local include(s).
#include "jagged_soa_container.hpp"
#include "simple_soa_container.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

/// Parameter (pack) for the copy tests
// using soa_copy_test_parameters =

/// Test fixture for the SoA copy tests
template <typename CONTAINER>
class soa_copy_tests_base
    : public ::testing::TestWithParam<
          std::tuple<vecmem::copy*&, vecmem::copy*&, vecmem::memory_resource*&,
                     vecmem::memory_resource*&>> {

protected:
    /// Access the "main" copy object
    vecmem::copy& main_copy();
    /// Access the "host" copy object
    vecmem::copy& host_copy();
    /// Access the "main" / "device" memory resource
    vecmem::memory_resource& main_mr();
    /// Access the "host" memory resource
    vecmem::memory_resource& host_mr();

    /// Test the simple/direct host->fixed device->host copy
    void host_to_fixed_device_to_host_direct();
    /// Test the "optimal" host->fixed device->host copy
    void host_to_fixed_device_to_host_optimal();
    /// Test the host->resizable device->host copy
    void host_to_resizable_device_to_host();
    /// Test the host->fixed device->resizable device->host copy
    void host_to_fixed_device_to_resizable_device_to_host();

};  // class soa_copy_tests_base

/// Parametrized copy tests for the "simple" SoA container
using soa_copy_tests_simple =
    soa_copy_tests_base<vecmem::testing::simple_soa_container>;

/// Parametrized copy tests for the "jagged" SoA container
using soa_copy_tests_jagged =
    soa_copy_tests_base<vecmem::testing::jagged_soa_container>;

// Include the implementation.
#include "soa_copy_tests.ipp"
