/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
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
#include <functional>
#include <tuple>

/// Parameter for the copy tests
using soa_device_test_parameters =
    std::tuple<vecmem::memory_resource&, vecmem::memory_resource&,
               vecmem::memory_resource&, vecmem::copy&, void*, void*>;

template <typename CONTAINER>
class soa_device_tests_base
    : public ::testing::TestWithParam<soa_device_test_parameters> {

protected:
    /// Test modifying the container in managed memory
    static void modify_managed(const soa_device_test_parameters& params);
    /// Test modifying the container in device memory
    static void modify_device(const soa_device_test_parameters& params);
    /// Test filling the container in device memory
    static void fill_device(const soa_device_test_parameters& params);

};  // class soa_device_tests_base

/// Parametrized device tests for the "simple" SoA container
using soa_device_tests_simple =
    soa_device_tests_base<vecmem::testing::simple_soa_container>;

/// Parametrized device tests for the "jagged" SoA container
using soa_device_tests_jagged =
    soa_device_tests_base<vecmem::testing::jagged_soa_container>;

// Include the implementation.
#include "soa_device_tests.ipp"
