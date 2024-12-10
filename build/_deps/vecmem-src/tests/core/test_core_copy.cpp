/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "../common/copy_tests.hpp"
#include "../common/soa_copy_tests.hpp"

// VecMem include(s).
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

// Objects used in the test(s).
static vecmem::host_memory_resource core_host_resource;
static vecmem::memory_resource* core_host_resource_ptr = &core_host_resource;
static vecmem::copy core_copy;
static vecmem::copy* core_copy_ptr = &core_copy;

/// The configurations to run the tests with.
static const auto core_copy_configs =
    testing::Values(std::tie(core_copy_ptr, core_copy_ptr,
                             core_host_resource_ptr, core_host_resource_ptr));

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(core_copy_tests, copy_tests, core_copy_configs);
INSTANTIATE_TEST_SUITE_P(core_soa_copy_tests_simple, soa_copy_tests_simple,
                         core_copy_configs);
INSTANTIATE_TEST_SUITE_P(core_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         core_copy_configs);
