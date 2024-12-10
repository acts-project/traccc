/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "../common/memory_resource_test_alignment.hpp"
#include "../common/memory_resource_test_basic.hpp"
#include "../common/memory_resource_test_host_accessible.hpp"
#include "../common/memory_resource_test_stress.hpp"
#include "vecmem/memory/arena_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/choice_memory_resource.hpp"
#include "vecmem/memory/coalescing_memory_resource.hpp"
#include "vecmem/memory/conditional_memory_resource.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/debug_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/identity_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/memory/pool_memory_resource.hpp"
#include "vecmem/memory/synchronized_memory_resource.hpp"
#include "vecmem/memory/terminal_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// Utility memory resource used to define others.
static vecmem::terminal_memory_resource terminal_resource;

// Memory resources to use in the test.
static vecmem::host_memory_resource host_resource;
static vecmem::binary_page_memory_resource binary_resource(host_resource);
static vecmem::pool_memory_resource pool_resource(host_resource);
static vecmem::contiguous_memory_resource contiguous_resource(host_resource,
                                                              20000);
static vecmem::arena_memory_resource arena_resource(host_resource, 20000,
                                                    10000000);
static vecmem::instrumenting_memory_resource instrumenting_resource(
    host_resource);
static vecmem::synchronized_memory_resource synchronized_resource(
    host_resource);

static vecmem::identity_memory_resource identity_resource(host_resource);
static vecmem::conditional_memory_resource conditional_resource(
    host_resource, [](std::size_t, std::size_t) { return true; });
static vecmem::coalescing_memory_resource coalescing_resource_1(
    {host_resource});
static vecmem::coalescing_memory_resource coalescing_resource_2(
    {terminal_resource, terminal_resource, host_resource, terminal_resource});
static vecmem::choice_memory_resource choice_resource(
    [](std::size_t, std::size_t) -> vecmem::memory_resource& {
        return host_resource;
    });

static vecmem::debug_memory_resource debug_host_resource(host_resource);
static vecmem::debug_memory_resource debug_binary_resource(binary_resource);
static vecmem::debug_memory_resource debug_pool_resource(pool_resource);
static vecmem::debug_memory_resource debug_arena_resource(arena_resource);
static vecmem::debug_memory_resource debug_synchronized_resource(
    synchronized_resource);

// Set up the test name generating helper object.
static vecmem::testing::memory_resource_name_gen name_gen(
    {{&host_resource, "host_resource"},
     {&binary_resource, "binary_resource"},
     {&pool_resource, "pool_resource"},
     {&contiguous_resource, "contiguous_resource"},
     {&arena_resource, "arena_resource"},
     {&instrumenting_resource, "instrumenting_resource"},
     {&synchronized_resource, "synchronized_resource"},
     {&identity_resource, "identity_resource"},
     {&conditional_resource, "conditional_resource"},
     {&coalescing_resource_1, "coalescing_resource_1"},
     {&coalescing_resource_2, "coalescing_resource_2"},
     {&choice_resource, "choice_resource"},
     {&debug_host_resource, "debug_host_resource"},
     {&debug_binary_resource, "debug_binary_resource"},
     {&debug_pool_resource, "debug_pool_resource"},
     {&debug_arena_resource, "debug_arena_resource"},
     {&debug_synchronized_resource, "debug_synchronized_resource"}});

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_tests, memory_resource_test_basic,
    testing::Values(&host_resource, &binary_resource, &pool_resource,
                    &arena_resource, &instrumenting_resource,
                    &synchronized_resource, &identity_resource,
                    &conditional_resource, &coalescing_resource_1,
                    &coalescing_resource_2, &choice_resource,
                    &debug_host_resource, &debug_binary_resource,
                    &debug_pool_resource, &debug_arena_resource,
                    &debug_synchronized_resource),
    name_gen);

INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_tests, memory_resource_test_host_accessible,
    testing::Values(&host_resource, &binary_resource, &pool_resource,
                    &arena_resource, &instrumenting_resource,
                    &synchronized_resource, &identity_resource,
                    &conditional_resource, &coalescing_resource_1,
                    &coalescing_resource_2, &choice_resource,
                    &debug_host_resource, &debug_binary_resource,
                    &debug_pool_resource, &debug_arena_resource,
                    &debug_synchronized_resource),
    name_gen);

INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_tests, memory_resource_test_stress,
    testing::Values(&host_resource, &binary_resource, &pool_resource,
                    &arena_resource, &instrumenting_resource,
                    &synchronized_resource, &identity_resource,
                    &conditional_resource, &coalescing_resource_1,
                    &coalescing_resource_2, &choice_resource,
                    &debug_host_resource, &debug_binary_resource,
                    &debug_pool_resource, &debug_arena_resource,
                    &debug_synchronized_resource),
    name_gen);

INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_tests, memory_resource_test_alignment,
    testing::Values(&host_resource, &instrumenting_resource, &pool_resource,
                    &synchronized_resource, &identity_resource,
                    &conditional_resource, &coalescing_resource_1,
                    &coalescing_resource_2, &choice_resource,
                    &debug_host_resource, &debug_pool_resource,
                    &debug_synchronized_resource),
    name_gen);
