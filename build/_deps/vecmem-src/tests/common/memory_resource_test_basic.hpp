/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Base test case for memory resources
///
/// This just makes sure that the memory resources are more-or-less functional.
///
class memory_resource_test_basic
    : public testing::TestWithParam<vecmem::memory_resource*> {};

// Include the implementation.
#include "memory_resource_test_basic.ipp"
