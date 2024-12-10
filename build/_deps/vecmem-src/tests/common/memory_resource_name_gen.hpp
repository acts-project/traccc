/* VecMem project, part of the ACTS project (R&D line)
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

// System include(s).
#include <map>
#include <string>

namespace vecmem::testing {

/// Custom functor for printing user readable names for memory resources
///
/// In parametrised GoogleTests, where the parameter is a pointer to a
/// memory resource, this functor can come in handy for printing nice names
/// for the tests to the screen.
///
class memory_resource_name_gen {

public:
    /// Storage type for the memory resource names
    typedef std::map<memory_resource*, std::string> storage_type;

    /// Constructor with the known memory resource instances, and their names
    memory_resource_name_gen(const storage_type& names);

    /// Operator returning a user readable name for a memory resource pointer
    std::string operator()(
        const ::testing::TestParamInfo<memory_resource*>& info);

private:
    /// Internal map keeping track of the user readable names of the resources
    storage_type m_names;
    /// Unknown name counter
    int m_unknown_count;

};  // class memory_resource_name_gen

}  // namespace vecmem::testing
