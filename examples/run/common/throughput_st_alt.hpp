/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <string_view>

namespace traccc {

/// Helper function running a single-threaded throughput test
///
/// @tparam FULL_CHAIN_ALG The type of the full chain algorithm to use
/// @tparam HOST_MR The host memory resource type to use
/// @param description A short description of the application
/// @param argc The count of command line arguments (from @c main(...))
/// @param argv The command line arguments (from @c main(...))
/// @param use_host_caching Flag specifying whether host-side memory caching
///                         should be used
/// @return The value to be returned from @c main(...)
///
template <typename FULL_CHAIN_ALG,
          typename HOST_MR = vecmem::host_memory_resource>
int throughput_st_alt(std::string_view description, int argc, char* argv[],
                      bool use_host_caching = false);

}  // namespace traccc

// Local include(s).
#include "throughput_st_alt.ipp"
