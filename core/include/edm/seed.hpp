/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/container.hpp"

namespace traccc {

/// Header: unsigned int for number of seeds

/// Item: seed consisting of three spacepoints, z origin and weight
struct seed {
    spacepoint spB;
    spacepoint spM;
    spacepoint spT;
    float weight;
    float z_vertex;

    __CUDA_HOST_DEVICE__
    seed& operator=(const seed& aSeed) {
        spB = aSeed.spB;
        spM = aSeed.spM;
        spT = aSeed.spT;
        weight = aSeed.weight;
        z_vertex = aSeed.z_vertex;
        return *this;
    }
};

inline bool operator==(const seed& lhs, const seed& rhs) {
    return (lhs.spB == rhs.spB && lhs.spM == rhs.spM && lhs.spT == rhs.spT);
}

/// Container of internal_spacepoint for an event
template <template <typename> class vector_t>
using seed_collection = vector_t<seed>;

/// Convenience declaration for the seed collection type to use
/// in host code
using host_seed_collection = seed_collection<vecmem::vector>;

/// Convenience declaration for the seed collection type to use
/// in device code
using device_seed_collection = seed_collection<vecmem::device_vector>;

/// Convenience declaration for the seed container type to use in
/// host code
using host_seed_container = host_container<unsigned int, seed>;

/// Convenience declaration for the seed container type to use in
/// device code
using device_seed_container = device_container<unsigned int, seed>;

/// Convenience declaration for the seed container data type to
/// use in host code
using seed_container_data = container_data<unsigned int, seed>;

/// Convenience declaration for the seed container buffer type to
/// use in host code
using seed_container_buffer = container_buffer<unsigned int, seed>;

/// Convenience declaration for the seed container view type to
/// use in host code
using seed_container_view = container_view<unsigned int, seed>;

};  // namespace traccc
