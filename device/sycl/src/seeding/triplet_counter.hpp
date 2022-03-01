/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/seeding/detail/doublet.hpp"

namespace traccc {
namespace sycl {

/// Definition the container for triplet counter
///
/// header: number of the mid-bot doublet and triplets per bin
struct triplet_counter_per_bin {
    unsigned int n_mid_bot = 0;
    unsigned int n_triplets = 0;

    TRACCC_HOST_DEVICE
    unsigned int get_ref_num() const { return n_mid_bot; }

    TRACCC_HOST_DEVICE
    void zeros() {
        n_mid_bot = 0;
        n_triplets = 0;
    }
};

/// item: number of triplets per mid-bot doublet
struct triplet_counter {

    /// indices of two spacepoints of mid-bot doublet
    doublet mid_bot_doublet;

    /// number of compatible triplets for a given mid-bot doublet
    unsigned int n_triplets = 0;
};

/// Container of triplet_counter belonging to one detector module
template <template <typename> class vector_t>
using triplet_counter_collection = vector_t<triplet_counter>;

/// Convenience declaration for the triplet_counter collection type to use in
/// host code
using host_triplet_counter_collection =
    triplet_counter_collection<vecmem::vector>;

/// Convenience declaration for the triplet_counter collection type to use in
/// device code
using device_triplet_counter_collection =
    triplet_counter_collection<vecmem::device_vector>;

/// Convenience declaration for the triplet_counter container type to use in
/// host code
using host_triplet_counter_container =
    host_container<triplet_counter_per_bin, triplet_counter>;

/// Convenience declaration for the triplet_counter container type to use in
/// device code
using device_triplet_counter_container =
    device_container<triplet_counter_per_bin, triplet_counter>;

/// Convenience declaration for the triplet_counter container data type to use
/// in host code
using triplet_counter_container_data =
    container_data<triplet_counter_per_bin, triplet_counter>;

/// Convenience declaration for the triplet_counter container buffer type to use
/// in host code
using triplet_counter_container_buffer =
    container_buffer<triplet_counter_per_bin, triplet_counter>;

/// Convenience declaration for the triplet_counter container view type to use
/// in host code
using triplet_counter_container_view =
    container_view<triplet_counter_per_bin, triplet_counter>;

}  // namespace sycl
}  // namespace traccc
