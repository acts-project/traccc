/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/container.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc {

/// Header: unsigned int for number of seeds

/// Item: seed consisting of three spacepoints, z origin and weight
struct seed {

    using link_type = typename host_spacepoint_container::link_type;

    link_type spB_link;
    link_type spM_link;
    link_type spT_link;

    scalar weight;
    scalar z_vertex;

    seed() = default;
    seed(const seed&) = default;

    TRACCC_HOST_DEVICE
    seed& operator=(const seed& aSeed) {

        spB_link.first = aSeed.spB_link.first;
        spB_link.second = aSeed.spB_link.second;

        spM_link.first = aSeed.spM_link.first;
        spM_link.second = aSeed.spM_link.second;

        spT_link.first = aSeed.spT_link.first;
        spT_link.second = aSeed.spT_link.second;

        weight = aSeed.weight;
        z_vertex = aSeed.z_vertex;
        return *this;
    }

    TRACCC_HOST
    std::array<measurement, 3> get_measurements(
        const host_spacepoint_container& spacepoints) const {
        return {spacepoints.at(spB_link).meas, spacepoints.at(spM_link).meas,
                spacepoints.at(spT_link).meas};
    }

    TRACCC_HOST
    std::array<spacepoint, 3> get_spacepoints(
        const host_spacepoint_container& spacepoints) const {
        return {spacepoints.at(spB_link), spacepoints.at(spM_link),
                spacepoints.at(spT_link)};
    }
};

template <typename seed_collection_t, typename spacepoint_container_t>
TRACCC_HOST std::vector<std::array<spacepoint, 3>> get_spacepoint_vector(
    const seed_collection_t& seeds, const spacepoint_container_t& container) {

    std::vector<std::array<spacepoint, 3>> result;
    result.reserve(seeds.size());

    std::transform(
        seeds.cbegin(), seeds.cend(), std::back_inserter(result),
        [&](const seed& value) { return value.get_spacepoints(container); });

    return result;
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

}  // namespace traccc
