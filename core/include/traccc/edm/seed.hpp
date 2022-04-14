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
    std::vector<measurement> get_measurements(
        const host_spacepoint_container& spacepoints) const {
        std::vector<measurement> result;
        result.reserve(3);
        result.push_back(spacepoints.at(spB_link).meas);
        result.push_back(spacepoints.at(spM_link).meas);
        result.push_back(spacepoints.at(spT_link).meas);
        return result;
    }

    TRACCC_HOST
    std::vector<spacepoint> get_spacepoints(
        const host_spacepoint_container& spacepoints) const {
        std::vector<spacepoint> result;
        result.reserve(3);
        result.push_back(spacepoints.at(spB_link));
        result.push_back(spacepoints.at(spM_link));
        result.push_back(spacepoints.at(spT_link));
        return result;
    }
};

inline bool operator==(const seed& lhs, const seed& rhs) {
    return (lhs.spB_link == rhs.spB_link && lhs.spM_link == rhs.spM_link &&
            lhs.spT_link == rhs.spT_link);
}

inline bool operator==(const std::vector<spacepoint>& lhs,
                       const std::vector<spacepoint>& rhs) {
    assert(lhs.size() == rhs.size());

    for (std::size_t i = 0; i < lhs.size(); i++) {
        if (!(lhs.at(i) == rhs.at(i))) {
            return false;
        }
    }

    return true;
}

template <typename seed_collection_t, typename spacepoint_container_t>
std::vector<std::vector<spacepoint>> get_spacepoint_vector(
    const seed_collection_t& seeds, const spacepoint_container_t& container) {

    std::vector<std::vector<spacepoint>> result;
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
