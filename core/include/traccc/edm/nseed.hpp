/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/edm/container.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>

namespace traccc {
/**
 * @brief Seed of arbitrary size.
 *
 * This type represents a set of at least three and at most N spacepoints which
 * are presumed to belong to a single track.
 *
 * @tparam N The maximum capacity of the seed.
 */
template <std::size_t N>
struct nseed {
    /*
     * Enforce the minimum seed size of three.
     */
    static_assert(N >= 3, "Seeds must contain at least three spacepoints.");

    /*
     * Should be the same as the three-seed type.
     */
    using link_type = spacepoint_collection_types::host::size_type;

    /**
     * @brief Construct a new n-seed object from a 3-seed object.
     *
     * @param s A 3-seed.
     */
    nseed(const seed& s)
        : _size(3), _sps({s.spB_link, s.spM_link, s.spT_link}) {}

    /**
     * @brief Get the size of the seed.
     */
    std::size_t size() const { return _size; }

    /**
     * @brief Get the first space point identifier in the seed.
     */
    const link_type* cbegin() const { return &_sps[0]; }

    /**
     * @brief Get the one-after-last space point identifier in the seed.
     */
    const link_type* cend() const { return &_sps[_size]; }

    private:
    std::size_t _size;
    std::array<link_type, N> _sps;
};
}  // namespace traccc
