/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/edm/alt_seed.hpp>
#include <traccc/edm/container.hpp>
#include <traccc/edm/spacepoint.hpp>

namespace traccc {
template <std::size_t N>
struct nseed {
    static_assert(N >= 3, "Seeds must contain at least three spacepoints.");

    using link_type = spacepoint_collection_types::host::size_type;

    nseed(const alt_seed& s)
        : _size(3), _sps({s.spB_link, s.spM_link, s.spT_link}) {}

    std::size_t size() const { return _size; }

    const link_type* cbegin() const { return &_sps[0]; }

    const link_type* cend() const { return &_sps[_size]; }

    private:
    std::size_t _size;
    std::array<link_type, 3> _sps;
};
}  // namespace traccc
