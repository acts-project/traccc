/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <detray/geometry/shapes/line.hpp>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

struct is_line_visitor {
    template <typename mask_group_t, typename index_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline bool operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/) const {
        using shape_type = typename mask_group_t::value_type::shape;
        return std::is_same_v<shape_type, detray::line<true>> ||
               std::is_same_v<shape_type, detray::line<false>>;
    }
};

}  // namespace traccc
