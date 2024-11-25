/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "silicon_pixel_spacepoint_formation.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

namespace traccc::host {

silicon_pixel_spacepoint_formation_algorithm::output_type
silicon_pixel_spacepoint_formation_algorithm::operator()(
    const telescope_detector::host& det,
    const measurement_collection_types::const_view& meas) const {

    return details::silicon_pixel_spacepoint_formation(det, meas, m_mr);
}

}  // namespace traccc::host
