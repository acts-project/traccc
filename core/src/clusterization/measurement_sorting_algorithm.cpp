/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/clusterization/measurement_sorting_algorithm.hpp"

#include <algorithm>

namespace traccc::host {

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const measurement_collection_types::view& measurements_view) const {

    // Create a device container on top of the view.
    measurement_collection_types::device measurements{measurements_view};

    // Sort the measurements in place
    std::sort(measurements.begin(), measurements.end(),
              measurement_sort_comp());

    // Return the view of the sorted measurements.
    return measurements_view;
}

}  // namespace traccc::host
