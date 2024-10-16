/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/csv/cell.hpp"

namespace traccc::io::csv {

bool operator<(const cell& lhs, const cell& rhs) {
    if (lhs.geometry_id != rhs.geometry_id) {
        return lhs.geometry_id < rhs.geometry_id;
    } else if (lhs.channel1 != rhs.channel1) {
        return (lhs.channel1 < rhs.channel1);
    } else {
        return (lhs.channel0 < rhs.channel0);
    }
}

}  // namespace traccc::io::csv
