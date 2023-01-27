/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/device/partitioning_algorithm.hpp"

// System include(s).
#include <stdexcept>

namespace {
// Check if two cells have at least 1 full empty column between them or belong
// to different modules in order to be partitioned. This assumes sorting by
// channel1 already done
bool isFarEnough(const traccc::alt_cell& c1, const traccc::alt_cell& c2) {
    if (c1.module_link != c2.module_link || c1.c.channel1 + 1 < c2.c.channel1) {
        return true;
    }
    return false;
}
}  // namespace

namespace traccc::device {

partitioning_algorithm::partitioning_algorithm(
    vecmem::memory_resource& mr, const unsigned short max_cells_per_partition)
    : m_max_cells_per_partition(max_cells_per_partition), m_mr(mr) {}

partitioning_algorithm::output_type partitioning_algorithm::operator()(
    const alt_cell_collection_types::host& cells) const {

    // Total number of cells
    const unsigned int num_cells = cells.size();
    // Number of cells yet to be partitioned
    unsigned int num_cells_remaining = num_cells;
    // End point of last partition
    unsigned int last_end = 0;

    // Create result object
    partitioning_algorithm::output_type partitions{0, &(m_mr.get())};
    // Reserve the minimum possible size of the result object
    partitions.reserve(num_cells / m_max_cells_per_partition);

    // The first partition point is the beggining of the vector
    partitions.push_back(0);

    while (num_cells_remaining > m_max_cells_per_partition) {

        bool broke = false;
        // Check if cell at a distance of max_cells_per_partition from last end
        // point can be used for partitioning. If not, look for nearest possible
        // previous cell.
        for (unsigned int i = m_max_cells_per_partition + last_end;
             i > last_end; --i) {
            if (isFarEnough(cells[i - 1], cells[i])) {
                partitions.push_back(i);
                last_end = i;
                num_cells_remaining = num_cells - i;
                broke = true;
                break;
            }
        }
        if (!broke) {
            // Prevent infinite loop
            throw std::invalid_argument("Select larger partitions_per_cell.");
        }
    }
    assert(num_cells_remaining <= std::numeric_limits<unsigned short>::max());
    assert(last_end + num_cells_remaining == num_cells);

    // The final partition point is the end of the vector
    partitions.push_back(num_cells);

    return partitions;
}

}  // namespace traccc::device
