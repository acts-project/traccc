/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_spacepoints.hpp"

#include "make_hit_reader.hpp"

// System include(s).
#include <algorithm>

namespace traccc::io::csv {

spacepoint_container_types::host read_spacepoints(std::string_view filename,
                                                  const geometry& geom,
                                                  vecmem::memory_resource* mr) {

    // Construct the spacepoint reader object.
    auto reader = make_hit_reader(filename);

    // Create the result container.
    spacepoint_container_types::host result;
    if (mr != nullptr) {
        result = spacepoint_container_types::host{mr};
    }

    // Read the spacepoints from the input file.
    hit iohit;
    while (reader.read(iohit)) {

        // Find the local<->global transformation for the spacepoint's detector
        // module.
        const transform3& placement = geom[iohit.geometry_id];

        // Construct the global 3D position of the spacepoint.
        const point3 pos{iohit.tx, iohit.ty, iohit.tz};

        // Construct the local 3D(2D) position of the measurement.
        const point3 lpos = placement.point_to_local(pos);

        // Create the spacepoint object (with its member measurement) from all
        // this information.
        const traccc::spacepoint sp{
            pos, {point2{lpos[0], lpos[1]}, variance2{0., 0.}}};

        // Find the detector module that this spacepoint belongs to.
        const spacepoint_container_types::host::header_vector& headers =
            result.get_headers();
        auto rit =
            std::find(headers.rbegin(), headers.rend(), iohit.geometry_id);

        // Add the spacepoint to the correct place in the container.
        if (rit == headers.rend()) {
            if (mr != nullptr) {
                result.push_back(
                    iohit.geometry_id,
                    spacepoint_container_types::host::item_vector::value_type(
                        {sp}, mr));
            } else {
                result.push_back(
                    iohit.geometry_id,
                    spacepoint_container_types::host::item_vector::value_type(
                        {sp}));
            }
        } else {
            // The reverse iterator.base() returns the equivalent normal
            // iterator shifted by 1, so that the (r)end and (r)begin iterators
            // match consistently, due to the extra past-the-last element
            const std::size_t idx =
                std::distance(headers.begin(), rit.base()) - 1;
            result.at(idx).items.push_back(sp);
        }
    }

    // Return the container.
    return result;
}

}  // namespace traccc::io::csv
