/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"

// clusterization
#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"

#ifdef _OPENMP
#include "omp.h"
#endif

namespace traccc {

class clusterization_algorithm
    : public algorithm<
          std::pair<host_measurement_container, host_spacepoint_container>(
              const host_cell_container&)> {
    public:
    output_type operator()(
        const host_cell_container& cells_per_event) const override {
        output_type o;
        this->operator()(cells_per_event, o);
        return o;
    }

    void operator()(const host_cell_container& cells_per_event,
                    output_type& o) const {
        // output containers
        auto& measurements_per_event = o.first;
        auto& spacepoints_per_event = o.second;

        // reserve the vector size
        measurements_per_event.reserve(cells_per_event.size());
        spacepoints_per_event.reserve(cells_per_event.size());

        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto module = cells_per_event.at(i).header;

            // The algorithmic code part: start
            traccc::cluster_collection clusters_per_module =
                cc(cells_per_event.at(i).items, cells_per_event.at(i).header);
            clusters_per_module.position_from_cell = module.pixel;

            traccc::host_measurement_collection measurements_per_module =
                mt(clusters_per_module, module);
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp(module, measurements_per_module);
            // The algorithmnic code part: end

            measurements_per_event.push_back(
                module, std::move(measurements_per_module));

            spacepoints_per_event.push_back(module.module,
                                            std::move(spacepoints_per_module));
        }
    }

    private:
    // algorithms
    component_connection cc;
    measurement_creation mt;
    spacepoint_formation sp;
};

}  // namespace traccc
