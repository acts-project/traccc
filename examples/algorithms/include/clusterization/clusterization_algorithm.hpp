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

namespace traccc {

class clusterization_algorithm
    : public algorithm<
          const host_cell_container&,
          std::pair<host_measurement_container, host_spacepoint_container> > {
    public:
    struct config {
        // currently left empty
    };

    output_type operator()(const input_type& cells_per_event) const override {
        output_type o;
        this->operator()(cells_per_event, o);
        return o;
    }

    void operator()(const input_type& cells_per_event,
                    output_type& o) const override {
        // output containers
        auto& measurements_per_event = o.first;
        auto& spacepoints_per_event = o.second;

        // reserve the vector size
        measurements_per_event.headers.reserve(cells_per_event.headers.size());
        measurements_per_event.items.reserve(cells_per_event.headers.size());
        spacepoints_per_event.headers.reserve(cells_per_event.headers.size());
        spacepoints_per_event.items.reserve(cells_per_event.headers.size());

        for (std::size_t i = 0; i < cells_per_event.items.size(); ++i) {
            const auto& module = cells_per_event.headers[i];

            // The algorithmic code part: start
            traccc::cluster_collection clusters_per_module =
                cc({cells_per_event.items[i], cells_per_event.headers[i]});
            clusters_per_module.position_from_cell = module.pixel;

            traccc::host_measurement_collection measurements_per_module =
                mt({clusters_per_module, module});
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp({module, measurements_per_module});
            // The algorithmnic code part: end

            measurements_per_event.items.push_back(
                std::move(measurements_per_module));
            measurements_per_event.headers.push_back(module);

            spacepoints_per_event.items.push_back(
                std::move(spacepoints_per_module));
            spacepoints_per_event.headers.push_back(module.module);
        }
    }

    private:
    // algorithms
    component_connection cc;
    measurement_creation mt;
    spacepoint_formation sp;
};

}  // namespace traccc
