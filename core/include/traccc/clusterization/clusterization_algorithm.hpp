/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"

// clusterization
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"

namespace traccc {

class clusterization_algorithm
    : public algorithm<host_measurement_container(const host_cell_container&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is the memory resource
    clusterization_algorithm(vecmem::memory_resource& mr) : m_mr(mr) {

        cc = std::make_shared<traccc::component_connection>(
            traccc::component_connection(mr));
        mt = std::make_shared<traccc::measurement_creation>(
            traccc::measurement_creation(mr));
    }

    output_type operator()(
        const host_cell_container& cells_per_event) const override {

        output_type measurements_per_event(&m_mr.get());

        measurements_per_event.reserve(cells_per_event.size());

        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto module = cells_per_event.at(i).header;

            // The algorithmic code part: start
            traccc::host_cluster_container clusters = cc->operator()(
                cells_per_event.at(i).items, cells_per_event.at(i).header);
            for (auto& cl_id : clusters.get_headers()) {
                cl_id.pixel = module.pixel;
            }
            traccc::host_measurement_collection measurements_per_module =
                mt->operator()(clusters, module);

            // The algorithmnic code part: end
            measurements_per_event.push_back(
                module, std::move(measurements_per_module));
        }

        return measurements_per_event;
    }

    private:
    // algorithms
    std::shared_ptr<traccc::component_connection> cc;
    std::shared_ptr<traccc::measurement_creation> mt;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
