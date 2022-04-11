/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"

// clusterization
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"

#ifdef _OPENMP
#include "omp.h"
#endif

namespace traccc {

class clusterization_algorithm
    : public algorithm<
          std::pair<host_measurement_container, host_spacepoint_container>(
              const host_cell_container&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is the memory resource
    clusterization_algorithm(vecmem::memory_resource& mr) : m_mr(mr) {

        cc = std::make_shared<traccc::component_connection>(
            traccc::component_connection(mr));
        mt = std::make_shared<traccc::measurement_creation>(
            traccc::measurement_creation(mr));
        sp = std::make_shared<traccc::spacepoint_formation>(
            traccc::spacepoint_formation(mr));
    }

    output_type operator()(
        const host_cell_container& cells_per_event) const override {
        output_type o({host_measurement_container(&m_mr.get()),
                       host_spacepoint_container(&m_mr.get())});
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
            traccc::host_cluster_container clusters = cc->operator()(
                cells_per_event.at(i).items, cells_per_event.at(i).header);
            for (auto& cl_id : clusters.get_headers()) {
                cl_id.is_default = false;
                cl_id.pixel = module.pixel;
            }

            traccc::host_measurement_collection measurements_per_module =
                mt->operator()(clusters, module);
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp->operator()(module, measurements_per_module);
            // The algorithmnic code part: end

            measurements_per_event.push_back(
                module, std::move(measurements_per_module));

            spacepoints_per_event.push_back(module.module,
                                            std::move(spacepoints_per_module));
        }
    }

    private:
    // algorithms
    std::shared_ptr<traccc::component_connection> cc;
    std::shared_ptr<traccc::measurement_creation> mt;
    std::shared_ptr<traccc::spacepoint_formation> sp;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
