/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"

// clusterization
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/sycl/clusterization/measurement_creation.hpp"

namespace traccc::sycl {

class clusterization_algorithm
    : public algorithm<
          std::pair<host_measurement_container, host_spacepoint_container>(
              const host_cell_container&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is the memory resource
    clusterization_algorithm(vecmem::memory_resource& mr,
                             ::sycl::queue* q = nullptr)
        : m_mr(mr) {

        cc = std::make_shared<traccc::component_connection>(
            traccc::component_connection(mr));
        mt = std::make_shared<traccc::sycl::measurement_creation>(
            traccc::sycl::measurement_creation(mr, q));
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
        spacepoints_per_event.reserve(cells_per_event.size());

        // Container for all the clusters
        traccc::host_cluster_container clusters(&m_mr.get());

        // The algorithmic code part: start

        // Perform component connection per module
        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto module = cells_per_event.at(i).header;

            traccc::host_cluster_container clusters_per_module = cc->operator()(
                cells_per_event.at(i).items, cells_per_event.at(i).header);

            // Add module information to the cluster headers
            for (std::size_t j = 0; j < clusters_per_module.size(); ++j) {

                auto& cluster_id = clusters_per_module.at(j).header;
                cluster_id.module_idx = i;
                cluster_id.pixel = module.pixel;

                // Push the clusters from module to the total cluster container
                clusters.push_back(std::move(clusters_per_module.at(j).header),
                                   std::move(clusters_per_module.at(j).items));
            }
        }

        // Perform measurement creation across clusters from all modules in
        // parallel
        measurements_per_event =
            mt->operator()(clusters, cells_per_event.get_headers());

        // Perform the spacepoint creation
        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto module = cells_per_event.at(i).header;
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp->operator()(module, measurements_per_event.at(i).items);

            spacepoints_per_event.push_back(module.module,
                                            std::move(spacepoints_per_module));
        }
        // The algorithmic code part: end
    }

    private:
    // algorithms
    std::shared_ptr<traccc::component_connection> cc;
    std::shared_ptr<traccc::sycl::measurement_creation> mt;
    std::shared_ptr<traccc::spacepoint_formation> sp;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::sycl