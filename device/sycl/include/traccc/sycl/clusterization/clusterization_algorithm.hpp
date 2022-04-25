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
#include "traccc/sycl/clusterization/measurement_creation.hpp"

namespace traccc::sycl {

class clusterization_algorithm
    : public algorithm<host_measurement_container(const host_cell_container&)> {

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
    }

    output_type operator()(
        const host_cell_container& cells_per_event) const override {

        output_type measurements_per_event(&m_mr.get());

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

        return measurements_per_event;
    }

    private:
    // algorithms
    std::shared_ptr<traccc::component_connection> cc;
    std::shared_ptr<traccc::sycl::measurement_creation> mt;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::sycl