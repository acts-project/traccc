/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/measurement_creation_algorithm.hpp"

#include "traccc/clusterization/details/measurement_creation.hpp"
#include "traccc/definitions/primitives.hpp"

namespace traccc::host {

measurement_creation_algorithm::measurement_creation_algorithm(
    vecmem::memory_resource &mr)
    : m_mr(mr) {}

measurement_creation_algorithm::output_type
measurement_creation_algorithm::operator()(
    const cluster_container_types::const_view &clusters_view,
    const cell_module_collection_types::const_view &modules_view) const {

    // Create device containers for the input variables.
    const cluster_container_types::const_device clusters{clusters_view};
    const cell_module_collection_types::const_device modules{modules_view};

    // Create the result object.
    output_type result(clusters.size(), &(m_mr.get()));
    measurement_collection_types::device measurements{vecmem::get_data(result)};

    // Process the clusters one-by-one.
    for (std::size_t i = 0; i < clusters.size(); ++i) {
        // To calculate the mean and variance with high numerical stability
        // we use a weighted variant of Welford's algorithm. This is a
        // single-pass online algorithm that works well for large numbers
        // of samples, as well as samples with very high values.
        //
        // To learn more about this algorithm please refer to:
        // [1] https://doi.org/10.1080/00401706.1962.10490022
        // [2] The Art of Computer Programming, Donald E. Knuth, second
        //     edition, chapter 4.2.2.

        // Get the cluster.
        cluster_container_types::device::item_vector::const_reference cluster =
            clusters.get_items()[i];

        // A security check.
        assert(cluster.empty() == false);

        // Get the cell module
        const unsigned int module_link = cluster.at(0).module_link;
        const auto &module = modules.at(module_link);

        // Fill measurement from cluster
        details::fill_measurement(measurements, i, cluster, module,
                                  module_link);
    }

    return result;
}

}  // namespace traccc::host
