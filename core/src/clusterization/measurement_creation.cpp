/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/measurement_creation.hpp"

#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include "traccc/definitions/primitives.hpp"

namespace traccc {

measurement_creation::measurement_creation(vecmem::memory_resource &mr)
    : m_mr(mr) {}

measurement_creation::output_type measurement_creation::operator()(
    const cell_container_types::host &cells,
    const cluster_container_types::host &clusters) const {

    // Create the result object.
    output_type result(&(m_mr.get()));
    result.reserve(cells.size());

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

        // Get the cell module
        const auto &module_link = clusters.at(i).header;
        const auto &module = cells.at(module_link).header;

        if (result.size() == 0 ||
            result.get_headers().back().module != module.module) {
            result.push_back(cell_module(module),
                             measurement_collection_types::host(&(m_mr.get())));
        }

        // Get the cluster.
        cluster_container_types::host::item_vector::const_reference cluster =
            clusters.at(i).items;

        // A security check.
        assert(cluster.empty() == false);

        // Fill measurement from cluster
        detail::fill_measurement(result, cluster, module, module_link, i);
    }

    return result;
}

}  // namespace traccc
