/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/navigation/volume_graph.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/hash_tree.hpp"

// Example linear algebra plugin: std::array
#include "detray/tutorial/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <iostream>

// Hash of the "correct" geometry
constexpr std::size_t root_hash = 3458765454686814475ul;

///
/// Work in progress (!)
///
/// Get a graph that represents the detector volumes (nodes) and their links via
/// their adjacent boundary surfaces (edges) and hash it to detect linking
/// changes that might affect the navigation.
/// This could be useful in a CI job, but is poorly tested at this point (!).
int main() {

    // Get an example detector
    vecmem::host_memory_resource host_mr;
    const auto [det, names] = detray::build_toy_detector(host_mr);

    // Build the graph and get its adjacency matrix
    detray::volume_graph graph(det);
    const auto &adj_mat = graph.adjacency_matrix();

    // Construct a hash tree on the graph and compare against existing hash to
    // detect changes
    auto geo_checker = detray::hash_tree(adj_mat);

    if (geo_checker.root() == root_hash) {
        std::cout << "Geometry links are consistent" << std::endl;
    } else {
        std::cerr << "\nGeometry linking has changed! Please check:\n"
                  << graph.to_string() << std::endl;
    }
}
