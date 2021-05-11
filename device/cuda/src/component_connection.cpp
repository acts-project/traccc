/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/containers/vector.hpp"

#include "cuda/algorithms/component_connection.hpp"

#include "details/sparse_ccl.cuh"

namespace traccc::cuda {
    host_measurement_collection
    component_connection::operator()(
        const host_cell_container & data
    ) const {
        vecmem::cuda::managed_memory_resource upstream;
        vecmem::binary_page_memory_resource mem(upstream);

        std::size_t total_cells = 0;

        for (std::size_t i = 0; i < data.headers.size(); ++i) {
            total_cells += data.items.at(i).size();
        }

        vecmem::vector<cell> cells(&mem);
        cells.reserve(total_cells);

        vecmem::vector<unsigned int> blocks(data.headers.size() + 1, &mem);
        blocks[0] = 0;

        vecmem::vector<float> out(4 * MAX_CLUSTERS_PER_MODULE * data.headers.size(), &mem);

        std::size_t modules = data.headers.size();

        for (std::size_t i = 0; i < data.headers.size(); ++i) {
            cells.insert(
                cells.end(),
                data.items.at(i).begin(),
                data.items.at(i).end()
            );
            blocks[i + 1] = blocks[i] + data.items.at(i).size();
        }

        details::sparse_ccl(
            cells.data(),
            blocks.data(),
            out.data(),
            modules
        );

        // for (std::size_t i = 0; i < modules; ++i) {
        //     std::cout << "\n==== " << i << " ====" << std::endl;
        //     for (const traccc::cell & i : data.items.at(i)) {
        //         std::cout << "(" << i.channel0 << ", " << i.channel1 << "), ";
        //     }
        //     std::cout << std::endl;
        //     for (std::size_t j = 0; j < 16; ++j) {
        //         std::cout << i << ", " << j << " (" << out[4 * 128 * i + 4 * j] <<
        //         ", " << out[4 * 128 * i + 4 * j + 1] << ") with variance (" <<
        //         out[4 * 128 * i + 4 * j + 2] << ", " << out[4 * 128 * i + 4 * j + 3] <<
        //         ")" << std::endl;
        //     }
        // }

        return {};
    }
}
