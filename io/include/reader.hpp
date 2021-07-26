#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "tmp_edm.hpp"
#include "csv/csv_io.hpp"
#include <vecmem/memory/host_memory_resource.hpp>

#include <functional>

using namespace std::placeholders;

namespace traccc {

    const std::string data_directory() {
        auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
        if (env_d_d == nullptr) {
            throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
        }
        std::string data_dir = std::string(env_d_d);
        return data_dir.append("/");
    }

    std::string get_event_filename(size_t event) {
        std::string event_string{"000000000"};
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(), event_number.size(), event_number);
        return event_string;
    }

    traccc::geometry read_geometry(const std::string &detector_file) {
        // Read the surface transforms
        std::string io_detector_file = data_directory() + detector_file;
        traccc::surface_reader sreader(io_detector_file,
                                       {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu",
                                        "rot_zv", "rot_zw"});
        return traccc::read_surfaces(sreader);
    }

    traccc::host_cell_container read_cells_from_event(size_t event, const std::string &cells_dir,
                                                      traccc::geometry surface_transforms, vecmem::host_memory_resource& resource) {
        // Read the cells from the relevant event file
        std::string io_cells_file =
                data_directory() + cells_dir + std::string("/event") + get_event_filename(event) +
                std::string("-cells.csv");
        traccc::cell_reader creader(io_cells_file,
                                    {"geometry_id", "hit_id", "cannel0", "channel1", "activation", "time"});
        return traccc::read_cells(creader, resource, surface_transforms);
    }

    traccc::demonstrator_input
    read(size_t events, const std::string &detector_file, const std::string &cell_directory, vecmem::host_memory_resource& resource) {
        auto geometry = read_geometry(detector_file);
        auto readFn = std::bind(read_cells_from_event, _1, cell_directory, geometry, resource);

        traccc::demonstrator_input input_data(events, &resource);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t event = 0; event < events; ++event) {
            traccc::host_cell_container cells_per_event = readFn(event);

#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                input_data[event] = cells_per_event;
            }

        }
        return input_data;
    }

} // end namespace