/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/geometry.hpp"
#include "traccc/io/binary.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/io/demonstrator_edm.hpp"
#include "traccc/io/utils.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <fstream>
#include <functional>

namespace traccc {

inline traccc::geometry read_geometry(const std::string &detector_file) {
    // Read the surface transforms
    std::string io_detector_file = data_directory() + detector_file;
    traccc::surface_reader sreader(
        io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
                           "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    return traccc::read_surfaces(sreader);
}

/// Function for cell file reading. The output is traccc container.
///
/// @param event is the event index
/// @param cells_directory is the directory of cell file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param surface_transforms is the input geometry data
/// @param resource is the vecmem resource
inline traccc::cell_container_types::host read_cells_from_event(
    size_t event, const std::string &cells_directory,
    const traccc::data_format &data_format, traccc::geometry surface_transforms,
    vecmem::memory_resource &resource) {

    // Read the cells from the relevant event file
    if (data_format == traccc::data_format::csv) {
        std::string io_cells_file = data_directory() + cells_directory +
                                    get_event_filename(event, "-cells.csv");

        traccc::cell_reader creader(
            io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1",
                            "activation", "time"});
        return traccc::read_cells(creader, resource, &surface_transforms);

    } else if (data_format == traccc::data_format::binary) {
        std::string io_cells_file = data_directory() + cells_directory +
                                    get_event_filename(event, "-cells.dat");

        vecmem::copy copy;

        return traccc::read_binary<traccc::cell_container_types::host>(
            io_cells_file, copy, resource);
    } else {
        throw std::invalid_argument("Allowed data format is csv or binary");
    }
}

/// Function for spacepoint file reading. The output is traccc container.
///
/// @param event is the event index
/// @param hits_directory is the directory of cell file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param surface_transforms is the input geometry data
/// @param resource is the vecmem resource
inline traccc::host_spacepoint_container read_spacepoints_from_event(
    size_t event, const std::string &hits_directory,
    const traccc::data_format &data_format, traccc::geometry surface_transforms,
    vecmem::memory_resource &resource) {

    // Read the cells from the relevant event file
    if (data_format == traccc::data_format::csv) {
        std::string io_hits_file = data_directory() + hits_directory +
                                   get_event_filename(event, "-hits.csv");
        traccc::fatras_hit_reader hreader(
            io_hits_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});
        return traccc::read_hits(hreader, resource, &surface_transforms);
    } else if (data_format == traccc::data_format::binary) {
        std::string io_hits_file = data_directory() + hits_directory +
                                   get_event_filename(event, "-hits.dat");

        vecmem::copy copy;

        return traccc::read_binary<traccc::host_spacepoint_container>(
            io_hits_file, copy, resource);
    } else {
        throw std::invalid_argument("Allowed data format is csv or binary");
    }
}

inline traccc::demonstrator_input read(size_t events,
                                       const std::string &detector_file,
                                       const std::string &cell_directory,
                                       const traccc::data_format &data_format,
                                       vecmem::host_memory_resource &resource) {
    using namespace std::placeholders;
    auto geom = read_geometry(detector_file);
    auto readFn = std::bind(read_cells_from_event, _1, cell_directory,
                            data_format, geom, resource);
    traccc::demonstrator_input input_data(events, &resource);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t event = 0; event < events; ++event) {
        traccc::cell_container_types::host cells_per_event = readFn(event);

#if defined(_OPENMP)
#pragma omp critical
#endif
        { input_data[event] = cells_per_event; }
    }
    return input_data;
}

}  // namespace traccc
