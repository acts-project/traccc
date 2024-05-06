/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/write.hpp"

#include "obj/write_spacepoints.hpp"
#include "traccc/io/utils.hpp"
#include "write_binary.hpp"

// System include(s).
#include <filesystem>
#include <stdexcept>

namespace traccc::io {

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           traccc::cell_collection_types::const_view cells,
           traccc::cell_module_collection_types::const_view modules) {

    switch (format) {
        case data_format::binary:
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.dat")))
                                      .native()),
                traccc::cell_collection_types::const_device{cells});
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-modules.dat")))
                                      .native()),
                traccc::cell_module_collection_types::const_device{modules});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           spacepoint_collection_types::const_view spacepoints,
           cell_module_collection_types::const_view modules) {

    switch (format) {
        case data_format::binary:
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-hits.dat")))
                                      .native()),
                traccc::spacepoint_collection_types::const_device{spacepoints});
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-modules.dat")))
                                      .native()),
                traccc::cell_module_collection_types::const_device{modules});
            break;
        case data_format::obj:
            obj::write_spacepoints(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-spacepoints.obj")))
                                      .native()),
                spacepoints);
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           measurement_collection_types::const_view measurements,
           traccc::cell_module_collection_types::const_view modules) {

    switch (format) {
        case data_format::binary:
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-measurements.dat")))
                                      .native()),
                traccc::measurement_collection_types::const_device{
                    measurements});
            details::write_binary_collection(
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-modules.dat")))
                                      .native()),
                traccc::cell_module_collection_types::const_device{modules});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
