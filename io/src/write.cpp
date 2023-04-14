/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/write.hpp"

#include "traccc/io/utils.hpp"
#include "write_binary.hpp"

namespace traccc::io {

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           traccc::cell_collection_types::const_view cells,
           traccc::cell_module_collection_types::const_view modules) {

    switch (format) {
        case data_format::binary:
            details::write_binary_collection(
                data_directory() + directory.data() +
                    get_event_filename(event, "-cells.dat"),
                traccc::cell_collection_types::const_device{cells});
            details::write_binary_collection(
                data_directory() + directory.data() +
                    get_event_filename(event, "-modules.dat"),
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
                data_directory() + directory.data() +
                    get_event_filename(event, "-hits.dat"),
                traccc::spacepoint_collection_types::const_device{spacepoints});
            details::write_binary_collection(
                data_directory() + directory.data() +
                    get_event_filename(event, "-modules.dat"),
                traccc::cell_module_collection_types::const_device{modules});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           alt_measurement_collection_types::const_view measurements,
           traccc::cell_module_collection_types::const_view modules) {

    switch (format) {
        case data_format::binary:
            details::write_binary_collection(
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.dat"),
                traccc::alt_measurement_collection_types::const_device{
                    measurements});
            details::write_binary_collection(
                data_directory() + directory.data() +
                    get_event_filename(event, "-modules.dat"),
                traccc::cell_module_collection_types::const_device{modules});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
