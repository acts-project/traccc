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
           traccc::cell_container_types::const_view cells) {

    switch (format) {
        case data_format::binary:
            details::write_binary_container(
                data_directory() + directory.data() +
                    get_event_filename(event, "-cells.dat"),
                traccc::cell_container_types::const_device{cells});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           spacepoint_container_types::const_view spacepoints) {

    switch (format) {
        case data_format::binary:
            details::write_binary_container(
                data_directory() + directory.data() +
                    get_event_filename(event, "-hits.dat"),
                traccc::spacepoint_container_types::const_device{spacepoints});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           measurement_container_types::const_view measurements) {

    switch (format) {
        case data_format::binary:
            details::write_binary_container(
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.dat"),
                traccc::measurement_container_types::const_device{
                    measurements});
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
