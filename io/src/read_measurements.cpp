/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_measurements.hpp"

#include "csv/read_measurements.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

measurement_container_types::host read_measurements(
    std::size_t event, std::string_view directory, data_format format,
    vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_measurements(
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.csv"),
                format, mr);
        case data_format::binary:
            return read_measurements(
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.dat"),
                format, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

measurement_container_types::host read_measurements(
    std::string_view filename, data_format format,
    vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_measurements(filename, mr);
        case data_format::binary:
            return details::read_binary_container<
                measurement_container_types::host>(filename, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
