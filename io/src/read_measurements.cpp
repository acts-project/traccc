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

void read_measurements(measurement_reader_output& out, std::size_t event,
                       std::string_view directory, data_format format) {

    switch (format) {
        case data_format::csv: {
            read_measurements(
                out,
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.csv"),
                format);
            break;
        }
        case data_format::binary: {

            details::read_binary_collection<
                alt_measurement_collection_types::host>(
                out.measurements,
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.dat"));
            details::read_binary_collection<cell_module_collection_types::host>(
                out.modules, data_directory() + directory.data() +
                                 get_event_filename(event, "-modules.dat"));
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_measurements(measurement_reader_output& out,
                       std::string_view filename, data_format format) {

    switch (format) {
        case data_format::csv:
            return csv::read_measurements(out, filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

measurement_container_types::host read_measurements_container(
    std::size_t event, std::string_view directory, data_format format,
    vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_measurements_container(
                data_directory() + directory.data() +
                    get_event_filename(event, "-measurements.csv"),
                format, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

measurement_container_types::host read_measurements_container(
    std::string_view filename, data_format format,
    vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_measurements_container(filename, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
