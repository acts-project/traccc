/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/utils/create_path.hpp"

// System include(s)
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace detray::io {

/// @brief Wrapper around a file stream
///
/// Performs some checks and offers some convenience functionality:
/// - In out mode w.o. file replacement: Checks if file exists and automatically
///   finds a new filename in case it does.
/// - In in mode: Checks for empty file names and whether the file exists
/// - Enforces a limit on the number of files that can be written/opened
/// - Checks whether a file was opened correctly
/// - Closes the stream when the handle goes out of scope and checks whether
///   anything went wrong during the IO operations
///
/// @note Not thread safe.
/// @note Can throw exceptions during construction.
class file_handle final {

    public:
    /// All writers must define a file name
    file_handle() = delete;

    /// File gets created with a @param name and @param extension
    file_handle(const std::string& file_name,
                std::ios_base::openmode mode = std::ios_base::in |
                                               std::ios_base::out)
        : file_handle(std::filesystem::path{file_name}.parent_path() /
                          std::filesystem::path{file_name}.stem(),
                      std::filesystem::path{file_name}.extension(), mode) {}

    /// File gets created with a @param name and @param extension
    file_handle(const std::string& name, const std::string& extension,
                std::ios_base::openmode mode = std::ios_base::in |
                                               std::ios_base::out) {
        // File name
        std::string file_name{name};

        // Pure output mode without replacement of file: Check if name is taken
        // and modify it if necessary
        if (mode == std::ios_base::out ||
            (mode == (std::ios_base::out | std::ios_base::binary))) {
            // Default name for output
            file_name = name.empty() ? "./detray_" + std::to_string(n_files)
                                     : file_name;

            // Does the file stem need to be adjusted (in case the file exists)?
            std::string new_name = io::alt_file_name(file_name + extension);
            auto new_path = std::filesystem::path{new_name};
            file_name = new_path.parent_path() / new_path.stem();

            // Pure input mode: Check if file name makes sense and file exists
        } else if ((mode == std::ios_base::in) ||
                   (mode == (std::ios_base::in | std::ios_base::binary))) {
            if (file_name.empty()) {
                throw std::invalid_argument("File name empty");
            }

            std::filesystem::path file_path{file_name + extension};
            if (!std::filesystem::exists(file_path)) {
                throw std::invalid_argument(
                    "Could not open file: File does not exist: " + file_name +
                    extension);
            }
        } else if (file_name.empty()) {
            std::cout << "WARNING: Empty file name" << std::endl;
        }

        // Count the new file
        const std::string file_path{file_name + extension};
        ++n_files;
        ++n_open_files;
        if (n_files >= std::numeric_limits<std::uint_least16_t>::max()) {
            throw std::runtime_error(
                "Could not open file: Too many files written: " + file_path);
        } else if (n_open_files >= 1000u) {
            throw std::runtime_error(
                "Could not open file: Too many files currently open: " +
                file_path);
        }

        // Open file
        m_stream.open(file_path, mode);

        if (!m_stream.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }
    }

    /// Destructor closes the file
    ~file_handle() {
        if (m_stream.bad()) {
            std::cout << "ERROR: Could not read from/write to file";
        }
        try {
            m_stream.close();
        } catch (std::fstream::failure& err) {
            std::cout << "ERROR: Could not properly close file:\n"
                      << err.what() << std::endl;
        }
        --n_open_files;
    }

    /// @returns the output stream
    std::fstream& operator*() { return m_stream; }

    private:
    /// Output file handle
    std::fstream m_stream;

    /// How many files have been created? Maximum: 65'536
    inline static std::size_t n_files{0u};
    inline static std::size_t n_open_files{0u};
};

}  // namespace detray::io
