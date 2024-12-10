/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s)
#include <array>
#include <ostream>
#include <string>
#include <vector>

namespace detray::io {

/// @brief config struct for detector reading.
struct detector_reader_config {
    /// Input files
    std::vector<std::string> m_files;
    /// Run detector consistency check after reading
    bool m_do_check{true};
    /// Verbosity of the detector consistency check
    bool m_verbose{false};

    /// Getters
    /// @{
    const std::vector<std::string>& files() const { return m_files; }
    bool do_check() const { return m_do_check; }
    bool verbose_check() const { return m_verbose; }
    /// @}

    /// Setters
    /// @{
    detector_reader_config& add_file(const std::string& file_name) {
        m_files.push_back(file_name);
        return *this;
    }
    detector_reader_config& do_check(const bool check) {
        m_do_check = check;
        return *this;
    }
    detector_reader_config& verbose_check(const bool verbose) {
        if (verbose && !m_do_check) {
            m_do_check = true;
        }
        m_verbose = verbose;
        return *this;
    }
    /// @}

    /// Print the detector reader configuration
    friend std::ostream& operator<<(std::ostream& out,
                                    const detector_reader_config& cfg) {

        out << "\nDetector reader\n"
            << "----------------------------\n"
            << "  Detector files:       : \n";
        for (const auto& file_name : cfg.files()) {
            out << "    -> " << file_name << "\n";
        }

        return out;
    }
};

}  // namespace detray::io
