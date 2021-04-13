/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "algorithms/component_connection.hpp"
#include "csv/csv_io.hpp"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <chrono>

namespace {
    double delta_ms(
        std::chrono::high_resolution_clock::time_point s,
        std::chrono::high_resolution_clock::time_point e
    ) {
        return std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / 1000.0;
    }
}

void print_statistics(
    const traccc::cell_container & modules
) {
    static std::vector<std::size_t> bins_edges = {
        0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362,
        512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192
    };

    static std::size_t max_width = 50;

    std::vector<std::size_t> bins(bins_edges.size());

    for (const traccc::cell_collection & module : modules) {
        std::size_t count = module.items.size();

        for (std::size_t i = 0; i < bins_edges.size(); ++i) {
            if (
                count >= bins_edges[i] &&
                (i + 1 >= bins_edges.size() || count < bins_edges[i + 1])
            ) {
                ++bins[i];
                break;
            }
        }
    }

    std::size_t max = *std::max_element(bins.begin(), bins.end());

    std::size_t per_pixel = std::max(
        static_cast<std::size_t>(1),
        (max + (max % max_width)) / max_width
    );

    std::cout << "\nNon-zero pixels per module" << std::endl;

    std::cout << std::setw(5) << "Min" << "   " << std::setw(5) << "Max" <<
    " | " << std::setw(5) << "Count" << std::endl;

    for (std::size_t i = 0; i < bins.size(); ++i) {
        std::cout << std::setw(5) << bins_edges[i] << " - " << std::setw(5) <<
        (i + 1 >= bins.size() ? "inf" : std::to_string(bins_edges[i + 1] - 1)) <<
        " | " << std::setw(5) << bins[i] << " ";

        for (std::size_t j = 0; j < bins[i]; j += per_pixel) {
            std::cout << "*";
        }

        std::cout << std::endl;
    }
}

int main(
    int argc,
    char * argv[]
) {
    if (argc < 2){
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << argv[0] << " <event_file>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);

    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    traccc::component_connection cc;

    auto time_read_start = std::chrono::high_resolution_clock::now();

    traccc::cell_reader creader(event_file, {"geometry_id", "hit_id", "cannel0", "channel1", "activation", "time"});
    traccc::cell_container modules = traccc::read_cells(creader);

    auto time_read_end = std::chrono::high_resolution_clock::now();

    print_statistics(modules);

    auto time_process_start = std::chrono::high_resolution_clock::now();

    for (const traccc::cell_collection & module : modules) {
        traccc::cluster_collection clusters_per_module = cc(module);
    }

    auto time_process_end = std::chrono::high_resolution_clock::now();

    std::cout << "\nCPU budget allocation" << std::endl;
    std::cout << std::fixed;
    std::cout << std::setw(13) << "Component" << " | " << std::setw(13) << "Runtime" << std::endl;
    std::cout << std::setw(13) << "Data loading" << " | " << std::setw(10) << std::setprecision(3) << delta_ms(time_read_start, time_read_end) << " ms" << std::endl;
    std::cout << std::setw(13) << "Statistics" << " | " << std::setw(10) << std::setprecision(3) << delta_ms(time_read_end, time_process_start) << " ms" << std::endl;
    std::cout << std::setw(13) << "Clustering" << " | " << std::setw(10) << std::setprecision(3) << delta_ms(time_process_start, time_process_end) << " ms" << std::endl;
    std::cout << std::setw(13) << "Total" << " | " << std::setw(10) << std::setprecision(3) << delta_ms(time_read_start, time_process_end) << " ms" << std::endl;

    return 0;
}
