/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray core include(s).
#include "detray/grids/grid2.hpp"

#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/grids/axis.hpp"
#include "detray/grids/populator.hpp"
#include "detray/grids/serializer2.hpp"

// Detray test include(s).
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s)
#include <cstdlib>
#include <iostream>
#include <random>

// Use the detray:: namespace implicitly.
using namespace detray;

namespace {

#ifdef DETRAY_BENCHMARK_PRINTOUTS
/// Test point for printouts
const auto tp = test::point2{12.f, 30.f};
#endif

/// Prepare test points
auto make_random_points() {

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<scalar> dist1(0.f, 24.f);
    std::uniform_real_distribution<scalar> dist2(0.f, 59.f);

    std::vector<test::point2> points{};
    for (unsigned int itest = 0u; itest < 1000000u; ++itest) {
        points.push_back({dist1(gen), dist2(gen)});
    }

    return points;
}

/// Make a regular grid for the tests.
template <unsigned int kDIM = 1u,
          template <template <typename...> class, template <typename...> class,
                    template <typename, std::size_t> class, typename, bool,
                    unsigned int>
          class populator_t = replace_populator>
auto make_regular_grid(vecmem::memory_resource &mr) {

    // Return the grid.
    return grid2<populator_t, axis2::regular, axis2::regular, serializer2,
                 dvector, djagged_vector, darray, dtuple, dindex, false, kDIM>{
        {25u, 0.f, 25.f, mr}, {60u, 0.f, 60.f, mr}, mr};
}

/// Make an irregular grid for the tests.
template <unsigned int kDIM = 1u,
          template <template <typename...> class, template <typename...> class,
                    template <typename, std::size_t> class, typename, bool,
                    unsigned int>
          class populator_t = replace_populator>
auto make_irregular_grid(vecmem::memory_resource &mr) {

    // Fill a 25 x 60 grid with an "irregular" axis
    dvector<scalar> xboundaries;
    dvector<scalar> yboundaries;
    xboundaries.reserve(25u);
    yboundaries.reserve(60u);
    for (scalar i = 0.f; i < 61.f; i += 1.f) {
        if (i < 26.f) {
            xboundaries.push_back(i);
        }
        yboundaries.push_back(i);
    }

    return grid2<populator_t, axis2::irregular, axis2::irregular, serializer2,
                 dvector, djagged_vector, darray, dtuple, dindex, false, kDIM>(
        {xboundaries, mr}, {yboundaries, mr}, mr);
}

/// Fill a grid with some values.
template <typename grid_t>
void populate_grid(grid_t &grid, std::size_t bin_cap = 1) {

    for (dindex gbin = 0u; gbin < grid.nbins(); ++gbin) {
        // Fill the entire bin
        for (std::size_t i = 0u; i < bin_cap; ++i) {
            grid.populate(
                gbin,
                static_cast<typename grid_t::populator_type::bare_value>(gbin));
        }
    }
}

}  // namespace

// This runs a reference test with a regular grid structure
void BM_GRID2_REGULAR_BIN_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid(host_mr);
    populate_grid(g2r);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            benchmark::DoNotOptimize(g2r.bin(p));
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_BIN_CAP1:" << std::endl;
    std::cout << g2r.bin(tp) << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID2_REGULAR_BIN_CAP4(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<4u, complete_populator>(host_mr);
    populate_grid(g2r, 4u);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.bin(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_BIN_CAP4:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.bin(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID2_REGULAR_BIN_CAP25(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<25u, complete_populator>(host_mr);
    populate_grid(g2r, 25u);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.bin(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_BIN_CAP25:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.bin(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID2_REGULAR_BIN_CAP100(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<100u, complete_populator>(host_mr);
    populate_grid(g2r, 100u);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.bin(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_BIN_CAP100:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.bin(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID2_REGULAR_NEIGHBOR_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid(host_mr);
    populate_grid(g2r);

    auto points = make_random_points();

    // Helper zone object.
    static const darray<dindex, 2> zone22 = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.zone(p, {zone22, zone22})) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_NEIGHBOR_CAP1:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.zone(tp, {zone22, zone22})) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID2_REGULAR_NEIGHBOR_CAP4(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<4u, complete_populator>(host_mr);
    populate_grid(g2r, 4u);

    auto points = make_random_points();

    // Helper zone object.
    static const darray<dindex, 2> zone22 = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.zone(p, {zone22, zone22})) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_REGULAR_NEIGHBOR_CAP4:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.zone(tp, {zone22, zone22})) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a irregular grid structure
void BM_GRID2_IRREGULAR_BIN_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2irr = make_irregular_grid(host_mr);
    populate_grid(g2irr);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            benchmark::DoNotOptimize(g2irr.bin(p));
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_IRREGULAR_BIN_CAP1:" << std::endl;
    std::cout << g2irr.bin(tp) << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID2_IRREGULAR_NEIGHBOR_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2irr = make_irregular_grid(host_mr);
    populate_grid(g2irr);

    auto points = make_random_points();

    // Helper zone object.
    static const darray<dindex, 2> zone22 = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2irr.zone(p, {zone22, zone22})) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID2_IRREGULAR_NEIGHBOR_CAP1:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2irr.zone(tp, {zone22, zone22})) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_GRID2_REGULAR_BIN_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_REGULAR_BIN_CAP4)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_REGULAR_BIN_CAP25)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_REGULAR_BIN_CAP100)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_REGULAR_NEIGHBOR_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_REGULAR_NEIGHBOR_CAP4)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_IRREGULAR_BIN_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID2_IRREGULAR_NEIGHBOR_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);
