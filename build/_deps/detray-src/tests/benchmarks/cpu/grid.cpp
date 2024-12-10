/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray core include(s).
#include "detray/utils/grid/grid.hpp"

#include "detray/builders/grid_factory.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/utils/grid/detail/concepts.hpp"

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
#include <vector>

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
template <typename bin_t>
auto make_regular_grid(vecmem::memory_resource &mr) {

    // Data-owning grids with bin capacity 1
    auto gr_factory = grid_factory<bin_t, simple_serializer>{mr};

    // Spans of the axes
    std::vector<scalar> spans = {0.f, 25.f, 0.f, 60.f};
    // #bins of the axes
    std::vector<std::size_t> nbins = {25u, 60u};

    // Rectangular grid with closed bin bounds and regular binning on all axes
    return gr_factory.template new_grid<rectangle2D>(
        spans, nbins, {}, {},
        types::list<axis::closed<axis::label::e_x>,
                    axis::closed<axis::label::e_y>>{},
        types::list<axis::regular<>, axis::regular<>>{});
}

/// Make an irregular grid for the tests.
template <typename bin_t>
auto make_irregular_grid(vecmem::memory_resource &mr) {

    // Fill a 25 x 60 grid with an "irregular" axis
    std::vector<std::vector<scalar>> boundaries(2);
    boundaries[0].reserve(25u);
    boundaries[1].reserve(60u);
    for (scalar i = 0.f; i < 61.f; i += 1.f) {
        if (i < 26.f) {
            boundaries[0].push_back(i);
        }
        boundaries[1].push_back(i);
    }

    // Data-owning grids with bin capacity 1
    auto gr_factory = grid_factory<bin_t, simple_serializer>{mr};

    // Rectangular grid with closed bin bounds and irregular binning on all axes
    return gr_factory.template new_grid<rectangle2D>(
        {}, {}, {}, boundaries,
        types::list<axis::closed<axis::label::e_x>,
                    axis::closed<axis::label::e_y>>{},
        types::list<axis::irregular<>, axis::irregular<>>{});
}

/// Fill a grid with some values.
template <typename populator_t, concepts::grid grid_t>
void populate_grid(grid_t &grid) {

    for (dindex gbin = 0u; gbin < grid.nbins(); ++gbin) {
        grid.template populate<populator_t>(
            gbin, static_cast<typename grid_t::value_type>(gbin));
    }
}

}  // namespace

// This runs a reference test with a regular grid structure
void BM_GRID_REGULAR_BIN_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::single<dindex>>(host_mr);
    populate_grid<replace<>>(g2r);

    // Prepare test points
    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            benchmark::DoNotOptimize(g2r.search(p).value());
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_BIN_CAP1:" << std::endl;
    std::cout << g2r.search(p).value() << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID_REGULAR_BIN_CAP4(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::static_array<dindex, 4>>(host_mr);
    populate_grid<complete<>>(g2r);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.search(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_BIN_CAP4:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.search(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID_REGULAR_BIN_CAP25(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::static_array<dindex, 25>>(host_mr);
    populate_grid<complete<>>(g2r);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.search(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_BIN_CAP25:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.search(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a regular grid structure
void BM_GRID_REGULAR_BIN_CAP100(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::static_array<dindex, 100>>(host_mr);
    populate_grid<complete<>>(g2r);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.search(p)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_BIN_CAP100:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.search(tp)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID_REGULAR_NEIGHBOR_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::single<dindex>>(host_mr);
    populate_grid<replace<>>(g2r);

    auto points = make_random_points();

    // Search window size.
    static const darray<dindex, 2> window = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.search(p, window)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_NEIGHBOR_CAP1:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.search(tp, window)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID_REGULAR_NEIGHBOR_CAP4(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2r = make_regular_grid<bins::static_array<dindex, 4>>(host_mr);
    populate_grid<complete<>>(g2r);

    auto points = make_random_points();

    // Search window size.
    static const darray<dindex, 2> window = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2r.search(p, window)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_REGULAR_NEIGHBOR_CAP4:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2r.search(tp, window)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

// This runs a reference test with a irregular grid structure
void BM_GRID_IRREGULAR_BIN_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2irr = make_irregular_grid<bins::single<dindex>>(host_mr);
    populate_grid<replace<>>(g2irr);

    auto points = make_random_points();

    for (auto _ : state) {
        for (const auto &p : points) {
            benchmark::DoNotOptimize(g2irr.search(p).value());
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_IRREGULAR_BIN_CAP1:" << std::endl;
    std::cout << g2irr.search(tp).value() << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

void BM_GRID_IRREGULAR_NEIGHBOR_CAP1(benchmark::State &state) {

    // Set up the tested grid object.
    vecmem::host_memory_resource host_mr;
    auto g2irr = make_irregular_grid<bins::single<dindex>>(host_mr);
    populate_grid<replace<>>(g2irr);

    auto points = make_random_points();

    // Search window size.
    static const darray<dindex, 2> window = {2u, 2u};

    for (auto _ : state) {
        for (const auto &p : points) {
            for (const dindex entry : g2irr.search(p, window)) {
                benchmark::DoNotOptimize(entry);
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "BM_GRID_IRREGULAR_NEIGHBOR_CAP1:" << std::endl;
    std::size_t count{0u};
    for (const dindex entry : g2irr.search(tp, window)) {
        std::cout << entry << ", ";
        ++count;
    }
    std::cout << "\n=> Neighbors: " << count << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_GRID_REGULAR_BIN_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_REGULAR_BIN_CAP4)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_REGULAR_BIN_CAP25)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_REGULAR_BIN_CAP100)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_REGULAR_NEIGHBOR_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_REGULAR_NEIGHBOR_CAP4)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_IRREGULAR_BIN_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GRID_IRREGULAR_NEIGHBOR_CAP1)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);
