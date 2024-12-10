/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// TODO: Remove this when gcc fixes their false positives.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

// Detray core include(s).
#include "detray/definitions/detail/indexing.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"
#include "detray/navigation/intersection/intersection.hpp"

// Detray test include(s).
#include "detray/test/utils/types.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s).
#include <iostream>

// Use the detray:: namespace implicitly.
using namespace detray;
using point3 = test::point3;

static constexpr unsigned int steps_x3{1000u};
static constexpr unsigned int steps_y3{1000u};
static constexpr unsigned int steps_z3{1000u};

static const test::transform3 trf{};

// Tolerance applied to the mask
static constexpr scalar tol{0.f};

// This runs a benchmark on a rectangle2D mask
void BM_MASK_CUBOID_3D(benchmark::State &state) {

    using mask_type = mask<cuboid3D>;
    constexpr mask_type cb(0u, 0.f, 0.f, 0.f, 3.f, 4.f, 1.f);

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{cb.to_local_frame(trf, {x, y, z})};
                    if (cb.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar volume{cb[3] * cb[4] * cb[5]};
    constexpr scalar rest{world * world * world - volume};
    std::cout << "Cuboid : Inside/outside ... " << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << volume / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_CUBOID_3D)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a rectangle2D mask
void BM_MASK_RECTANGLE_2D(benchmark::State &state) {

    using mask_type = mask<rectangle2D>;
    constexpr mask_type r(0u, 3.f, 4.f);

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{r.to_local_frame(trf, {x, y, z})};
                    if (r.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar area{4.f * r[0] * r[1]};
    constexpr scalar rest{world * world - area};
    std::cout << "Rectangle : Inside/outside ... " << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << area / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_RECTANGLE_2D)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a trapezoid2D mask
void BM_MASK_TRAPEZOID_2D(benchmark::State &state) {

    using mask_type = mask<trapezoid2D>;
    constexpr mask_type t{0u, 2.f, 3.f, 4.f, 1.f / (2.f * 4.f)};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{t.to_local_frame(trf, {x, y, z})};
                    if (t.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar area{2.f * (t[0] + t[1]) * t[2]};
    constexpr scalar rest{world * world - area};
    std::cout << "Trapezoid : Inside/outside ..." << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << area / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_TRAPEZOID_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a ring2D mask (as disc)
void BM_MASK_DISC_2D(benchmark::State &state) {

    using mask_type = mask<ring2D>;
    constexpr mask_type r{0u, 0.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{r.to_local_frame(trf, {x, y, z})};
                    if (r.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar area{r[1] * r[1] * constant<scalar>::pi};
    constexpr scalar rest{world * world - area};
    std::cout << "Disc : Inside/outside ..." << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << area / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_DISC_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a ring2D mask
void BM_MASK_RING_2D(benchmark::State &state) {

    using mask_type = mask<ring2D>;
    constexpr mask_type r{0u, 2.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{r.to_local_frame(trf, {x, y, z})};
                    if (r.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar area{(r[1] * r[1] - r[0] * r[0]) * constant<scalar>::pi};
    constexpr scalar rest{world * world - area};
    std::cout << "Ring : Inside/outside ..." << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << area / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_RING_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a cylinder2D mask
void BM_MASK_CYLINDER_3D(benchmark::State &state) {

    using mask_type = mask<cylinder3D>;
    constexpr mask_type c{
        0u, 1.f, -constant<scalar>::pi, 0.f, 3.f, constant<scalar>::pi, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{c.to_local_frame(trf, {x, y, z})};
                    if (c.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar volume{constant<scalar>::pi * (c[5] - c[2]) *
                            (c[3] * c[3] - c[0] * c[0])};
    constexpr scalar rest{world * world * world - volume};
    std::cout << "Cylinder 3D : Inside/outside ... " << inside << " / "
              << outside << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << volume / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_CYLINDER_3D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a cylinder2D mask
void BM_MASK_CYLINDER_2D(benchmark::State &state) {

    using mask_type = mask<cylinder2D>;
    constexpr mask_type c{0u, 3.f, 0.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{c.to_local_frame(trf, {x, y, z})};
                    if (c.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "Cylinder 2D : Inside/outside ..." << inside << " / "
              << outside << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << 1.f << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_CYLINDER_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a cylinder2D mask
void BM_MASK_CONCENTRIC_CYLINDER_2D(benchmark::State &state) {

    using mask_type = mask<concentric_cylinder2D>;
    constexpr mask_type c{0u, 3.f, 0.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{c.to_local_frame(trf, {x, y, z})};
                    if (c.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "Concnetric Cylinder : Inside/outside ..." << inside << " / "
              << outside << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << 1.f << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_CONCENTRIC_CYLINDER_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on an annulus2D mask
void BM_MASK_ANNULUS_2D(benchmark::State &state) {

    using mask_type = mask<annulus2D>;
    constexpr mask_type ann{0u, 2.5f, 5.f, -0.64299f, 4.13173f, 1.f, 0.5f, 0.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{ann.to_local_frame(trf, {x, y, z})};
                    if (ann.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "Annulus : Inside/outside ..." << inside << " / " << outside
              << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_ANNULUS_2D)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a straw tube mask
void BM_MASK_LINE_CIRCULAR(benchmark::State &state) {

    using mask_type = mask<line_circular>;
    constexpr mask_type st{0u, 3.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{st.to_local_frame(trf, {x, y, z})};
                    if (st.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar volume{constant<scalar>::pi * 2.f * st[1] * st[0] * st[0]};
    constexpr scalar rest{world * world * world - volume};
    std::cout << "Straw Tube : Inside/outside ... " << inside << " / "
              << outside << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << volume / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_LINE_CIRCULAR)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);

// This runs a benchmark on a wire cell mask
void BM_MASK_LINE_SQUARE(benchmark::State &state) {

    using mask_type = mask<line_square>;
    constexpr mask_type dcl{0u, 3.f, 5.f};

    constexpr scalar world{10.f};

    constexpr scalar sx{world / steps_x3};
    constexpr scalar sy{world / steps_y3};
    constexpr scalar sz{world / steps_z3};

    unsigned long inside = 0u;
    unsigned long outside = 0u;

    for (auto _ : state) {
        for (unsigned int ix = 0u; ix < steps_x3; ++ix) {
            scalar x{-0.5f * world + static_cast<scalar>(ix) * sx};
            for (unsigned int iy = 0u; iy < steps_y3; ++iy) {
                scalar y{-0.5f * world + static_cast<scalar>(iy) * sy};
                for (unsigned int iz = 0u; iz < steps_z3; ++iz) {
                    scalar z{-0.5f * world + static_cast<scalar>(iz) * sz};

                    benchmark::DoNotOptimize(inside);
                    benchmark::DoNotOptimize(outside);
                    const point3 loc_p{dcl.to_local_frame(trf, {x, y, z})};
                    if (dcl.is_inside(loc_p, tol)) {
                        ++inside;
                    } else {
                        ++outside;
                    }
                }
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    constexpr scalar volume{8.f * dcl[1] * dcl[0] * dcl[0]};
    constexpr scalar rest{world * world * world - volume};
    std::cout << "Drift Chamber Cell : Inside/outside ... " << inside << " / "
              << outside << " = "
              << static_cast<scalar>(inside) / static_cast<scalar>(outside)
              << " (theoretical = " << volume / rest << ") " << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_MASK_LINE_SQUARE)
#ifdef DETRAY_BENCHMARKS_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);
