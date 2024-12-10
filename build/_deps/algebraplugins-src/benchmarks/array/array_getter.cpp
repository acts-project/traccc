/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/array_cmath.hpp"
#include "benchmark/array/data_generator.hpp"
#include "benchmark/common/benchmark_getter.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

// System include(s)
#include <string>

using namespace algebra;

/// Run vector benchmarks
int main(int argc, char** argv) {

  constexpr std::size_t n_samples{160000};
  constexpr std::size_t n_warmup{static_cast<std::size_t>(0.1 * n_samples)};

  //
  // Prepare benchmarks
  //
  algebra::benchmark_base::configuration cfg{};
  cfg.n_samples(n_samples).n_warmup(n_warmup);
  cfg.do_sleep(false);

  vector_unaryOP_bm<array::vector3, float, bench_op::phi> v_phi_s{cfg};
  vector_unaryOP_bm<array::vector3, float, bench_op::theta> v_theta_s{cfg};
  vector_unaryOP_bm<array::vector3, float, bench_op::perp> v_perp_s{cfg};
  vector_unaryOP_bm<array::vector3, float, bench_op::norm> v_norm_s{cfg};
  vector_unaryOP_bm<array::vector3, float, bench_op::eta> v_eta_s{cfg};

  vector_unaryOP_bm<array::vector3, double, bench_op::phi> v_phi_d{cfg};
  vector_unaryOP_bm<array::vector3, double, bench_op::theta> v_theta_d{cfg};
  vector_unaryOP_bm<array::vector3, double, bench_op::perp> v_perp_d{cfg};
  vector_unaryOP_bm<array::vector3, double, bench_op::norm> v_norm_d{cfg};
  vector_unaryOP_bm<array::vector3, double, bench_op::eta> v_eta_d{cfg};

  std::cout << "Algebra-Plugins 'getter' benchmark (std::array)\n"
            << "-----------------------------------------------\n\n"
            << cfg;

  //
  // Register all benchmarks
  //
  ::benchmark::RegisterBenchmark((v_phi_s.name() + "_single").c_str(), v_phi_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_phi_d.name() + "_double").c_str(), v_phi_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_theta_s.name() + "_single").c_str(),
                                 v_theta_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_theta_d.name() + "_double").c_str(),
                                 v_theta_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_perp_s.name() + "_single").c_str(),
                                 v_perp_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_perp_d.name() + "_double").c_str(),
                                 v_perp_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_norm_s.name() + "_single").c_str(),
                                 v_norm_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_norm_d.name() + "_double").c_str(),
                                 v_norm_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_eta_s.name() + "_single").c_str(), v_eta_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_eta_d.name() + "_double").c_str(), v_eta_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
}
