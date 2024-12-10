/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/eigen_eigen.hpp"
#include "benchmark/common/benchmark_transform3.hpp"
#include "benchmark/eigen/data_generator.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

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

  transform3_bm<eigen::transform3<float>> v_trf_s{cfg};
  transform3_bm<eigen::transform3<double>> v_trf_d{cfg};

  std::cout << "Algebra-Plugins 'transform3' benchmark (Eigen3)\n"
            << "-----------------------------------------------\n\n"
            << cfg;

  //
  // Register all benchmarks
  //
  ::benchmark::RegisterBenchmark((v_trf_s.name() + "_single").c_str(), v_trf_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_trf_d.name() + "_double").c_str(), v_trf_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
}