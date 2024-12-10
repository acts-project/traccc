/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/vc_soa.hpp"
#include "benchmark/common/benchmark_transform3.hpp"
#include "benchmark/vc_soa/data_generator.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

using namespace algebra;

/// Run vector benchmarks
int main(int argc, char** argv) {

  constexpr std::size_t n_samples{160000};

  //
  // Prepare benchmarks
  //
  algebra::benchmark_base::configuration cfg_s{};
  // Reduce the number of samples, since a single SoA struct contains multiple
  // vectors
  cfg_s.n_samples(n_samples / Vc::float_v::Size)
      .n_warmup(static_cast<std::size_t>(0.1 * cfg_s.n_samples()));
  cfg_s.do_sleep(false);

  // For double precision we need more samples (less vectors per SoA)
  algebra::benchmark_base::configuration cfg_d{cfg_s};
  cfg_d.n_samples(n_samples / Vc::double_v::Size)
      .n_warmup(static_cast<std::size_t>(0.1 * cfg_d.n_samples()));

  transform3_bm<vc_soa::transform3<float>> v_trf_s{cfg_s};
  transform3_bm<vc_soa::transform3<double>> v_trf_d{cfg_d};

  std::cout << "Algebra-Plugins 'transform3' benchmark (Vc SoA)\n"
            << "-----------------------------------------------\n\n"
            << "(single)\n"
            << cfg_s << "(double)\n"
            << cfg_d;

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