/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/vc_soa.hpp"
#include "benchmark/common/benchmark_getter.hpp"
#include "benchmark/vc_soa/data_generator.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

// System include(s)
#include <string>

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

  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::phi> v_phi_s{cfg_s};
  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::theta> v_theta_s{cfg_s};
  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::perp> v_perp_s{cfg_s};
  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::norm> v_norm_s{cfg_s};
  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::eta> v_eta_s{cfg_s};

  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::phi> v_phi_d{cfg_d};
  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::theta> v_theta_d{cfg_d};
  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::perp> v_perp_d{cfg_d};
  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::norm> v_norm_d{cfg_d};
  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::eta> v_eta_d{cfg_d};

  std::cout << "Algebra-Plugins 'getter' benchmark (Vc SoA)\n"
            << "-------------------------------------------\n\n"
            << "(single)\n"
            << cfg_s << "(double)\n"
            << cfg_d;

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
