/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <benchmark/benchmark.h>
#include <boost/mp11.hpp>

#include <covfie/benchmark/register.hpp>

#include "backends/constant.hpp"
#include "backends/test_field.hpp"
#include "patterns/euler.hpp"
#include "patterns/lorentz.hpp"
#include "patterns/runge_kutta4.hpp"

void register_benchmarks(void)
{
    covfie::benchmark::register_product_bm<
        boost::mp11::mp_list<
            Lorentz<Euler, Deep>,
            Lorentz<Euler, Wide>,
            RungeKutta4Pattern<Deep>,
            RungeKutta4Pattern<Wide>,
            EulerPattern<Deep>,
            EulerPattern<Wide>>,
        boost::mp11::mp_list<
            FieldConstant,
            Field<InterpolateNN, LayoutStride>,
            Field<InterpolateNN, LayoutMortonNaive>,
            Field<InterpolateNN, LayoutMortonBMI2>,
            Field<InterpolateLin, LayoutStride>,
            Field<InterpolateLin, LayoutMortonNaive>,
            Field<InterpolateLin, LayoutMortonBMI2>>>();
}

int main(int argc, char ** argv)
{
    register_benchmarks();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
