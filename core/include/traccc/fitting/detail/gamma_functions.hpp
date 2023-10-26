/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"

// gamma and related functions from Cephes library
// see:  http://www.netlib.org/cephes
//
// Copyright 1985, 1987, 2000 by Stephen L. Moshier
namespace traccc::detail {

// incomplete gamma function (complement integral)
//  igamc(a,x)   =   1 - igam(a,x)
//
//                            inf.
//                              -
//                     1       | |  -t  a-1
//               =   -----     |   e   t   dt.
//                    -      | |
//                   | (a)    -
//                             x
//
//

// In this implementation both arguments must be positive.
// The integral is evaluated by either a power series or
// continued fraction expansion, depending on the relative
// values of a and x.
TRACCC_HOST_DEVICE double igamc(const double a, const double x);

// left tail of incomplete gamma function:
//
//          inf.      k
//   a  -x   -       x
//  x  e     >   ----------
//           -     -
//          k=0   | (a+k+1)
//
TRACCC_HOST_DEVICE double igam(const double a, const double x);

TRACCC_HOST_DEVICE double lgam(const double x);

/*
 * calculates a value of a polynomial of the form:
 * a[0]x^N+a[1]x^(N-1) + ... + a[N]
 */
TRACCC_HOST_DEVICE double Polynomialeval(const double x, const double* a,
                                         const unsigned int N);

/*
 * calculates a value of a polynomial of the form:
 * x^N+a[0]x^(N-1) + ... + a[N-1]
 */
TRACCC_HOST_DEVICE double Polynomial1eval(const double x, const double* a,
                                          const unsigned int N);

}  // namespace traccc::detail