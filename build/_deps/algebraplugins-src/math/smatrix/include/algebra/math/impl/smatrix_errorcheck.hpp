/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// ROOT include(s).
#include <TError.h>

/// Macro helping with checking ROOT return/error codes
#define SMATRIX_CHECK(EXP)                                                  \
  do {                                                                      \
    const int _error_code = EXP;                                            \
    if (_error_code != 0) {                                                 \
      Fatal("algebra::smatrix", "%s:%i Failure detected in expression: %s", \
            __FILE__, __LINE__, #EXP);                                      \
    }                                                                       \
  } while (false)
