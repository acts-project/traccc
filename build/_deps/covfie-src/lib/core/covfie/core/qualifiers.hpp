/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(__CUDACC__)
#define COVFIE_DEVICE __host__ __device__
#else
#define COVFIE_DEVICE
#endif
