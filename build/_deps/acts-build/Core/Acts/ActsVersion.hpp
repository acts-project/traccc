// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

//  Caution: this is the only Acts header that is guaranteed
//  to change with every Acts release. Including this header
//  will cause a recompile every time a new Acts version is
//  used.

#include <iosfwd>

namespace Acts {

// clang-format does not like the CMake  replacement variables
// clang-format off
constexpr unsigned int VersionMajor = 37u;
constexpr unsigned int VersionMinor = 3u;
constexpr unsigned int VersionPatch = 1u;
// clang-format on
constexpr unsigned int Version =
    10000u * VersionMajor + 100u * VersionMinor + VersionPatch;
constexpr const char* const CommitHash = "c9d68283f4f33a8ab4021cc656ee35b8ee77827c";
constexpr const char* const CommitHashShort = "c9d68283f";

struct VersionInfo {
  unsigned int versionMajor;
  unsigned int versionMinor;
  unsigned int versionPatch;
  const char* const commitHash;

  VersionInfo() = delete;

  static VersionInfo fromHeader() {
    return VersionInfo(VersionMajor, VersionMinor, VersionPatch, CommitHash);
  }

  static VersionInfo fromLibrary();

  bool operator==(const VersionInfo& other) const;

  friend std::ostream& operator<<(std::ostream& os, const VersionInfo& vi);

 private:
  VersionInfo(unsigned int majorIn, unsigned int minorIn, unsigned int patchIn,
              const char* const commitHashIn);
};

}  // namespace Acts
