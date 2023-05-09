/* Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifdef TRACCC_HAVE_CXXABI_H
extern "C" {
#include <cxxabi.h>
}
#endif // TRACCC_HAVE_CXX_ABI_H
#include <dlfcn.h>
#include <stdio.h>

#include <array>
#include <exception>
#include <iostream>

#include "nvtx3/nvToolsExt.h"

#define UNUSED(x) (void)(x)
const char *const default_name = "Unknown";

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);
static int color_id = 0;

extern "C" void __cyg_profile_func_enter(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));
extern "C" void __cyg_profile_func_exit(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));

void rangePush(const char *const name) __attribute__((no_instrument_function));
void rangePop() __attribute__((no_instrument_function));
Dl_info this_fn_info;







extern "C" void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    (void)this_fn;
    (void)call_site;
    if (dladdr(this_fn, &this_fn_info)) {
#ifdef TRACCC_HAVE_CXXABI_H
        std::array<char const*, 4> lookup =
            {{
                "The demangling operation succeeded",
                "A memory allocation failure occurred",
                "mangled_name is not a valid name under the C++ ABI mangling rules",
                "One of the arguments is invalid"
            }};

        std::size_t sz = 17;
        char* buffer = static_cast<char*>(std::malloc(sz));
        int status=0;
        std::bad_exception x;
        std::exception & e = x;
        char *realname = abi::__cxa_demangle(e.what(), buffer, &sz, &status);
        char * fname = abi::__cxa_demangle(this_fn_info.dli_sname, 0, 0, &status);
        if (status!=0) {
            std::cout << e.what() << "\t=> `" << realname << "'\t: " << status
                      << '\n';
            buffer = realname;
        } else {
            std::cout << "demangle for `" << e.what() << "' failed due to `"
                      << lookup[std::abs(status)] << "'\n";
        }
#else
        const char* fname = this_fn_info.dli_sname;
#endif
        rangePush(fname);
    } else {
        rangePush(default_name);
    }
} /* __cyg_profile_func_enter */

extern "C" void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)this_fn;
    (void)call_site;
    rangePop();
} /* __cyg_profile_func_enter */

void rangePush(const char *const name) {
    nvtxEventAttributes_t eventAttrib;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = colors[color_id];
    color_id = (color_id + 1) % num_colors;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.category = 3;
    eventAttrib.message.ascii = "Example Range";
    //        nvtxRangeId_t rangeId = nvtxRangeStartEx(&eventAttrib);
    if (name != 0) {
        eventAttrib.message.ascii = name;
    } else {
        eventAttrib.message.ascii = default_name;
    }
    nvtxRangePushEx(&eventAttrib);
}

void rangePop() {
    nvtxRangePop();
}

#ifdef TRACCC_HAVE_CXX_ABI_H
free(fname);
#endif