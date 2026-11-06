# Copyright 2026 Ant International, Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

WARNING_FLAGS = [
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-unused-parameter",
]

DEBUG_FLAGS = ["-DSPDLOG_ACTIVE_LEVEL=1", "-O0", "-g"]

# RELEASE_FLAGS = ["-O2"]
RELEASE_FLAGS = ["-O3", "-mavx2", "-msse4.1", "-maes", "-msse2", "-march=native"]
FAST_FLAGS = ["-O1"]

def _capricorn_copts():
    return select({
        "@capricorn//bazel:capricorn_build_as_release": RELEASE_FLAGS,
        "@capricorn//bazel:capricorn_build_as_debug": DEBUG_FLAGS,
        "@capricorn//bazel:capricorn_build_as_fast": FAST_FLAGS,
        "//conditions:default": FAST_FLAGS,
    }) + WARNING_FLAGS

def capricorn_cc_binary(
        linkopts = [],
        copts = [],
        defines = [],
        **kargs):
    cc_binary(
        linkopts = linkopts + ["-lpthread"],
        copts = copts + _capricorn_copts(),
        defines = defines + ["THREADING"],
        linkstatic = True,
        **kargs
    )

def capricorn_cc_library(
        linkopts = [],
        copts = [],
        deps = [],
        defines = [],
        **kargs):
    cc_library(
        linkopts = linkopts + ["-lpthread"],
        copts = _capricorn_copts() + copts,
        deps = deps + [
            "@spdlog//:spdlog",
        ],
        defines = defines + ["THREADING"],
        linkstatic = True,
        **kargs
    )

def capricorn_cc_test(
        linkopts = [],
        copts = [],
        deps = [],
        defines = [],
        **kwargs):
    cc_test(
        # -lm for tcmalloc
        linkopts = linkopts + ["-lm", "-lpthread"],
        copts = _capricorn_copts() + copts,
        deps = deps + [
            "@googletest//:gtest_main",
            "@spdlog//:spdlog",
        ],
        defines = defines + ["THREADING"],
        linkstatic = True,
        **kwargs
    )

def capricorn_cc_test_with_main(
        linkopts = [],
        copts = [],
        deps = [],
        defines = [],
        **kwargs):
    cc_test(
        # -lm for tcmalloc
        linkopts = linkopts + ["-lm", "-lpthread"],
        copts = _capricorn_copts() + copts,
        deps = deps + [
            "@googletest//:gtest",
            "@spdlog//:spdlog",
        ],
        defines = defines + ["THREADING"],
        linkstatic = True,
        **kwargs
    )
