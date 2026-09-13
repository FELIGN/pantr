#!/usr/bin/env python
"""Measure whether giving the n-d Bézier contraction a compile-time degree pays.

This is the measurement behind issue #379 and it closes ``design/simd.md``'s open
questions 1 and 2. It implements nothing: ``contract_leading_axis``
(``cpp/include/pantr/bezier/evaluate.hpp``) keeps both trip counts at runtime, and
every specialised variant below lives in a scratch translation unit that is
compiled, run and thrown away.

    cmake -S . -B build/gcc                      # once: for the fetched mdspan
    .venv/bin/python scripts/measure_bezier_degree_templating.py --cpu <n>

It needs no `pantr` import, so no `PYTHONPATH`; what it does need is a configured
build tree, because on a toolchain without `std::mdspan` the headers reach the
Kokkos reference implementation that `PantrDependencies.cmake` fetches. Without
one it says so and stops rather than measuring something else. `--cpu` pins the
timed runs to one core; without it the figures are a statement about the host's
load as much as about the code, and the script says that too.

``design/simd.md``'s *Auto-vectorize first* fixes the discipline this follows:
*auto-vectorize, measure, and reach for an explicit batch type only where the
compiler demonstrably fails*, on the evidence of a ``-fopt-info-vec-missed`` (GCC)
or ``-Rpass-missed=loop-vectorize`` (Clang) report, **not intuition**. So section 1
below is the compiler's own report and nothing else.

Every reference to that note here names a **section**, not a line. This script's own
first landing shifted its line numbers, so a range cited from here is one edit away
from pointing at the wrong paragraph, and nothing would notice.

The kernel is an AXPY::

    for term in 0 .. n_terms:            # n_terms == degree + 1
        for t in 0 .. stride:            # stride == the trailing block size
            out[t] += weights[term] * block[term * stride + t]

so there are **two** trip counts a specialisation could fix, and the ticket names
only one of them. Both are measured separately, because separating them is what
turns "does templating pay?" into an answer that says *what to template*.

What is measured
----------------

1. **Whether the loop auto-vectorises today**, from the compiler's own report, per
   compiler and per ISA level.
2. **Bit identity** (the ticket's `AC4`) between the shipped kernel and every
   specialised variant, over the whole sweep, at every compiler and ISA level.
   Plain unrolling does not reassociate, so under ``-ffp-contract=on`` and without
   ``-ffast-math`` a known trip count should execute the same operations in the
   same order; that is a claim to verify, not to assume.
3. **Speed**, per degree, per trailing block size, per scalar type.
4. **Instantiation cost** -- compile time and emitted text -- as a function of how
   many degrees are specialised, since it multiplies by scalar type and by the ISA
   variant count ``design/simd.md``'s *Shipping several ISA variants* plans for.
5. **Where the trailing block sizes actually are**, derived from the two schedules
   in ``evaluate.hpp`` rather than measured, so a recommendation is a statement
   about pantr's workload and not about the host this ran on.

Two controls, because a check that cannot fail confirms nothing
---------------------------------------------------------------

The failure mode of a benchmark like this is that the compiler constant-folds the
runtime trip counts at the harness's call site, so both rivals are the same code
and the speedup is one. Every trip count therefore crosses an ``asm volatile``
barrier before the shipped kernel sees it, and the ``folded`` variant -- the
shipped kernel handed literal constants -- is the control that says whether the
barrier held. If ``folded`` does not beat ``runtime`` the barrier leaked and no
figure in section 3 means anything.

The failure mode of the bit-identity table is the mirror of it: a comparison that
would report "identical" whatever happened. The ``reversed`` variant sums the same
terms in descending order, which is the same mathematics and a different summation
order, so it *must* disagree on most cases. If it does not, section 2 is vacuous.

Both controls are checked, and this script exits non-zero when either fails.

No figure from any run appears anywhere in this file, in ``evaluate.hpp``, or in
``design/simd.md``. A measured number in a permanent artifact rots while reading as
current; it belongs in this script's output, next to the commit, the compiler
version and the exact command, which :func:`report_provenance` prints.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final, NamedTuple

_STRIDES: Final = (1, 2, 3, 4, 6, 8, 9, 12, 16)
"""Trailing block sizes to sweep and to specialise.

Chosen from what the two schedules in `evaluate.hpp` can produce rather than as a
round sweep: the last direction of either schedule contracts with `stride` equal to
`cp_size`, which is `rank` plus one for a rational Bézier, so 1 to 4 covers scalar
through 3-D rational fields. The rest reach up to and past the point where the
vectorised main loop starts to run at all, which is where the answer changes."""

_MAX_TERMS: Final = 13
"""Largest `degree + 1` to specialise, so degrees 1 to 12.

pantr's degrees are low; going past this measures the cost of instantiation
without adding anything to the speed table, which has already flattened."""

_COST_DEPTHS: Final = (0, 3, 5, 9, 13)
"""Specialisation depths for the instantiation-cost sweep.

Zero is the shipped state -- the runtime kernel and nothing else -- and is the
baseline every other row is read against."""

_ISA_LEVELS: Final = ((), ("-march=x86-64-v3",), ("-march=native",))
"""Instruction-set levels to compile at.

The shipped build carries no `-march` at all (`cmake/PantrCompileOptions.cmake:132`
records why), so the first entry is the one that describes a wheel. The other two
are what a multi-ISA build would ship and are skipped on a host whose compiler
rejects them."""

_VARIANTS: Final = ("runtime", "folded", "terms", "stride", "both", "restrict", "reversed")
"""The rivals, in the order the harness prints them.

`runtime` is the shipped kernel. `terms` fixes the outer trip count at compile
time, which is the specialisation the ticket asks about; `stride` fixes the inner
one, which is `design/simd.md`'s open question 2 in another guise; `both` fixes
each. `folded` and `reversed` are the two controls the module docstring describes,
and `restrict` is the alternative that templates nothing."""

_SPECIALISED: Final = ("folded", "terms", "stride", "both", "restrict")
"""The variants that claim to compute exactly what the shipped kernel computes."""

_CONTROL: Final = "reversed"
"""The variant that must *not* agree with the shipped kernel."""

_CONTROL_MIN_TERMS: Final = 3
"""Where the control's verdict is taken from, and it is not from degree 1.

A two-term contraction sums `0 + p0` then `+ p1`, and IEEE addition **is**
commutative even though it is not associative, so ascending and descending order
give the same bits for every input -- unless the build fuses, in which case each
order leaves a different one of the two products unrounded and the floor
disappears. Either way the degree-1 cases say nothing about whether the
comparison discriminates, so the soundness test reads the rest."""

_KERNEL_SIGNATURE: Final = "void contract_leading_axis("
"""How to find the kernel in `evaluate.hpp`.

A hardcoded line number would go stale the first time anything above it moved,
and the failure would be silent: the vectoriser report would come back empty and
read as "the compiler said nothing" rather than as "the filter missed". So
:func:`kernel_line_span` locates the function and the report prints the line each
remark landed on."""

_LATTICE_SIZES: Final = (2, 4, 8, 16, 64)
"""Evaluation points per direction to sweep the derived work profile over.

Two is a per-element quadrature rule; 64 is a tabulation grid. The two ends
disagree about where the work is, and the honest statement needs both."""

_WORKLOADS: Final = (
    ("2-D surface in R^3, p=2", (3, 3), 3),
    ("2-D rational in R^3, p=3", (4, 4), 4),
    ("3-D scalar field, p=2", (3, 3, 3), 1),
    ("3-D map to R^3, p=3", (4, 4, 4), 3),
)
"""Control-net shapes and component counts to derive the work profile from.

Name, extents, `cp_size`. Chosen to span the `cp_size` values a caller actually
reaches: a scalar field, a planar-to-spatial map, and a rational one, whose
weight column puts `cp_size` one above the rank."""

_TWO_DIGIT: Final = 10.0
"""Where a ratio needs a second digit before the point, and so can spare one after."""

_BASE_FLAGS: Final = ("-O3", "-DNDEBUG", "-std=gnu++20", "-ffp-contract=on")
"""The numerical flags of a Release build of this project.

Taken from `compile_commands.json` of a configured tree: `-O3 -DNDEBUG` from
`CMAKE_BUILD_TYPE=Release`, the standard from `target_compile_features`, and
`-ffp-contract=on` from `pantr::core` in the top-level `CMakeLists.txt`. The
warning flags are `pantr::cxx_flags` and are deliberately left off: they are the
project's internal policy and change no code that runs.

`-ffast-math` and `-ffinite-math-only` appear nowhere here and must not: they
permit reassociation, which is a correctness change and would make the bit
identity in section 2 meaningless."""

_HARNESS = r"""
// Scratch measurement harness for the degree-templating question. Written here by
// scripts/measure_bezier_degree_templating.py, compiled with the project's own
// numerical flags, run, and deleted. Not library code and not built by CMake.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "pantr/bezier/evaluate.hpp"

namespace {

using pantr::bezier::detail::contract_leading_axis;

constexpr std::size_t kMaxTerms = PANTR_MEASURE_MAX_TERMS;
constexpr std::size_t kStrides[] = {PANTR_MEASURE_STRIDES};
constexpr std::size_t kNumStrides = sizeof(kStrides) / sizeof(kStrides[0]);
constexpr std::size_t kPool = 64;

// Keep a value out of the optimiser's reach. `evaluate` derives both trip counts
// from `net.shape()`, so it cannot fold them at the real call site; a harness that
// lets it fold them times the specialised variant twice and reports a speedup of
// one. The `folded` variant below is the control that says whether this held.
template <typename U>
[[gnu::always_inline]] inline U opaque(U value) {
    asm volatile("" : "+r"(value));
    return value;
}

[[gnu::always_inline]] inline void clobber() { asm volatile("" ::: "memory"); }

// ---- the rivals -----------------------------------------------------------
// Each body below is the body of contract_leading_axis. Only where a trip count
// comes from differs, except in contract_reversed, which is the control.

template <typename T, std::size_t NTerms>
void contract_terms(std::span<const T> weights, std::span<const T> block, std::size_t stride,
                    std::span<T> out) {
    for (std::size_t t = 0; t < stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < NTerms; ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * stride;
        for (std::size_t t = 0; t < stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

template <typename T, std::size_t Stride>
void contract_stride(std::span<const T> weights, std::span<const T> block, std::size_t,
                     std::span<T> out) {
    for (std::size_t t = 0; t < Stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < weights.size(); ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * Stride;
        for (std::size_t t = 0; t < Stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

template <typename T, std::size_t NTerms, std::size_t Stride>
void contract_both(std::span<const T> weights, std::span<const T> block, std::size_t,
                   std::span<T> out) {
    for (std::size_t t = 0; t < Stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < NTerms; ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * Stride;
        for (std::size_t t = 0; t < Stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

// The shipped kernel handed both trip counts as literals: the control for the
// barriers above, and the ceiling any specialisation could reach.
template <typename T, std::size_t NTerms, std::size_t Stride>
void contract_folded(std::span<const T> weights, std::span<const T> block, std::size_t,
                     std::span<T> out) {
    contract_leading_axis<T>(weights.first(NTerms), block, Stride, out);
}

// The alternative that templates nothing: promise the three blocks do not overlap
// and leave both trip counts at runtime. The __restrict has to sit on a helper's
// PARAMETERS -- on locals initialised from std::span::data() it changes no code,
// which the vectoriser report in section 1 shows.
template <typename T>
[[gnu::always_inline]] inline void contract_pointers(const T* __restrict weights,
                                                     const T* __restrict block,
                                                     std::size_t n_terms, std::size_t stride,
                                                     T* __restrict out) {
    for (std::size_t t = 0; t < stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < n_terms; ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * stride;
        for (std::size_t t = 0; t < stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

template <typename T>
void contract_restrict(std::span<const T> weights, std::span<const T> block, std::size_t stride,
                       std::span<T> out) {
    contract_pointers<T>(weights.data(), block.data(), weights.size(), stride, out.data());
}

// Descending term order: the same mathematics, a different summation order, and
// the positive control for the bit-identity table.
template <typename T>
void contract_reversed(std::span<const T> weights, std::span<const T> block, std::size_t stride,
                       std::span<T> out) {
    for (std::size_t t = 0; t < stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t k = weights.size(); k > 0; --k) {
        const std::size_t term = k - 1;
        const T weight = weights[term];
        const std::size_t offset = term * stride;
        for (std::size_t t = 0; t < stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

// ---- data -----------------------------------------------------------------

// A deterministic stream, so a rerun compares the same numbers and the parity
// table reproduces without shipping a data file.
std::uint64_t next(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

template <typename T>
std::vector<T> random_values(std::size_t count, std::uint64_t& state) {
    std::vector<T> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double unit = static_cast<double>(next(state) >> 11) / 9007199254740992.0;
        values[i] = static_cast<T>(unit - 0.5);
    }
    return values;
}

// ---- the timed region -----------------------------------------------------

// The kernel is a template argument rather than a function-pointer argument so
// that the dispatch picking it is resolved outside the timed region -- which is
// where a real specialisation would resolve it too, once per `evaluate` call
// rather than once per point.
template <typename T, auto Kernel>
double timed_loop(std::size_t n_terms, std::size_t stride, std::size_t reps, std::uint64_t seed) {
    std::uint64_t state = seed;
    const std::vector<T> block = random_values<T>(n_terms * stride, state);
    const std::vector<T> weights = random_values<T>(kPool * n_terms, state);
    std::vector<T> out(kPool * stride, T(0));

    const auto sweep = [&] {
        for (std::size_t p = 0; p < kPool; ++p) {
            // Every variant pays the same two barriers, so the only difference
            // between them is what the compiler knows about the trip counts.
            Kernel(std::span<const T>(&weights[p * n_terms], opaque(n_terms)),
                   std::span<const T>(block), opaque(stride),
                   std::span<T>(&out[p * stride], stride));
            clobber();
        }
    };
    sweep();

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < reps; ++r) {
        sweep();
    }
    const auto stop = std::chrono::steady_clock::now();

    double checksum = 0.0;
    for (const T value : out) {
        checksum += static_cast<double>(value);
    }
    if (checksum == 8675309.0) {
        std::fprintf(stderr, "impossible checksum\n");
    }
    const double elapsed = std::chrono::duration<double>(stop - start).count();
    return elapsed * 1e9 / static_cast<double>(reps * kPool);
}

template <typename T, auto Kernel>
void one_call(std::span<const T> weights, std::span<const T> block, std::size_t stride,
              std::span<T> out) {
    Kernel(std::span<const T>(weights.data(), opaque(weights.size())), block, opaque(stride), out);
}

// ---- dispatch over the instantiation grid ---------------------------------

enum class Variant : int { Runtime, Folded, Terms, Stride, Both, Restrict, Reversed };

constexpr Variant kVariants[] = {Variant::Runtime, Variant::Folded,   Variant::Terms,
                                 Variant::Stride,  Variant::Both,     Variant::Restrict,
                                 Variant::Reversed};
constexpr std::size_t kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::Runtime: return "runtime";
        case Variant::Folded: return "folded";
        case Variant::Terms: return "terms";
        case Variant::Stride: return "stride";
        case Variant::Both: return "both";
        case Variant::Restrict: return "restrict";
        case Variant::Reversed: return "reversed";
    }
    return "?";
}

// A visitor receives the chosen kernel as a template argument. The `if constexpr`
// chain is what turns a runtime (n_terms, stride) pair into that argument.
template <typename T, typename Visitor, std::size_t NTerms, std::size_t StrideIndex>
bool select_stride(Variant variant, Visitor& visitor);

template <typename T, typename Visitor, std::size_t NTerms>
bool select_terms(Variant variant, Visitor& visitor) {
    if constexpr (NTerms > kMaxTerms) {
        (void)variant;
        (void)visitor;
        return false;
    } else if (visitor.n_terms == NTerms) {
        return select_stride<T, Visitor, NTerms, 0>(variant, visitor);
    } else {
        return select_terms<T, Visitor, NTerms + 1>(variant, visitor);
    }
}

template <typename T, typename Visitor, std::size_t NTerms, std::size_t StrideIndex>
bool select_stride(Variant variant, Visitor& visitor) {
    if constexpr (StrideIndex == kNumStrides) {
        (void)variant;
        (void)visitor;
        return false;
    } else {
        constexpr std::size_t kStride = kStrides[StrideIndex];
        if (visitor.stride == kStride) {
            switch (variant) {
                case Variant::Folded:
                    visitor.template apply<&contract_folded<T, NTerms, kStride>>();
                    return true;
                case Variant::Terms:
                    visitor.template apply<&contract_terms<T, NTerms>>();
                    return true;
                case Variant::Stride:
                    visitor.template apply<&contract_stride<T, kStride>>();
                    return true;
                case Variant::Both:
                    visitor.template apply<&contract_both<T, NTerms, kStride>>();
                    return true;
                default:
                    return false;
            }
        }
        return select_stride<T, Visitor, NTerms, StrideIndex + 1>(variant, visitor);
    }
}

template <typename T, typename Visitor>
bool dispatch(Variant variant, Visitor& visitor) {
    switch (variant) {
        case Variant::Runtime:
            visitor.template apply<&contract_leading_axis<T>>();
            return true;
        case Variant::Restrict:
            visitor.template apply<&contract_restrict<T>>();
            return true;
        case Variant::Reversed:
            visitor.template apply<&contract_reversed<T>>();
            return true;
        default:
            return select_terms<T, Visitor, 2>(variant, visitor);
    }
}

template <typename T>
struct TimeVisitor {
    std::size_t n_terms;
    std::size_t stride;
    std::size_t reps;
    std::uint64_t seed;
    double nanoseconds = 0.0;

    template <auto Kernel>
    void apply() {
        nanoseconds = timed_loop<T, Kernel>(n_terms, stride, reps, seed);
    }
};

template <typename T>
struct CallVisitor {
    std::size_t n_terms;
    std::size_t stride;
    std::span<const T> weights;
    std::span<const T> block;
    std::span<T> out;

    template <auto Kernel>
    void apply() {
        one_call<T, Kernel>(weights, block, stride, out);
    }
};

// ---- the two modes --------------------------------------------------------

template <typename T>
void time_all(const char* dtype, std::size_t reps, std::size_t trials) {
    for (std::size_t n_terms = 2; n_terms <= kMaxTerms; ++n_terms) {
        for (std::size_t index = 0; index < kNumStrides; ++index) {
            // Trials outside variants, so the variants of one shape are interleaved
            // rather than run in blocks. A frequency or thermal ramp over the sweep
            // then hits every variant alike instead of penalising whichever came
            // last in the order.
            for (std::size_t trial = 0; trial < trials; ++trial) {
                for (const Variant variant : kVariants) {
                    TimeVisitor<T> visitor{n_terms, kStrides[index], reps, 0x9E3779B97F4A7C15ULL};
                    if (!dispatch<T>(variant, visitor)) {
                        continue;
                    }
                    std::printf("TIME %s %zu %zu %s %zu %.6f\n", dtype, n_terms, kStrides[index],
                                variant_name(variant), trial, visitor.nanoseconds);
                }
            }
        }
    }
}

template <typename T>
void parity_all(const char* dtype, std::size_t samples) {
    std::uint64_t state = 0xD1B54A32D192ED03ULL;
    for (std::size_t n_terms = 2; n_terms <= kMaxTerms; ++n_terms) {
        for (std::size_t index = 0; index < kNumStrides; ++index) {
            const std::size_t stride = kStrides[index];
            std::size_t compared[kNumVariants] = {};
            std::size_t identical[kNumVariants] = {};
            for (std::size_t sample = 0; sample < samples; ++sample) {
                const std::vector<T> block = random_values<T>(n_terms * stride, state);
                const std::vector<T> weights = random_values<T>(n_terms, state);
                std::vector<T> reference(stride, T(0));
                CallVisitor<T> base{n_terms, stride, std::span<const T>(weights),
                                    std::span<const T>(block), std::span<T>(reference)};
                dispatch<T>(Variant::Runtime, base);
                for (std::size_t v = 1; v < kNumVariants; ++v) {
                    std::vector<T> got(stride, T(0));
                    CallVisitor<T> rival{n_terms, stride, std::span<const T>(weights),
                                         std::span<const T>(block), std::span<T>(got)};
                    if (!dispatch<T>(kVariants[v], rival)) {
                        continue;
                    }
                    ++compared[v];
                    const bool same =
                        std::memcmp(got.data(), reference.data(), stride * sizeof(T)) == 0;
                    identical[v] += same ? 1U : 0U;
                }
            }
            for (std::size_t v = 1; v < kNumVariants; ++v) {
                std::printf("PARITY %s %zu %zu %s %zu %zu\n", dtype, n_terms, stride,
                            variant_name(kVariants[v]), compared[v], identical[v]);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "time";
    if (mode == "parity") {
        parity_all<float>("float32", argc > 2 ? std::stoul(argv[2]) : 64);
        parity_all<double>("float64", argc > 2 ? std::stoul(argv[2]) : 64);
        return 0;
    }
    const std::size_t reps = argc > 2 ? std::stoul(argv[2]) : 2000;
    const std::size_t trials = argc > 3 ? std::stoul(argv[3]) : 5;
    time_all<float>("float32", reps, trials);
    time_all<double>("float64", reps, trials);
    return 0;
}
"""
"""The scratch benchmark, compiled once per compiler and ISA level."""

_COST = r"""
// Instantiation-cost probe: what a shipped specialisation would add to one
// translation unit, and nothing else. No timing, no parity, no scaffolding, so
// the compile time and the emitted text are the specialisation's own.

#include <cstddef>
#include <span>

#include "pantr/bezier/evaluate.hpp"

namespace {

constexpr std::size_t kMaxTerms = PANTR_MEASURE_MAX_TERMS;
constexpr std::size_t kStrides[] = {PANTR_MEASURE_STRIDES};
constexpr std::size_t kNumStrides = sizeof(kStrides) / sizeof(kStrides[0]);

template <typename T, std::size_t NTerms, std::size_t Stride>
void contract_both(std::span<const T> weights, std::span<const T> block, std::span<T> out) {
    for (std::size_t t = 0; t < Stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < NTerms; ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * Stride;
        for (std::size_t t = 0; t < Stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

template <typename T, std::size_t NTerms, std::size_t StrideIndex>
bool select_stride(std::span<const T> w, std::span<const T> b, std::size_t stride,
                   std::span<T> out) {
    if constexpr (StrideIndex == kNumStrides) {
        (void)w;
        (void)b;
        (void)stride;
        (void)out;
        return false;
    } else if (stride == kStrides[StrideIndex]) {
        contract_both<T, NTerms, kStrides[StrideIndex]>(w, b, out);
        return true;
    } else {
        return select_stride<T, NTerms, StrideIndex + 1>(w, b, stride, out);
    }
}

template <typename T, std::size_t NTerms>
bool select_terms(std::span<const T> w, std::span<const T> b, std::size_t stride,
                  std::span<T> out) {
    if constexpr (NTerms > kMaxTerms) {
        (void)w;
        (void)b;
        (void)stride;
        (void)out;
        return false;
    } else if (w.size() == NTerms) {
        return select_stride<T, NTerms, 0>(w, b, stride, out);
    } else {
        return select_terms<T, NTerms + 1>(w, b, stride, out);
    }
}

}  // namespace

template <typename T>
void contract_dispatch(std::span<const T> w, std::span<const T> b, std::size_t stride,
                       std::span<T> out) {
    if (!select_terms<T, 2>(w, b, stride, out)) {
        pantr::bezier::detail::contract_leading_axis<T>(w, b, stride, out);
    }
}

template void contract_dispatch<float>(std::span<const float>, std::span<const float>, std::size_t,
                                       std::span<float>);
template void contract_dispatch<double>(std::span<const double>, std::span<const double>,
                                        std::size_t, std::span<double>);
"""
"""The instantiation-cost probe, compiled once per specialisation depth."""

_VECTOR_PROBE: Final = """
#include <cstddef>
#include <span>

#include "pantr/bezier/evaluate.hpp"

template void pantr::bezier::detail::contract_leading_axis<double>(
    std::span<const double>, std::span<const double>, std::size_t, std::span<double>);
template void pantr::bezier::detail::contract_leading_axis<float>(
    std::span<const float>, std::span<const float>, std::size_t, std::span<float>);
"""
"""A translation unit holding nothing but the shipped kernel, for the vectoriser
report. Instantiated explicitly so the compiler must emit it."""


class Toolchain(NamedTuple):
    """One compiler at one instruction-set level.

    Attributes:
        compiler (str): The driver, e.g. ``g++``.
        version (str): Its own first version line.
        isa (tuple[str, ...]): The ISA flags, empty for the shipped baseline.
        includes (tuple[str, ...]): The include flags the headers need.
    """

    compiler: str
    version: str
    isa: tuple[str, ...]
    includes: tuple[str, ...]

    @property
    def label(self) -> str:
        """Name this toolchain in one column.

        Returns:
            str: The driver and the ISA level, e.g. ``g++/x86-64-v3``.
        """
        level = self.isa[0].removeprefix("-march=") if self.isa else "baseline"
        return f"{Path(self.compiler).name}/{level}"

    def flags(self) -> list[str]:
        """Assemble the full flag list for a compile.

        Returns:
            list[str]: Include flags, the project's numerical flags, then the ISA.
        """
        return [*self.includes, *_BASE_FLAGS, *self.isa]


class Timing(NamedTuple):
    """The trials of one variant at one shape.

    Attributes:
        dtype (str): ``float32`` or ``float64``.
        n_terms (int): The outer trip count, ``degree + 1``.
        stride (int): The trailing block size.
        variant (str): One of :data:`_VARIANTS`.
        nanoseconds (tuple[float, ...]): One per trial, per call.
    """

    dtype: str
    n_terms: int
    stride: int
    variant: str
    nanoseconds: tuple[float, ...]

    @property
    def best(self) -> float:
        """Take the fastest trial.

        Returns:
            float: Nanoseconds per call. The minimum is the estimator here because
            every source of noise on a shared host adds time and none removes it.
        """
        return min(self.nanoseconds)

    @property
    def spread(self) -> float:
        """Report how much the trials disagreed.

        Returns:
            float: Slowest trial over fastest. Far above one means the host was
            busy and the row is a statement about the load.
        """
        return max(self.nanoseconds) / min(self.nanoseconds)


class Parity(NamedTuple):
    """The bit-identity tally of one variant against the shipped kernel.

    Attributes:
        variant (str): One of :data:`_VARIANTS`.
        compared (int): Cases run.
        identical (int): Cases whose every output byte matched.
    """

    variant: str
    compared: int
    identical: int


class Instantiation(NamedTuple):
    """What one specialisation depth costs a translation unit.

    Attributes:
        compiler (str): The driver.
        depth (int): Largest ``degree + 1`` specialised; 0 is the shipped state.
        seconds (float): Fastest of the repeated compiles.
        text_bytes (int): Emitted machine code, summed over every ``.text`` section
            because template instantiations land in per-symbol COMDAT sections.
    """

    compiler: str
    depth: int
    seconds: float
    text_bytes: int


def repo_root() -> Path:
    """Locate the checkout this script lives in.

    Returns:
        Path: The repository root, derived from this file rather than the working
        directory, which ``conda run`` is free to change.
    """
    return Path(__file__).resolve().parent.parent


def find_includes(root: Path, build_dir: Path | None) -> tuple[str, ...]:
    """Assemble the include flags ``evaluate.hpp`` needs.

    The headers need ``cpp/include`` and, on a toolchain without ``std::mdspan``,
    the Kokkos reference implementation that ``PantrDependencies.cmake`` fetches.
    Eigen is not needed: nothing on this include path reaches it.

    Args:
        root (Path): The repository root.
        build_dir (Path | None): A configured build tree to take the fetched mdspan
            from, or None to search ``build/*``.

    Returns:
        tuple[str, ...]: The flags. A toolchain that still cannot compile the
        headers with them is dropped by :func:`discover_toolchains`.
    """
    flags = (f"-I{root / 'cpp' / 'include'}",)
    candidates = [build_dir] if build_dir is not None else sorted((root / "build").glob("*"))
    for candidate in candidates:
        mdspan = candidate / "_deps" / "mdspan-src" / "include"
        if mdspan.is_dir():
            return (*flags, "-isystem", str(mdspan))
    # A toolchain with std::mdspan of its own needs nothing else; one without it
    # will fail to compile the probe and be dropped by discover_toolchains.
    return flags


def _compiles(command: list[str], source: str) -> str | None:
    """Try one compile and keep what the compiler said if it refused.

    Dropping an ISA level the compiler rejects is the normal case and needs no
    noise. But a probe that fails because the flags or the headers are broken takes
    the same path, and reporting the two identically sends a reader to configure a
    build tree they already have. So the diagnostic is kept and printed if nothing
    at all survives.

    Args:
        command (list[str]): The driver and its flags, without input or output.
        source (str): The translation unit.

    Returns:
        str | None: None when the compiler accepted it, else what it printed.
    """
    if not shutil.which(command[0]):
        return f"{command[0]}: not on PATH"
    with tempfile.TemporaryDirectory() as scratch:
        probe = Path(scratch) / "probe.cpp"
        probe.write_text(source)
        full = [*command, "-c", str(probe), "-o", str(Path(scratch) / "probe.o")]
        result = subprocess.run(full, capture_output=True, text=True, check=False)
        return None if result.returncode == 0 else result.stderr


def discover_toolchains(
    compilers: list[str], includes: tuple[str, ...]
) -> tuple[list[Toolchain], str]:
    """Find which of the requested compilers and ISA levels this host can build.

    Args:
        compilers (list[str]): The drivers to try.
        includes (tuple[str, ...]): The include flags from :func:`find_includes`.

    Returns:
        tuple[list[Toolchain], str]: One entry per compiler and accepted ISA level,
        and the first refusal any of them gave. A level the compiler rejects is
        dropped, which is what makes this run on a host that is not x86; the
        refusal comes back so the caller can print it if nothing survived.
    """
    found: list[Toolchain] = []
    refusal = ""
    for compiler in compilers:
        if not shutil.which(compiler):
            refusal = refusal or f"{compiler}: not on PATH"
            continue
        version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        for isa in _ISA_LEVELS:
            probe = [compiler, *includes, *_BASE_FLAGS, *isa]
            complaint = _compiles(probe, _VECTOR_PROBE)
            if complaint is None:
                found.append(Toolchain(compiler, version, isa, includes))
            else:
                refusal = refusal or complaint
    return found, refusal


def kernel_line_span(root: Path) -> range:
    """Locate ``contract_leading_axis`` in ``evaluate.hpp``, by reading it.

    Args:
        root (Path): The repository root.

    Returns:
        range: The lines the function body occupies, from its signature to the
        closing brace at column one.

    Raises:
        LookupError: If the function is not where this script expects it, which is
        a state to stop in rather than to report an empty table from.
    """
    header = (root / "cpp" / "include" / "pantr" / "bezier" / "evaluate.hpp").read_text()
    lines = header.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(_KERNEL_SIGNATURE):
            continue
        for end in range(index, len(lines)):
            if lines[end] == "}":
                return range(index + 1, end + 2)
    raise LookupError(f"no line of evaluate.hpp starts with {_KERNEL_SIGNATURE!r}")


def vectoriser_report(toolchain: Toolchain) -> list[str]:
    """Ask the compiler what it did to the contraction's loops.

    ``design/simd.md``'s *Auto-vectorize first* names these two reports as the
    evidence to act on. GCC writes its report to a file; Clang emits remarks on
    stderr.

    Args:
        toolchain (Toolchain): The compiler and ISA level to ask.

    Returns:
        list[str]: The remarks that landed on the kernel's own lines, deduplicated
        and each prefixed with the line it came from.
    """
    is_gcc = "clang" not in Path(toolchain.compiler).name
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "probe.cpp"
        source.write_text(_VECTOR_PROBE)
        report = Path(scratch) / "report.txt"
        extra = (
            [f"-fopt-info-vec-all={report}"]
            if is_gcc
            else ["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize"]
        )
        command = [
            toolchain.compiler, *toolchain.flags(), *extra,
            "-c", str(source), "-o", str(Path(scratch) / "probe.o"),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        text = report.read_text() if is_gcc and report.is_file() else result.stderr

    wanted = kernel_line_span(repo_root())
    seen: list[str] = []
    for line in text.splitlines():
        match = re.search(r"evaluate\.hpp:(\d+):\d+:\s*(.*)$", line)
        if match is None or int(match.group(1)) not in wanted:
            continue
        remark = re.sub(r"\s*\[-R\S+\]\s*$", "", match.group(2)).strip()
        if not remark.startswith(("optimized", "missed", "remark")):
            continue
        entry = f"{match.group(1)}: {remark}"
        if entry not in seen:
            seen.append(entry)
    return seen


def build_harness(toolchain: Toolchain, scratch: Path, max_terms: int) -> Path | None:
    """Compile the scratch benchmark.

    Args:
        toolchain (Toolchain): The compiler and ISA level.
        scratch (Path): Where to put the source and the program.
        max_terms (int): Largest ``degree + 1`` to instantiate.

    Returns:
        Path | None: The program, or None if it did not build.
    """
    source = scratch / f"harness_{toolchain.label.replace('/', '_')}.cpp"
    source.write_text(_HARNESS)
    program = source.with_suffix("")
    command = [
        toolchain.compiler, *toolchain.flags(),
        f"-DPANTR_MEASURE_MAX_TERMS={max_terms}",
        f"-DPANTR_MEASURE_STRIDES={','.join(str(s) for s in _STRIDES)}",
        "-o", str(program), str(source),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        print(f"  {toolchain.label}: did not build\n{result.stderr[:2000]}")
        return None
    return program


def run_harness(program: Path, arguments: list[str], cpu: int | None) -> list[list[str]]:
    """Run the benchmark and split its output into fields.

    Args:
        program (Path): The compiled harness.
        arguments (list[str]): Its mode and sizing arguments.
        cpu (int | None): A CPU to pin to, or None to inherit the affinity mask.

    Returns:
        list[list[str]]: One list of fields per output line.
    """
    command = [*_pin(cpu), str(program), *arguments]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return [line.split() for line in result.stdout.splitlines()]


def _pin(cpu: int | None) -> list[str]:
    """Build the affinity prefix for a run.

    Args:
        cpu (int | None): The CPU to pin to, or None.

    Returns:
        list[str]: ``taskset -c N`` when asked for and available, else empty.
    """
    if cpu is None or not shutil.which("taskset"):
        return []
    return ["taskset", "-c", str(cpu)]


def collect_timings(rows: list[list[str]]) -> dict[tuple[str, int, int, str], Timing]:
    """Gather the harness's timing lines into one record per variant and shape.

    Args:
        rows (list[list[str]]): The split output lines.

    Returns:
        dict[tuple[str, int, int, str], Timing]: Keyed by dtype, terms, stride and
        variant.
    """
    trials: dict[tuple[str, int, int, str], list[float]] = {}
    for row in rows:
        if row[0] != "TIME":
            continue
        key = (row[1], int(row[2]), int(row[3]), row[4])
        trials.setdefault(key, []).append(float(row[6]))
    return {key: Timing(*key, tuple(values)) for key, values in trials.items()}


def collect_parity(rows: list[list[str]], min_terms: int = 0) -> dict[str, Parity]:
    """Total the harness's bit-identity lines per variant.

    Args:
        rows (list[list[str]]): The split output lines.
        min_terms (int): Ignore shapes below this term count. Defaults to 0, which
            keeps everything; :data:`_CONTROL_MIN_TERMS` is what the control's own
            verdict is taken over.

    Returns:
        dict[str, Parity]: One tally per variant, over the shapes kept.
    """
    totals: dict[str, tuple[int, int]] = {}
    for row in rows:
        if row[0] != "PARITY" or int(row[2]) < min_terms:
            continue
        compared, identical = totals.get(row[4], (0, 0))
        totals[row[4]] = (compared + int(row[5]), identical + int(row[6]))
    return {name: Parity(name, *counts) for name, counts in totals.items()}


def measure_instantiation(toolchain: Toolchain, scratch: Path, repeats: int) -> list[Instantiation]:
    """Compile the cost probe at each specialisation depth and weigh the result.

    Args:
        toolchain (Toolchain): The compiler; the ISA level is used as given.
        scratch (Path): Where to put the source and the objects.
        repeats (int): Compiles per depth, of which the fastest is reported.

    Returns:
        list[Instantiation]: One row per depth.
    """
    source = scratch / "cost.cpp"
    source.write_text(_COST)
    rows: list[Instantiation] = []
    for depth in _COST_DEPTHS:
        obj = scratch / f"cost_{Path(toolchain.compiler).name}_{depth}.o"
        command = [
            toolchain.compiler, *toolchain.flags(),
            f"-DPANTR_MEASURE_MAX_TERMS={depth}",
            f"-DPANTR_MEASURE_STRIDES={','.join(str(s) for s in _STRIDES)}",
            "-c", str(source), "-o", str(obj),
        ]  # fmt: skip
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            best = min(best, time.perf_counter() - start)
            if result.returncode:
                print(f"  {toolchain.compiler} depth {depth}: did not build")
                return rows
        rows.append(Instantiation(toolchain.compiler, depth, best, _text_bytes(obj)))
    return rows


def _text_bytes(obj: Path) -> int:
    """Sum every executable section of an object file.

    A template instantiation lands in its own ``.text._Z...`` COMDAT section, so
    reading ``.text`` alone reports zero on exactly the builds that instantiate
    most.

    Args:
        obj (Path): The object file.

    Returns:
        int: Total bytes across sections whose name starts with ``.text``, or 0
        when ``size`` is unavailable.
    """
    if not shutil.which("size"):
        return 0
    listing = subprocess.run(
        ["size", "-A", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    sections = re.findall(r"^\.text\S*\s+(\d+)", listing, flags=re.MULTILINE)
    return sum(int(size) for size in sections)


def contraction_calls(
    net_shape: tuple[int, ...], lattice: tuple[int, ...], cp_size: int
) -> list[tuple[int, int, int]]:
    """Enumerate the contractions one ``evaluate_on_lattice`` performs.

    Read straight off the schedule in ``evaluate.hpp``: direction ``d`` contracts
    with ``outer * m_d`` calls over a trailing block of the extents still to come,
    times the component count.

    Args:
        net_shape (tuple[int, ...]): The control net's extents, one per direction.
        lattice (tuple[int, ...]): Points per direction.
        cp_size (int): Stored components, the homogeneous weight included.

    Returns:
        list[tuple[int, int, int]]: ``(calls, n_terms, stride)`` per direction.
    """
    extents = list(net_shape)
    schedule: list[tuple[int, int, int]] = []
    for direction, points in enumerate(lattice):
        outer = math.prod(extents[:direction])
        inner = math.prod(extents[direction + 1 :]) * cp_size
        schedule.append((outer * points, extents[direction], inner))
        extents[direction] = points
    return schedule


def report_provenance(toolchains: list[Toolchain], cpu: int | None) -> None:
    """Print what every figure below is pinned to.

    A measurement without its commit, its compiler and its host is not
    reproducible, and a number nobody can re-derive is a number nobody can refute.

    Args:
        toolchains (list[Toolchain]): The toolchains that will be used.
        cpu (int | None): The CPU the timed runs are pinned to, if any.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root(),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a git tree
        commit = "unknown"
    print(f"commit {commit}, python {sys.version.split()[0]}, {platform.platform()}")
    print(f"host {platform.processor() or platform.machine()}")
    seen: set[str] = set()
    for toolchain in toolchains:
        if toolchain.version not in seen:
            seen.add(toolchain.version)
            print(f"  {toolchain.version}")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    where = f"pinned to CPU {cpu}" if cpu is not None else f"{len(affinity)} CPUs in the mask"
    print(f"  timed runs {where}, 1-minute load average {load:.1f}")
    if cpu is None:
        print("  A timing taken on a shared host without --cpu is a statement about the load")
        print("  as much as about the code. Read the spread column before quoting a ratio.")
    print(f"  invocation: {' '.join([Path(sys.executable).name, *sys.argv])}")


def report_vectorisation(toolchains: list[Toolchain]) -> None:
    """Print the compilers' own reports on the contraction's loops.

    Args:
        toolchains (list[Toolchain]): The toolchains to ask.
    """
    print("\n1. Does the contraction auto-vectorise today?")
    print("   " + "-" * 72)
    print("   Lines are evaluate.hpp's: the zeroing loop, the term loop, the AXPY.")
    for toolchain in toolchains:
        print(f"\n   {toolchain.label}")
        remarks = vectoriser_report(toolchain)
        if not remarks:
            print("     no remark landed on the kernel's lines")
        for remark in remarks:
            print(f"     {remark}")


def report_parity(
    results: dict[str, dict[str, Parity]], controls: dict[str, dict[str, Parity]]
) -> bool:
    """Print the bit-identity table and say whether it discriminates.

    Args:
        results (dict[str, dict[str, Parity]]): Tallies over the whole sweep, keyed
            by toolchain label then by variant.
        controls (dict[str, dict[str, Parity]]): The same, restricted to shapes of
            at least :data:`_CONTROL_MIN_TERMS` terms, which is where the control's
            verdict is taken from and why.

    Returns:
        bool: True when every specialised variant matched every case at **every**
        toolchain, and the control disagreed on the majority of the shapes where it
        is free to. The second half is what makes the first half mean anything.
    """
    print("\n2. Bit identity against the shipped kernel (AC4)")
    print("   " + "-" * 72)
    print(f"   {'toolchain':<22}" + "".join(f"{name:>14}" for name in _VARIANTS[1:]))
    sound = True
    for label, tallies in results.items():
        cells = []
        for name in _VARIANTS[1:]:
            tally = tallies.get(name)
            cells.append("-" if tally is None else f"{tally.identical}/{tally.compared}")
        print(f"   {label:<22}" + "".join(f"{cell:>14}" for cell in cells))
        sound &= all(
            (tally := tallies.get(name)) is not None and tally.identical == tally.compared
            for name in _SPECIALISED
        )
        control = controls.get(label, {}).get(_CONTROL)
        sound &= control is not None and control.identical * 2 < control.compared
    total = sum(
        tally.compared
        for tallies in results.values()
        for name, tally in tallies.items()
        if name in _SPECIALISED
    )
    print(f"\n   {total} comparisons over the specialised variants.")
    print(f"   `{_CONTROL}` is the control: the same terms in descending order, which is")
    print("   the same mathematics and a different summation order. It has to disagree,")
    print("   or the whole column is a check that could not have failed.")
    print(f"   Its verdict is taken over shapes of {_CONTROL_MIN_TERMS} terms and up, and")
    print("   only there: two-term sums commute exactly, so at degree 1 the control is")
    print("   pinned to agree on a build that does not fuse and is free on one that does.")
    return sound


_SHOWN: Final = ("terms", "stride", "both", "restrict", "folded")
"""The columns of the speed table, the shipped kernel being the unit."""


def speedups(
    timings: dict[tuple[str, int, int, str], Timing], dtype: str, stride: int, variant: str
) -> list[float]:
    """Collect one variant's speedup at one block size, over every degree.

    Args:
        timings (dict[tuple[str, int, int, str], Timing]): One toolchain's runs.
        dtype (str): ``float32`` or ``float64``.
        stride (int): The trailing block size.
        variant (str): The rival.

    Returns:
        list[float]: The shipped kernel's time over the rival's, one per degree.
    """
    ratios = []
    for n_terms in sorted({key[1] for key in timings if key[0] == dtype}):
        base = timings.get((dtype, n_terms, stride, "runtime"))
        rival = timings.get((dtype, n_terms, stride, variant))
        if base is not None and rival is not None:
            ratios.append(base.best / rival.best)
    return ratios


def _median(values: list[float]) -> float:
    """Take the middle value.

    Args:
        values (list[float]): A non-empty list.

    Returns:
        float: The median, low value of the two for an even count.
    """
    return sorted(values)[len(values) // 2]


def report_speed(
    label: str, timings: dict[tuple[str, int, int, str], Timing], detail: bool
) -> bool:
    """Print the speedup table for one toolchain and say whether it is meaningful.

    The degree axis is collapsed to its median, because that is the finding: the
    speedup tracks the trailing block size and not the degree. The next table
    quantifies how much that collapse throws away, so the claim is checkable rather
    than asserted.

    Args:
        label (str): The toolchain's label.
        timings (dict[tuple[str, int, int, str], Timing]): Its measurements.
        detail (bool): Print every degree separately as well.

    Returns:
        bool: True when the ``folded`` control beat this toolchain's shipped kernel
        somewhere in its sweep, which is what says its barrier held. Somewhere and
        not everywhere: at a block size wide enough that the trip counts cost
        nothing, the control has nothing to win and a tie is the correct outcome.
    """
    # The control has to clear this run's own noise floor rather than a fixed
    # factor: the widest disagreement between trials of the shipped kernel is the
    # smallest ratio this measurement can tell apart from nothing happening.
    noise = max(
        (value.spread for key, value in timings.items() if key[3] == "runtime"), default=1.0
    )
    print(f"\n   {label}: speedup over the shipped kernel, median over degrees 1 to 12")
    print(
        f"     {'dtype':<9}{'stride':>7}"
        + "".join(f"{name:>10}" for name in _SHOWN)
        + f"{'ns/call':>10}{'spread':>8}"
    )
    folded_held = False
    for dtype in ("float64", "float32"):
        for stride in _STRIDES:
            cells = []
            for name in _SHOWN:
                ratios = speedups(timings, dtype, stride, name)
                cells.append("-" if not ratios else f"{_median(ratios):.2f}")
                if name == "folded" and ratios and max(ratios) > noise:
                    folded_held = True
            shipped = [
                value
                for key, value in timings.items()
                if key[0] == dtype and key[2] == stride and key[3] == "runtime"
            ]
            worst = max((value.spread for value in shipped), default=float("nan"))
            span = (
                f"{min(v.best for v in shipped):.0f}-{max(v.best for v in shipped):.0f}"
                if shipped
                else "-"
            )
            print(
                f"     {dtype:<9}{stride:>7}"
                + "".join(f"{cell:>10}" for cell in cells)
                + f"{span:>10}{worst:>8.2f}"
            )
    report_variation(timings)
    if detail:
        report_every_degree(label, timings)
    if not folded_held:
        print("     WARNING: the folded control never beat this toolchain's shipped kernel.")
        print("     Its barrier leaked; nothing in this table is two different functions.")
    return folded_held


def _range(values: list[float]) -> str:
    """Format the extremes of a set of ratios into one narrow cell.

    Args:
        values (list[float]): The ratios, possibly empty.

    Returns:
        str: ``low-high``, kept inside the table's column width by dropping a
        decimal above ten, where the second one says nothing anyway.
    """
    if not values:
        return "-"
    low, high = min(values), max(values)
    return f"{low:.2f}-{high:.1f}" if high >= _TWO_DIGIT else f"{low:.2f}-{high:.2f}"


def report_variation(timings: dict[tuple[str, int, int, str], Timing]) -> None:
    """Say what the median above hides.

    Two lines, because they answer different questions. The **range** is the honest
    single statement about a variant: what it did at its best and at its worst over
    the whole sweep. The **degree ratio** is the check on collapsing the degree
    axis: the largest gap between the best and the worst degree *within one block
    size and one scalar type*. Near one there means the degree is not the variable
    and the median lost nothing.

    Args:
        timings (dict[tuple[str, int, int, str], Timing]): One toolchain's runs.
    """
    ranges = []
    degrees = []
    for name in _SHOWN:
        everything: list[float] = []
        worst = 1.0
        for dtype in ("float64", "float32"):
            for stride in _STRIDES:
                ratios = speedups(timings, dtype, stride, name)
                if ratios:
                    everything.extend(ratios)
                    worst = max(worst, max(ratios) / min(ratios))
        ranges.append(_range(everything))
        degrees.append(f"{worst:.2f}")
    print(f"     {'whole sweep':<16}" + "".join(f"{cell:>10}" for cell in ranges))
    print(f"     {'degree ratio':<16}" + "".join(f"{cell:>10}" for cell in degrees))


def report_every_degree(label: str, timings: dict[tuple[str, int, int, str], Timing]) -> None:
    """Print the uncollapsed table, one row per degree and block size.

    Args:
        label (str): The toolchain's label.
        timings (dict[tuple[str, int, int, str], Timing]): Its measurements.
    """
    print(f"     {label}: every degree")
    print(
        f"     {'dtype':<9}{'degree':>7}{'stride':>7}"
        + "".join(f"{name:>10}" for name in _SHOWN)
        + f"{'ns/call':>10}{'spread':>8}"
    )
    for dtype in ("float64", "float32"):
        for n_terms in sorted({key[1] for key in timings if key[0] == dtype}):
            for stride in _STRIDES:
                base = timings.get((dtype, n_terms, stride, "runtime"))
                if base is None:
                    continue
                cells = [
                    "-"
                    if (rival := timings.get((dtype, n_terms, stride, name))) is None
                    else f"{base.best / rival.best:.2f}"
                    for name in _SHOWN
                ]
                print(
                    f"     {dtype:<9}{n_terms - 1:>7}{stride:>7}"
                    + "".join(f"{cell:>10}" for cell in cells)
                    + f"{base.best:>10.1f}{base.spread:>8.2f}"
                )


def report_instantiation(rows: list[Instantiation]) -> None:
    """Print what specialising costs the build.

    Args:
        rows (list[Instantiation]): One row per compiler and depth.
    """
    print("\n4. What the instantiation costs one translation unit")
    print("   " + "-" * 72)
    print(f"     {'compiler':<12}{'max degree':>12}{'compile s':>12}{'text bytes':>13}")
    for row in rows:
        degree = "none" if row.depth == 0 else str(row.depth - 1)
        print(
            f"     {Path(row.compiler).name:<12}{degree:>12}{row.seconds:>12.2f}"
            f"{row.text_bytes:>13}"
        )
    print(f"\n   Depth 0 is the shipped state. Each row above it instantiates {len(_STRIDES)}")
    print("   trailing block sizes per degree per scalar type, and the whole table")
    print("   multiplies by the ISA-variant count design/simd.md plans for under")
    print('   "Shipping several ISA variants".')


def report_where_the_strides_are() -> None:
    """Derive which trailing block sizes pantr's own schedules actually produce.

    Not measured: read off the two schedules in ``evaluate.hpp`` and counted. A
    threshold derived this way is a statement about the workload and transfers to
    a machine with a different core count, which a fitted one does not.

    The lattice is swept rather than fixed, because the answer depends on it and a
    single column would read as a stronger claim than the derivation supports: the
    last direction dominates only once there are many more evaluation points per
    direction than control points, which is the regime where the cost is worth
    caring about in the first place.
    """
    print("\n5. Where the trailing block sizes actually are (derived, not measured)")
    print("   " + "-" * 72)
    print("   Both schedules contract the last direction with stride == cp_size.")
    print("   Element-operations are calls * n_terms * stride; the columns are the")
    print("   share of them, for that many evaluation points per direction.")
    header = "".join(f"{f'm={points}':>8}" for points in _LATTICE_SIZES)
    print(f"\n     {'case':<36}{'stride':>8}{header}")
    for name, shape, cp_size in _WORKLOADS:
        for index, stride in enumerate(_strides_of(shape, cp_size)):
            shares = "".join(
                f"{_share_at(shape, cp_size, points, index):>8.0%}" for points in _LATTICE_SIZES
            )
            print(f"     {name if index == 0 else '':<36}{stride:>8}{shares}")


def _strides_of(shape: tuple[int, ...], cp_size: int) -> list[int]:
    """List the trailing block sizes one evaluation walks through.

    Args:
        shape (tuple[int, ...]): The control net's extents.
        cp_size (int): Stored components, the homogeneous weight included.

    Returns:
        list[int]: One per direction, descending to ``cp_size``.
    """
    return [stride for _calls, _terms, stride in contraction_calls(shape, shape, cp_size)]


def _share_at(shape: tuple[int, ...], cp_size: int, points: int, direction: int) -> float:
    """Compute one direction's share of the element-operations.

    Args:
        shape (tuple[int, ...]): The control net's extents.
        cp_size (int): Stored components.
        points (int): Evaluation points per direction.
        direction (int): Which contraction of the schedule.

    Returns:
        float: Its element-operations over the whole evaluation's.
    """
    schedule = contraction_calls(shape, (points,) * len(shape), cp_size)
    work = [calls * terms * stride for calls, terms, stride in schedule]
    return work[direction] / sum(work)


def report_verdict(sound_parity: bool, folded_held: bool) -> None:
    """Print what the two controls say about everything above.

    Args:
        sound_parity (bool): Whether section 2 discriminated.
        folded_held (bool): Whether the `folded` control beat the shipped kernel at
            **every** toolchain, each being its own compilation and so its own claim.
    """
    print("\n6. The controls")
    print("   " + "-" * 72)
    if sound_parity:
        print("   Bit identity holds for every specialised variant, and the reversed-order")
        print("   control disagrees, so the comparison could have failed and did not.")
    else:
        print("   Bit identity did NOT hold, or the reversed-order control agreed. Either")
        print("   way section 2 is not the claim it looks like; read the table.")
    if folded_held:
        print("   The folded control beats the shipped kernel at every toolchain, so every")
        print("   barrier held and the runtime variant really did keep its trip counts at")
        print("   runtime.")
    else:
        print("   At some toolchain the folded control did NOT beat the shipped kernel. Its")
        print("   barrier leaked, and there every speedup in section 3 is one measurement of")
        print("   one function, taken twice. Section 3 says which toolchain.")


def parse_arguments() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(description="Measure degree templating for issue #379.")
    parser.add_argument(
        "--compiler", action="append", metavar="CXX", help="a C++ driver to measure with"
    )
    parser.add_argument("--build-dir", type=Path, help="configured build tree to take mdspan from")
    parser.add_argument("--reps", type=int, default=2000, help="timed sweeps per trial")
    parser.add_argument("--trials", type=int, default=5, help="trials per variant and shape")
    parser.add_argument("--samples", type=int, default=64, help="parity samples per shape")
    parser.add_argument("--compiles", type=int, default=3, help="compiles per instantiation depth")
    parser.add_argument("--cpu", type=int, help="pin the timed runs to this CPU")
    parser.add_argument(
        "--max-terms", type=int, default=_MAX_TERMS, help="largest degree + 1 to specialise"
    )
    parser.add_argument(
        "--detail", action="store_true", help="print the speed table one row per degree too"
    )
    return parser.parse_args()


class Run(NamedTuple):
    """Everything one toolchain produced.

    Attributes:
        toolchain (Toolchain): The compiler and ISA level.
        parity (dict[str, Parity]): Bit-identity tallies per variant.
        control (dict[str, Parity]): The same, over the shapes where the control is
            free to disagree. See :data:`_CONTROL_MIN_TERMS`.
        timings (dict[tuple[str, int, int, str], Timing]): The timed sweep.
    """

    toolchain: Toolchain
    parity: dict[str, Parity]
    control: dict[str, Parity]
    timings: dict[tuple[str, int, int, str], Timing]


def measure(
    toolchains: list[Toolchain], scratch: Path, args: argparse.Namespace
) -> tuple[list[Run], list[Instantiation]]:
    """Build and run everything, printing only what went wrong.

    Args:
        toolchains (list[Toolchain]): The toolchains to measure with.
        scratch (Path): A directory for sources, objects and programs.
        args (argparse.Namespace): The parsed command line.

    Returns:
        tuple[list[Run], list[Instantiation]]: One run per toolchain that built,
        and the instantiation-cost rows for the baseline toolchains.
    """
    runs: list[Run] = []
    for toolchain in toolchains:
        program = build_harness(toolchain, scratch, args.max_terms)
        if program is None:
            continue
        rows_out = run_harness(program, ["parity", str(args.samples)], args.cpu)
        timings = collect_timings(
            run_harness(program, ["time", str(args.reps), str(args.trials)], args.cpu)
        )
        runs.append(
            Run(
                toolchain,
                collect_parity(rows_out),
                collect_parity(rows_out, _CONTROL_MIN_TERMS),
                timings,
            )
        )
    rows: list[Instantiation] = []
    for toolchain in toolchains:
        if not toolchain.isa:
            rows.extend(measure_instantiation(toolchain, scratch, args.compiles))
    return runs, rows


def main() -> int:
    """Run every measurement and report.

    Returns:
        int: 0 when both controls held and every specialised variant reproduced the
        shipped kernel bit for bit, 1 otherwise.
    """
    args = parse_arguments()
    includes = find_includes(repo_root(), args.build_dir)
    toolchains, refusal = discover_toolchains(args.compiler or ["g++", "clang++"], includes)
    if not toolchains:
        print("No C++ compiler here can build the headers with these flags.", file=sys.stderr)
        print(f"Include flags tried: {' '.join(includes)}", file=sys.stderr)
        print(
            "Usually that means no configured build tree, so the fetched mdspan is", file=sys.stderr
        )
        print("missing: run cmake -S . -B build/gcc once, or pass --build-dir.", file=sys.stderr)
        print(f"\nWhat the compiler actually said:\n{refusal.strip()[:2000]}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        runs, rows = measure(toolchains, Path(scratch), args)

    print("Bézier contraction: is a compile-time degree worth it? (issue #379)")
    report_provenance(toolchains, args.cpu)
    report_vectorisation(toolchains)
    sound_parity = report_parity(
        {run.toolchain.label: run.parity for run in runs},
        {run.toolchain.label: run.control for run in runs},
    )

    print("\n3. Speed, per trailing block size")
    print("   " + "-" * 72)
    print("   `terms` is the ticket's question: the degree fixed at compile time.")
    print("   `stride` is design/simd.md's open question 2: the block width fixed.")
    print("   `folded` is the ceiling and the control; `restrict` templates nothing.")
    # Every toolchain is its own compilation, so every one has to demonstrate its
    # own barrier held. One that did is not evidence about the other five, and an
    # `or` here would print the reassuring verdict on the strength of a single cell.
    folded_held = True
    for run in runs:
        # Not `all()` over a generator: report_speed prints, so short-circuiting
        # would drop the remaining toolchains' tables from the output.
        held = report_speed(run.toolchain.label, run.timings, args.detail)
        folded_held = folded_held and held

    report_instantiation(rows)
    report_where_the_strides_are()
    report_verdict(sound_parity, folded_held)
    return 0 if sound_parity and folded_held else 1


if __name__ == "__main__":
    raise SystemExit(main())
