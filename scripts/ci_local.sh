#!/usr/bin/env bash
#
# The full local check for the C++ prototype, run before pushing.
#
# CI for this branch is workflow_dispatch-only (.github/workflows/cpp.yaml), so
# this script -- not GitHub -- is where the prototype is actually checked. It is
# written to be the thing whose output answers the questions the prototype
# exists to ask, rather than a wrapper that prints "ok".
#
# Usage:
#     scripts/ci_local.sh              # everything
#     scripts/ci_local.sh gates        # only the configure-time gate checks
#     scripts/ci_local.sh cxx          # only the two C++ toolchains
#     scripts/ci_local.sh python       # only the extension, parity and backends
#     scripts/ci_local.sh discipline   # only the source-level rule guards
#     scripts/ci_local.sh splitmode    # only the nanobind split-mode probe
#
# Run it inside the pantr conda environment (`conda activate pantr`), which is
# where the two compilers and the pinned CMake live.
#
# ---------------------------------------------------------------------------
# The CPU budget
# ---------------------------------------------------------------------------
#
# This is a shared 160-CPU server and the work is capped at 20 CPUs. The cap is
# already enforced twice -- `taskset -cp 0-19 $$` in ~/.bashrc, inherited by
# every child, and CMAKE_BUILD_PARALLEL_LEVEL / CTEST_PARALLEL_LEVEL in the
# environment's activate.d. This script therefore does NOT set -j itself, with
# one exception: the CMakePresets carry an explicit `jobs: 20`, because a preset
# is also read by editors and CI images that never sourced the activate.d.
#
# Never add `-j$(nproc)` or `pytest -n auto` here: both size themselves from
# os.cpu_count(), which reports 160 and steps straight over the affinity mask.

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

readonly VENV="$ROOT/.venv"

# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
#
# Every check appends one line to a summary printed at the end. A run that
# scrolls a thousand lines of compiler output and finishes with a bare exit code
# tells the reader nothing about which of the prototype's questions is still
# open, which is the whole point of running it.

declare -a RESULTS=()
FAILED=0

# Per-run log directory.
#
# These used to be fixed paths under /tmp (`pantr_ci_step.log` and friends). This
# is a SHARED machine and several agents run this script at once, so two runs
# overwrote each other's logs and the tail printed after a failure could belong to
# the other run. A run-scoped directory removes the question.
LOGDIR="$(mktemp -d -t pantr_ci_XXXXXXXX)"
readonly LOGDIR

# Scratch directory for the throwaway configure trees the gate checks build.
# Empty until `gates` runs; the trap tolerates that.
tmp=""

# The logs are kept on failure and removed on success: they are worth reading
# exactly when something went wrong, and a path printed after the directory has
# been deleted is worse than no path at all.
cleanup() {
    [[ -n "$tmp" ]] && rm -rf "$tmp"
    if [[ "$FAILED" -eq 0 ]]; then
        rm -rf "$LOGDIR"
    else
        printf '\nFull logs for this run: %s\n' "$LOGDIR"
    fi
    return 0
}
trap cleanup EXIT

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

record() {
    local status="$1" name="$2" detail="${3:-}"
    RESULTS+=("$(printf '%-7s %-38s %s' "$status" "$name" "$detail")")
    [[ "$status" == "FAIL" ]] && FAILED=1
    return 0
}

# Run a command, record PASS/FAIL, never abort the whole script on failure:
# a run that stops at the first problem hides the other five.
check() {
    local name="$1"; shift
    if "$@" >"$LOGDIR/step.log" 2>&1; then
        record PASS "$name"
    else
        record FAIL "$name" "(see output above)"
        tail -40 "$LOGDIR/step.log"
    fi
}

# --------------------------------------------------------------------------
# gates -- the configure-time behaviour of design/toolchain_requirements.md
# --------------------------------------------------------------------------
#
# Each of these asserts that a gate FIRES, not merely that a good toolchain
# builds. A gate nobody has ever seen reject anything is not known to be a gate.

gates() {
    step "Configure-time gates"

    # Script-scoped and cleaned by the EXIT trap, NOT function-local with a
    # RETURN trap: a RETURN trap fires again as the script itself unwinds, by
    # which point a `local` is out of scope, and `set -u` then kills the run
    # with "tmp: unbound variable" AFTER the summary has printed "All checks
    # passed". The exit status said 1 while the output said 0.
    tmp="$(mktemp -d)"

    # The -ffast-math refusal must actually refuse. This is the check most
    # likely to rot: the guard lives in cmake/PantrCompileOptions.cmake and
    # matches flag strings, so a reformat of that loop can silently stop
    # matching while the build still succeeds.
    if cmake -S . -B "$tmp/fastmath" -G Ninja \
             -DCMAKE_CXX_FLAGS="-ffast-math" \
             -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF \
             >"$tmp/fastmath.log" 2>&1; then
        record FAIL "-ffast-math is refused" "configure SUCCEEDED; the guard did not fire"
    elif grep -q "pantr refuses to build with -ffast-math" "$tmp/fastmath.log"; then
        record PASS "-ffast-math is refused"
    else
        record FAIL "-ffast-math is refused" "configure failed for some other reason"
        tail -20 "$tmp/fastmath.log"
    fi

    # Same for -ffinite-math-only, which is banned for an additional reason:
    # it licenses the compiler to delete the checks that detect a degenerate
    # input.
    #
    # The message is grepped for, exactly as above. Asserting only that configure
    # FAILED is not the same assertion: a typo in the flag name, an unrelated
    # CMake error, or a missing compiler all make configure fail too, and this
    # check would have gone on reporting PASS for a guard that had stopped firing.
    if cmake -S . -B "$tmp/finite" -G Ninja \
             -DCMAKE_CXX_FLAGS="-ffinite-math-only" \
             -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF \
             >"$tmp/finite.log" 2>&1; then
        record FAIL "-ffinite-math-only is refused" "configure SUCCEEDED; the guard did not fire"
    elif grep -q "pantr refuses to build with -ffinite-math-only" "$tmp/finite.log"; then
        record PASS "-ffinite-math-only is refused"
    else
        record FAIL "-ffinite-math-only is refused" "configure failed for some other reason"
        tail -20 "$tmp/finite.log"
    fi

    # The concepts probe must reject a compiler that lacks concepts. GCC 9 is
    # the one to hand on this machine: it does not even accept -std=c++20, so
    # it exercises the gate rather than the version filter below it.
    if [[ -x /usr/bin/g++-9 ]]; then
        if cmake -S . -B "$tmp/gcc9" -G Ninja \
                 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
                 -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF \
                 >"$tmp/gcc9.log" 2>&1; then
            record FAIL "concepts gate rejects GCC 9" "configure SUCCEEDED"
        elif grep -qE "requires working C\+\+20 concepts|C\+\+20|std=c\+\+20" "$tmp/gcc9.log"; then
            record PASS "concepts gate rejects GCC 9"
        else
            record FAIL "concepts gate rejects GCC 9" "rejected, but not by the gate"
        fi
    else
        record SKIP "concepts gate rejects GCC 9" "/usr/bin/g++-9 absent"
    fi

    # The version floor is 10 for GCC and Clang alike, and 10 is the lowest
    # version this tree has actually been built and tested with. So the assertion
    # here is that both of the machine's 2020 compilers are ACCEPTED -- the
    # opposite of what this check asserted until the floor was measured, when it
    # sat at 14 for Clang on a guess and at nothing at all for GCC.
    #
    # Configuring is not the interesting half. `cxx` below builds the whole tree
    # with the development toolchains; what these two prove is that the gate does
    # not stand in the way of a compiler that works, which is the failure mode a
    # version table produces and the reason this file distrusts them.
    for _old_cxx in /usr/bin/clang++-10 /usr/bin/g++-10; do
        _name="$(basename "$_old_cxx")"
        if [[ ! -x "$_old_cxx" ]]; then
            record SKIP "$_name is accepted" "$_old_cxx absent"
            continue
        fi
        if cmake -S . -B "$tmp/$_name" -G Ninja \
                 -DCMAKE_CXX_COMPILER="$_old_cxx" \
                 -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF \
                 >"$tmp/$_name.log" 2>&1; then
            record PASS "$_name is accepted"
        else
            record FAIL "$_name is accepted" "the floor rejects a version measured to work"
            tail -20 "$tmp/$_name.log"
        fi
    done

    # The override still has to work, because the floor still rejects something
    # below it and someone will need past it one day. There is no compiler below
    # 10 on this machine to try it against -- g++ 9.5 is stopped earlier, by the
    # concepts probe -- so this asserts the flag is at least accepted and does not
    # itself break a configure.
    if [[ -x /usr/bin/g++-10 ]]; then
        if cmake -S . -B "$tmp/override" -G Ninja \
                 -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
                 -DPANTR_ALLOW_UNTESTED_COMPILER=ON \
                 -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF \
                 >"$tmp/override.log" 2>&1; then
            record PASS "PANTR_ALLOW_UNTESTED_COMPILER is honoured"
        else
            record FAIL "PANTR_ALLOW_UNTESTED_COMPILER is honoured"
        fi
    else
        record SKIP "PANTR_ALLOW_UNTESTED_COMPILER is honoured" "no old compiler to try"
    fi

    # The offline escape. FETCHCONTENT_FULLY_DISCONNECTED with a populated
    # source dir must configure without touching the network; this is the path
    # a cluster compute node needs and the day it is needed is the day nobody
    # wants to be discovering it.
    #
    # The sources have to come from somewhere, and where they used to come from
    # was build/gcc/_deps -- which `cxx` creates and `main` runs AFTER this
    # function. So on any from-scratch run this check SKIPped, every time, and the
    # escape it exists to exercise was never once exercised. It now populates its
    # own copy when that directory is absent, which fixes the ordering and makes
    # `ci_local.sh gates` self-contained as a bonus.
    #
    # PANTR_BUILD_TESTS is left ON in both configures below, and has to be: Eigen
    # is only declared when the tests are built (cmake/PantrDependencies.cmake),
    # so with tests OFF there would be no eigen-src to point at and the check
    # would pass while testing half of what it claims.
    local deps_root="$ROOT/build/gcc/_deps"
    local have_deps=1
    if [[ ! -d "$deps_root/mdspan-src" || ! -d "$deps_root/eigen-src" ]]; then
        if cmake -S . -B "$tmp/fetch" -G Ninja -DPANTR_BUILD_PYTHON=OFF \
                 >"$LOGDIR/fetch.log" 2>&1; then
            deps_root="$tmp/fetch/_deps"
        else
            have_deps=0
        fi
    fi

    if [[ "$have_deps" -eq 0 ]]; then
        # A genuine SKIP rather than a FAIL: with no network and no prior build
        # there is nothing to point the disconnected configure at, and that is a
        # property of the machine rather than of the tree.
        record SKIP "offline escape configures" "could not populate _deps to point at"
        tail -20 "$LOGDIR/fetch.log"
    elif cmake -S . -B "$tmp/offline" -G Ninja \
               -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
               -DFETCHCONTENT_SOURCE_DIR_MDSPAN="$deps_root/mdspan-src" \
               -DFETCHCONTENT_SOURCE_DIR_EIGEN="$deps_root/eigen-src" \
               -DPANTR_BUILD_PYTHON=OFF \
               >"$tmp/offline.log" 2>&1; then
        record PASS "offline escape configures"
    else
        record FAIL "offline escape configures"
        tail -20 "$tmp/offline.log"
    fi
}

# --------------------------------------------------------------------------
# cxx -- both toolchains, from scratch
# --------------------------------------------------------------------------
#
# Each compiler gets its own build directory, and each is DELETED first. Probe
# results are cached in CMakeCache.txt, so a reused directory can answer for the
# compiler that configured it rather than the one being tested -- the failure
# mode design/toolchain_requirements.md flags as worth an afternoon.

cxx() {
    step "C++ toolchains"

    local preset
    for preset in gcc clang; do
        rm -rf "$ROOT/build/$preset"
        check "$preset: configure" cmake --preset "$preset"
        check "$preset: build (-Werror)" cmake --build --preset "$preset"
        check "$preset: ctest" ctest --preset "$preset"
    done

    # The sanitizer build, which until FELIGN/pantr#359 nothing ran.
    #
    # `gcc-debug` and its test preset were both defined and both unused, so
    # -fsanitize=address,undefined was configured and never executed. Two things
    # depend on it. The kernels' undefined behaviour on out-of-contract input is
    # only observable here; and the precondition tests are only REGISTERED here,
    # because PANTR_PRECONDITION is `assert` and a release build compiles it out,
    # which would leave those executables running the undefined behaviour rather
    # than reporting it. Running the release presets alone therefore says nothing
    # about either.
    #
    # It also ran in UBSan's recover mode until the preset gained
    # -fno-sanitize-recover=undefined, which means a UBSan finding was printed
    # here and the step still recorded PASS. Measured on a signed overflow: exit
    # 0 without the flag, exit 1 with it. The flag lives in the preset rather
    # than in this script so that .github/workflows/cpp.yaml, which builds the
    # same preset, cannot end up sanitizing something different.
    rm -rf "$ROOT/build/gcc-debug"
    check "gcc-debug: configure (asan, ubsan)" cmake --preset gcc-debug
    check "gcc-debug: build" cmake --build --preset gcc-debug
    check "gcc-debug: ctest" ctest --preset gcc-debug

    # The thread sanitizer, which is a separate build for a mechanical reason:
    # -fsanitize=thread cannot be combined with -fsanitize=address, so the leg
    # above cannot carry it.
    #
    # What only this leg can see is a race in a lazily-filled memo, and it is the
    # one defect class in the port that no assertion on a value will ever catch.
    # Measured on the bare `mutable std::optional` spelling of pantr::LazySlot:
    # 4 TSan reports with 8 threads, AND 60 correct answers in 60 unsanitized
    # runs. A suite that checks totals reports success on it every time.
    #
    # A clean run here is only evidence if the threads really contend, which is
    # the null-result trap design/bspline_derived_caches.md records. The control
    # was run rather than assumed: replacing LazySlot's double-checked locking
    # with an unsynchronised `optional` gave 23 data races and turned both
    # threaded tests red on this tree, and restoring it gave 0 and 47/47. Re-run
    # that control if this leg is ever green on a tree where it should not be.
    rm -rf "$ROOT/build/gcc-tsan"
    check "gcc-tsan: configure (tsan)" cmake --preset gcc-tsan
    check "gcc-tsan: build" cmake --build --preset gcc-tsan
    check "gcc-tsan: ctest" ctest --preset gcc-tsan

    # The version floor, built rather than merely accepted.
    #
    # `gates` above asserts that the floor compilers CONFIGURE. That is the gate's
    # behaviour, and it is not the claim the floor makes. The floor says version 10
    # of both families builds this tree and passes its tests, and until that is
    # re-checked it rests on one manual measurement taken the day the floor was
    # set: the first commit using something GCC 10 lacks would break the claim with
    # nothing to notice.
    #
    # This is the ONLY place that check exists. .github/workflows/cpp.yaml runs
    # GCC 14 only -- deliberately, as its own header says -- and ubuntu-24.04 does
    # not package GCC 10, so covering the floor there needs an older image or a
    # container -- more weight than a prototype should carry.
    # design/toolchain_requirements.md records that the floor is verified locally
    # and not by CI.
    local floor_cxx floor_name floor_dir
    for floor_cxx in /usr/bin/g++-10 /usr/bin/clang++-10; do
        floor_name="$(basename "$floor_cxx")"
        if [[ ! -x "$floor_cxx" ]]; then
            record SKIP "floor: $floor_name" "$floor_cxx absent"
            continue
        fi
        floor_dir="$ROOT/build/floor-$floor_name"
        rm -rf "$floor_dir"
        # PANTR_ALLOW_UNTESTED_COMPILER is deliberately NOT passed: the floor is
        # 10, so 10 must configure on its own. If it ever needs the override, the
        # floor moved and this check is what says so.
        check "floor: $floor_name configure" \
            cmake -S "$ROOT" -B "$floor_dir" -G Ninja \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_CXX_COMPILER="$floor_cxx" \
                  -DPANTR_WERROR=ON -DPANTR_BUILD_PYTHON=OFF
        check "floor: $floor_name build (-Werror)" cmake --build "$floor_dir"
        check "floor: $floor_name ctest" ctest --test-dir "$floor_dir" --output-on-failure
    done

    # The mdspan toggle is a decision the build makes silently; print which way
    # it went, because "which mdspan am I actually using" is the first question
    # asked when a build behaves oddly.
    local which_mdspan="Kokkos reference implementation"
    if grep -q "^PANTR_HAS_STD_MDSPAN:INTERNAL=1$" "$ROOT/build/gcc/CMakeCache.txt" 2>/dev/null; then
        which_mdspan="<mdspan> from the standard library"
    fi
    record INFO "mdspan implementation" "$which_mdspan"
}

# --------------------------------------------------------------------------
# discipline -- the source-level rules no compiler enforces
# --------------------------------------------------------------------------
#
# The four disciplines of design/automatic_differentiation.md, which a
# scalar-generic core has to keep if a forward-mode `Dual` is ever to be dropped
# in.
#
# Most of them are NOT checked here, and that is the right answer rather than a
# gap: cpp/tests/test_scalar_generic.cpp defines a `Dual1` with the five
# arithmetic operators and deliberately WITHOUT `operator==`, without ordering
# and without a conversion to `double`, then instantiates the kernel on it. So a
# comparison that bypasses `value_of`, a `floor` on the scalar, or an integer
# cast of it all fail to COMPILE. A type system that rejects the violation beats
# a regular expression that reports it, and the check that matters is therefore
# that the fixture still exists and is still instantiated.
#
# What is left for grep is the one discipline the compiler cannot see: a
# qualified call. `std::sqrt(x)` compiles perfectly for `double` AND for a
# `Dual` that happens to be convertible to one, while silently computing the
# wrong thing.

discipline() {
    step "Scalar-generic disciplines"

    local hits

    # The compile-time enforcement itself. If this fixture stops instantiating
    # the kernel, three of the four disciplines quietly stop being checked at
    # all and nothing else in this script would notice.
    if grep -q 'static_assert(pantr::Real<Dual1>' "$ROOT/cpp/tests/test_scalar_generic.cpp" \
       && grep -q '!std::equality_comparable<Dual1>' "$ROOT/cpp/tests/test_scalar_generic.cpp" \
       && grep -q 'tabulate_cardinal_bspline_1d<Dual1>' "$ROOT/cpp/tests/test_scalar_generic.cpp"; then
        record PASS "Dual fixture still enforces Tier B"
    else
        record FAIL "Dual fixture still enforces Tier B" \
               "test_scalar_generic.cpp no longer pins the concept or the instantiation"
    fi

    # Discipline 3: unqualified math calls with a using-declaration.
    # `std::sqrt(x)` names the overload directly, suppressing ADL, which is
    # exactly how a user-defined scalar's own overload gets excluded. The
    # comment in cpp/include/pantr/core/scalar.hpp promises this guard exists.
    #
    # Two exclusions, both deliberate. Comment lines are stripped, because the
    # rule is stated in prose in several headers and a guard that fires on its
    # own documentation gets switched off within a week. And cpp/tests and
    # cpp/benchmark are out of scope: they are instantiated on concrete `double`
    # and `float`, so `std::abs` there is a call on a known type rather than on
    # a scalar, and banning it would be enforcing a rule that does not apply.
    # The scope is the library -- the headers, and the bindings that call them.
    hits="$(grep -rnE 'std::(sqrt|cbrt|abs|fabs|pow|exp|log|log2|log10|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|hypot|fma|fmin|fmax|copysign|floor|ceil|round|trunc)\(' \
            --include='*.hpp' --include='*.cpp' "$ROOT/cpp/include" "$ROOT/cpp/bindings" 2>/dev/null \
            | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(///|//|\*)' || true)"
    if [[ -n "$hits" ]]; then
        record FAIL "no qualified std:: math calls" "$(wc -l <<<"$hits") site(s)"
        printf '%s\n' "$hits"
    else
        record PASS "no qualified std:: math calls"
    fi

    # No kernel takes an IndexMap. The MPI distribution type belongs to the
    # parallel layer; a kernel that accepts one has had the layering leak into
    # it. Vacuous today -- pantr has no such type in any language -- and
    # asserted so it stays that way, since this is cheap now and archaeology
    # later.
    hits="$(grep -rn 'IndexMap' --include='*.hpp' --include='*.cpp' "$ROOT/cpp" 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
        record FAIL "no kernel takes an IndexMap" "$(wc -l <<<"$hits") site(s)"
        printf '%s\n' "$hits"
    else
        record PASS "no kernel takes an IndexMap"
    fi

    # -march must not reach the BUILD. design/simd.md makes ISA variants stage 2
    # and gates them on a measurement; cpp/README.md now reports that
    # measurement, so documentation naturally mentions the flag and must not be
    # scanned. Only files that can actually set a compile option are.
    hits="$(grep -rn -- '-march=' "$ROOT/cmake" "$ROOT/CMakeLists.txt" \
            "$ROOT/CMakePresets.json" "$ROOT/pyproject.toml" \
            --include='*.cmake' --include='CMakeLists.txt' --include='*.json' \
            --include='*.toml' 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
        record FAIL "no -march in the build files" "$(wc -l <<<"$hits") site(s)"
        printf '%s\n' "$hits"
    else
        record PASS "no -march in the build files"
    fi

    # The quad kernels stay double-only, except the one that must not be.
    #
    # legendre.hpp records the measurement behind this: a rule generator has no
    # differentiable input (n is an integer), so Tier B never reaches it and
    # `Real` genericity buys nothing, while instantiating Newton at `float` was
    # measured to give a 1.46e-3 relative weight error at n = 200 -- so that
    # kernel, and gauss_legendre_symmetric/lambert_w_principal/tanh_sinh
    # alongside it, are plain `double`. Conversely `modified_chebyshev_nodes`
    # in simple_rules.hpp MUST stay templated: the Python computes in the
    # storage format, and a double-then-narrow port was measured to differ on
    # 17% of float32 arguments. So exactly one `template <` may appear under
    # cpp/include/pantr/quad, and it must be the Chebyshev one. This fails
    # both if a template turns up where it should not and if the one that
    # should be there disappears.
    hits="$(grep -rn 'template <' --include='*.hpp' "$ROOT/cpp/include/pantr/quad" 2>/dev/null || true)"
    hit_count="$(printf '%s\n' "$hits" | grep -c . || true)"
    if [[ "$hit_count" -eq 1 ]] \
       && grep -q 'template <' "$ROOT/cpp/include/pantr/quad/simple_rules.hpp" 2>/dev/null; then
        record PASS "quad stays double-only except modified_chebyshev_nodes"
    else
        record FAIL "quad stays double-only except modified_chebyshev_nodes" \
               "$hit_count template<> site(s) under cpp/include/pantr/quad, expected exactly 1, in simple_rules.hpp"
        printf '%s\n' "$hits"
    fi

    # The Newton and Halley step counts agree across the two languages.
    #
    # legendre.hpp and lambert_w.hpp both say in prose that their step count
    # must move together with the Python constant it mirrors or parity is
    # lost, but nothing enforces that. Extracted straight from both sources
    # rather than hardcoded here, so this guard does not itself go stale the
    # way a copied number would.
    local py_newton cpp_newton py_halley cpp_halley
    py_newton="$(sed -n -E 's/^_GAUSS_LEGENDRE_NEWTON_STEPS: int = ([0-9]+)$/\1/p' \
                 "$ROOT/src/pantr/quad/_rules_core.py")"
    cpp_newton="$(sed -n -E 's/^inline constexpr int gauss_legendre_newton_steps = ([0-9]+);$/\1/p' \
                  "$ROOT/cpp/include/pantr/quad/legendre.hpp")"
    py_halley="$(sed -n -E 's/^_LAMBERT_W_HALLEY_STEPS: int = ([0-9]+)$/\1/p' \
                 "$ROOT/src/pantr/quad/_rules_core.py")"
    cpp_halley="$(sed -n -E 's/^inline constexpr int lambert_w_halley_steps = ([0-9]+);$/\1/p' \
                  "$ROOT/cpp/include/pantr/quad/lambert_w.hpp")"

    if [[ -n "$py_newton" && -n "$cpp_newton" && "$py_newton" == "$cpp_newton" ]]; then
        record PASS "Gauss-Legendre Newton step count agrees (Python=C++=$py_newton)"
    else
        record FAIL "Gauss-Legendre Newton step count agrees" \
               "Python=${py_newton:-<not found>} C++=${cpp_newton:-<not found>}"
    fi
    if [[ -n "$py_halley" && -n "$cpp_halley" && "$py_halley" == "$cpp_halley" ]]; then
        record PASS "Lambert W Halley step count agrees (Python=C++=$py_halley)"
    else
        record FAIL "Lambert W Halley step count agrees" \
               "Python=${py_halley:-<not found>} C++=${cpp_halley:-<not found>}"
    fi

    # The BVH traversal-stack depth agrees across the two languages.
    #
    # The third guard of this shape, and the one with a caller outside this repo.
    # `_BVH_STACK_DEPTH` is imported by a downstream consumer, so the Python copy is
    # the source of truth and stays where it is; `kBvhStackDepth` mirrors it, and
    # since FELIGN/pantr#384 the mirror is what ENFORCES the limit under
    # PANTR_BACKEND=cpp. Two spellings of one number, each enforcing for one
    # backend, is exactly the drift a test cannot see: tests/test_grid_reexports.py
    # pins the Python one by value and the C++ mirror is not exposed to Python at
    # all, so nothing inside the suite compares them.
    #
    # Extracted from both sources by regex rather than hardcoded here, so this guard
    # does not itself go stale the way a copied number would.
    local py_bvh_depth cpp_bvh_depth
    py_bvh_depth="$(sed -n -E 's/^_BVH_STACK_DEPTH: Final\[int\] = ([0-9]+)$/\1/p' \
                    "$ROOT/src/pantr/grid/_bvh_core.py")"
    cpp_bvh_depth="$(sed -n -E \
        's/^inline constexpr std::int64_t kBvhStackDepth = ([0-9]+);$/\1/p' \
        "$ROOT/cpp/include/pantr/grid/bvh.hpp")"

    if [[ -n "$py_bvh_depth" && -n "$cpp_bvh_depth" && "$py_bvh_depth" == "$cpp_bvh_depth" ]]; then
        record PASS "BVH traversal stack depth agrees (Python=C++=$py_bvh_depth)"
    else
        record FAIL "BVH traversal stack depth agrees" \
               "Python=${py_bvh_depth:-<not found>} C++=${cpp_bvh_depth:-<not found>}"
    fi
}

# --------------------------------------------------------------------------
# python -- the extension, the parity harness, both backends
# --------------------------------------------------------------------------

python_checks() {
    step "Python: extension, parity, backends"

    if [[ ! -d "$VENV" ]]; then
        record SKIP "python checks" "no .venv; run: python -m venv --system-site-packages .venv"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$VENV/bin/activate"

    # This script is the calibrated-host runner, so the parity suite's liveness
    # guards are enforced here and only here.
    #
    # Those guards check that a bound is still doing work: that the two backends
    # still disagree somewhere, that the worst observed ratio is still close to the
    # bound. Worth having, because a bound nothing exercises can rot unnoticed. But
    # the quantities behind them are chosen at run time by the CPU, since glibc
    # dispatches `exp` through IFUNC on the processor's features and numpy
    # dispatches its own loops the same way, so a different host can make two
    # implementations agree that used to differ. On GitHub that produced a red from
    # an unchanged tree: the same commit gave 6 failures on one run and none on a
    # re-run. Off this variable the guards report instead of failing, and
    # tests/_parity_harness.py's demand_the_reference_host carries the argument.
    #
    # Running this on a machine that is NOT the one those numbers were measured on
    # will therefore fail them. That is the calibration not transferring, not the
    # code being wrong; re-measure before changing a number.
    export PANTR_REFERENCE_HOST=1

    check "editable install" pip install -e . --no-build-isolation -q

    # An explicit request that cannot be served must FAIL rather than fall back.
    # This is the rule that makes every A/B measurement in this prototype
    # trustworthy, so it is asserted rather than assumed -- on both selection
    # axes, because they are two variables answering two questions and a rule
    # asserted on one of them says nothing about the other.
    #
    # Above the extension check below, and deliberately: neither variable needs
    # the extension to be built for an unknown value to fail, and the ISA axis is
    # documented as independent of whether the C++ backend is installed, so
    # checking it only where the extension exists would not check that.
    local var
    for var in PANTR_BACKEND PANTR_ISA_VARIANT; do
        if env "$var=nonesuch" python -c "import pantr" 2>/dev/null; then
            record FAIL "unknown $var fails loudly" "import SUCCEEDED"
        else
            record PASS "unknown $var fails loudly"
        fi
    done

    # The extension must be present. Every check below it would otherwise SKIP,
    # and a suite that skips its way to green is the trap CLAUDE.md names: a
    # missing optional dependency skips without complaint.
    if python -c "import pantr._pantr_cpp" 2>/dev/null; then
        record PASS "pantr._pantr_cpp imports"
    else
        record FAIL "pantr._pantr_cpp imports" "everything below is meaningless"
        return 0
    fi

    check "parity: C++ vs numba" python -m pytest tests/parity -q

    # The acceptance criterion: the EXISTING suite passes against both backends.
    # change_basis is the module that reaches the ported kernel, so it is named
    # explicitly; the full suite follows because a kernel swap that breaks
    # something else is exactly what a narrow run would miss.
    local be
    for be in python cpp; do
        PANTR_BACKEND=$be check "change_basis on $be" python -m pytest \
            tests/test_change_basis_1D.py tests/test_change_basis_domain.py -q
    done
    for be in python cpp; do
        PANTR_BACKEND=$be check "full suite on $be" python -m pytest tests -q -x -m "not slow"
    done
}

# --------------------------------------------------------------------------
# splitmode -- the nanobind wheel-strategy probe
# --------------------------------------------------------------------------
#
# design/_memory/nanobind-status.md records that split mode supersedes the
# stable-ABI approach on both axes that mattered, and that it was unreleased.
# The bindings are written against the nanobind 3 API (NB_TRAMPOLINE without a
# Size, return-value policies and argument annotations as compile-time tags,
# gil_scoped_acquire consulting is_valid()), which also compiles under 2.x -- so
# this probe answers whether it BUILDS and IMPORTS under the dev release,
# without the shipped build depending on the answer.
#
# It is expected to be the flakiest check here, because it installs a
# pre-release into a throwaway environment. A failure is information, not a
# blocker: it is reported and does not fail the run.

splitmode() {
    step "nanobind split mode (pre-release probe)"

    # The probe gets a build directory of its own, and that is load-bearing
    # rather than tidiness. pyproject sets a PERSISTENT build-dir so an editable
    # rebuild stays incremental, and a CMake cache is sticky in both directions:
    #
    #   * cpp/bindings/CMakeLists.txt finds nanobind through the interpreter and
    #     writes `nanobind_ROOT` as a normal variable, but find_package records
    #     the result in the CACHE variable `nanobind_DIR`. On any later
    #     configure a valid cached <pkg>_DIR wins over <pkg>_ROOT, so the probe
    #     would silently configure against whichever nanobind the ordinary build
    #     resolved first -- reporting on 2.x while claiming to test 3.x.
    #   * PANTR_NANOBIND_SPLIT=ON would likewise stay in that cache and be
    #     inherited by the NEXT ordinary editable install, which then fails.
    #     That made `ci_local.sh all` pass once and fail the second time.
    #
    # Sharing nothing is the whole fix. See the issue this closes.
    local probedir tmpvenv probebuild
    probedir="$(mktemp -d)"
    tmpvenv="$probedir/venv"
    probebuild="$probedir/skbuild"
    if ! python -m venv --system-site-packages "$tmpvenv" >/dev/null 2>&1; then
        record SKIP "nanobind split mode" "could not create a probe venv"
        rm -rf "$probedir"
        return 0
    fi
    # shellcheck disable=SC1091
    source "$tmpvenv/bin/activate"

    # BOTH halves, because that is what split mode is. The frontend links
    # against the stable ABI and the backend module carries the runtime; with
    # only the frontend installed the extension builds and then refuses to
    # import, naming the missing backend. design/_memory/nanobind-status.md
    # records the pairing that was measured to work.
    if ! pip install -q --pre "nanobind>=3.0.0.dev0" "nanobind-backend" \
            >"$LOGDIR/nb3.log" 2>&1; then
        record SKIP "nanobind split mode" "pre-release not installable"
        deactivate || true
        rm -rf "$probedir"
        return 0
    fi

    local nbver
    nbver="$(python -c 'import nanobind; print(nanobind.__version__)' 2>/dev/null || echo unknown)"

    if pip install -e . --no-build-isolation -q \
         --config-settings=build-dir="$probebuild" \
         --config-settings=cmake.define.PANTR_NANOBIND_SPLIT=ON >"$LOGDIR/split.log" 2>&1 \
       && python -c "import pantr._pantr_cpp" >>"$LOGDIR/split.log" 2>&1; then
        record PASS "nanobind split mode" "nanobind $nbver"
    else
        record WARN "nanobind split mode" "nanobind $nbver: see the log tail below"
        tail -25 "$LOGDIR/split.log"
    fi

    # Which nanobind CMake actually loaded, rather than which one Python could
    # import. The two disagreeing is precisely the failure above, and the
    # version line the build prints comes from the interpreter, so it cannot
    # tell them apart on its own.
    #
    # This one is a FAIL rather than a WARN, and the distinction is deliberate:
    # the step is non-blocking about whether the PRE-RELEASE works, not about
    # whether the probe measured what it claims. A probe that quietly configured
    # against a different nanobind reports a verdict on the wrong subject, which
    # is worse than reporting nothing.
    #
    # `|| true` is required, not defensive: `set -euo pipefail` is in force, so
    # an unmatched grep inside a command substitution would abort the function
    # before it records anything.
    local nbcmake
    # Both levels: pyproject's build-dir carries a `{wheel_tag}` component, but
    # the override above is a literal path, so the cache lands directly in it.
    nbcmake="$(grep -hm1 '^nanobind_DIR:' \
        "$probebuild"/CMakeCache.txt "$probebuild"/*/CMakeCache.txt 2>/dev/null \
        | cut -d= -f2- || true)"
    if [ -z "$nbcmake" ]; then
        record WARN "split probe used the probe's nanobind" "no CMake cache to read"
    elif [[ "$nbcmake" == "$tmpvenv"/* ]]; then
        record PASS "split probe used the probe's nanobind"
    else
        record FAIL "split probe used the probe's nanobind" "configured against $nbcmake"
    fi

    deactivate || true
    rm -rf "$probedir"
}

# --------------------------------------------------------------------------
# consumer -- the installed package, used from outside the tree
# --------------------------------------------------------------------------
#
# Everything else here builds pantr. This is the only check that builds
# something AGAINST pantr the way a third party would: install to a throwaway
# prefix, then configure cpp/consumer standalone against it. The in-tree build
# cannot substitute, because it reaches the headers through BUILD_INTERFACE and
# the in-tree alias, so an export that is wrong is invisible to it. Its first run
# found the exported target was named `pantr::pantr_core`.
#
# Eigen and Kokkos mdspan are installed to the same prefix first, from the
# sources the gcc build already fetched, so this clones nothing. That is also
# what exercises FIND_PACKAGE_ARGS: pantr must pick up those installs rather
# than fetch its own.

consumer() {
    step "Installed package, consumed from outside the tree"

    local deps="$ROOT/build/gcc/_deps"
    if [[ ! -d "$deps/eigen-src" || ! -d "$deps/mdspan-src" ]]; then
        record SKIP "consumer" "run the cxx section first; $deps is not populated"
        return
    fi

    local tmp; tmp="$(mktemp -d -t pantr_consumer_XXXXXXXX)"
    local prefix="$tmp/prefix"

    # EIGEN_BUILD_BLAS/LAPACK OFF is not tidiness: with them ON, Eigen's install
    # step fails on a library this configuration never built.
    check "consumer: install Eigen" \
        cmake -S "$deps/eigen-src" -B "$tmp/eigen" -G Ninja \
              -DCMAKE_INSTALL_PREFIX="$prefix" -DEIGEN_BUILD_TESTING=OFF \
              -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_DEMOS=OFF \
              -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF \
              -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    check "consumer: install Eigen (step)" cmake --install "$tmp/eigen"

    check "consumer: install mdspan" \
        cmake -S "$deps/mdspan-src" -B "$tmp/mdspan" -G Ninja \
              -DCMAKE_INSTALL_PREFIX="$prefix" -DMDSPAN_CXX_STANDARD=20 \
              -DMDSPAN_ENABLE_TESTS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    check "consumer: install mdspan (step)" cmake --install "$tmp/mdspan"

    check "consumer: configure pantr for install" \
        cmake -S "$ROOT" -B "$tmp/pantr" -G Ninja -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix" \
              -DPANTR_INSTALL=ON -DPANTR_USE_SYSTEM_DEPS=ON \
              -DPANTR_BUILD_TESTS=OFF \
              -DPANTR_BUILD_BENCHMARK=OFF -DPANTR_BUILD_PYTHON=OFF
    check "consumer: install pantr" cmake --install "$tmp/pantr"

    # The point of PANTR_USE_SYSTEM_DEPS. Asserted on the artifact rather than on
    # a log: $LOGDIR/step.log holds only the most recent command's output, so
    # grepping it here would have read the install step and passed vacuously.
    if [[ -d "$tmp/pantr/_deps/eigen-src" ]]; then
        record FAIL "consumer: used the installed deps" "it fetched its own Eigen"
    elif grep -qE "^eigen_DIR:PATH=$prefix" "$tmp/pantr/CMakeCache.txt" 2>/dev/null; then
        # eigen_DIR, not any mention of the prefix: CMAKE_PREFIX_PATH is passed on
        # the command line, so it appears in the cache whether or not it was used,
        # and asserting on that would pass without measuring anything.
        record PASS "consumer: used the installed deps"
    else
        record FAIL "consumer: used the installed deps" \
               "eigen_DIR does not point into $prefix"
    fi

    check "consumer: configure against the install" \
        cmake -S "$ROOT/cpp/consumer" -B "$tmp/consumer" -G Ninja \
              -DCMAKE_PREFIX_PATH="$prefix"
    check "consumer: build" cmake --build "$tmp/consumer"
    check "consumer: run" "$tmp/consumer/consumer"

    rm -rf "$tmp"
}

# --------------------------------------------------------------------------

main() {
    local what="${1:-all}"
    case "$what" in
        gates)      gates ;;
        cxx)        cxx ;;
        discipline) discipline ;;
        python)     python_checks ;;
        splitmode)  splitmode ;;
        consumer)   consumer ;;
        all)        gates; cxx; discipline; python_checks; splitmode; consumer ;;
        *)          echo "unknown section: $what" >&2; exit 2 ;;
    esac

    step "Summary"
    printf '%s\n' "${RESULTS[@]}"
    if [[ "$FAILED" -eq 1 ]]; then
        printf '\n\033[31mFAILED\033[0m\n'
        exit 1
    fi
    printf '\n\033[32mAll checks passed\033[0m\n'
}

main "$@"
