/// \file
/// The once-only memo behind every lazily built derived quantity.
///
/// ## What can be checked here, and what cannot
///
/// The property this type exists for -- that concurrent first calls are not a data
/// race -- **is not observable from a value**. `design/bspline_derived_caches.md`
/// measured the unsynchronised shape answering correctly on every run of a sweep while
/// a thread sanitizer reported races in the same binary, which is the signature of
/// undefined behaviour that happens to work. So the assertions below are deliberately
/// split into two kinds, and only the first kind is evidence on its own:
///
///  - **What a value test settles**: the build runs exactly once however many threads
///    arrive together, every thread sees the same object, an empty memo stays empty
///    until asked, a throwing build leaves the memo retryable, and the copy and move
///    operations carry the value the way the plain `std::optional` member they replace
///    did.
///  - **What only a sanitizer settles**: the absence of the race. The concurrent case
///    below is written to be *hammered* -- many threads released together on a spin
///    barrier -- so that a build under `--preset gcc-tsan` has something to report on.
///    A green run of this file without that preset says nothing about the race, and is
///    not quoted as if it did.
///
/// ## Why the build counter is an ordinary `int`
///
/// It is written only under the memo's own lock, so counting with a plain integer is
/// exactly the assertion that the lock is doing its job: if the mutex were absent, the
/// counter would race and the sanitizer would say so. An atomic counter would hide the
/// very thing being tested.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/core/lazy_memo.hpp"

namespace {

using pantr::LazyMemo;

/// The memo is empty until something asks for the value.
void test_a_fresh_memo_is_not_filled() {
    const LazyMemo<int> memo;
    PANTR_CHECK(!memo.is_filled());
}

/// One call fills it; the value and the filled flag agree.
void test_the_first_call_builds() {
    const LazyMemo<int> memo;
    PANTR_CHECK(memo.get_or_build([] { return 7; }) == 7);
    PANTR_CHECK(memo.is_filled());
}

/// Later calls return the same object, and never run the build again.
void test_later_calls_reuse_the_same_object() {
    const LazyMemo<int> memo;
    int builds = 0;
    const int& first = memo.get_or_build([&builds] { return ++builds; });
    const int& second = memo.get_or_build([&builds] { return ++builds; });
    PANTR_CHECK(builds == 1);
    PANTR_CHECK(&first == &second);
    PANTR_CHECK(first == 1);
}

/// A build that throws leaves the memo empty, so the next call tries again.
///
/// The alternative -- publishing before the value is complete -- would hand every later
/// caller a half-built object with no diagnostic, which is worse than the exception.
void test_a_throwing_build_leaves_the_memo_retryable() {
    const LazyMemo<int> memo;
    bool threw = false;
    try {
        (void)memo.get_or_build([]() -> int { throw std::runtime_error("no"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    PANTR_CHECK(threw);
    PANTR_CHECK(!memo.is_filled());
    PANTR_CHECK(memo.get_or_build([] { return 3; }) == 3);
}

/// Many threads arriving together build exactly once and all read the same value.
///
/// The spin barrier is what makes the case worth running: without it the threads start
/// far enough apart that the first is usually finished before the second looks, and the
/// window the race lives in is never entered.
///
/// **Every thread sums the memoised vector before any join, and that is load-bearing
/// rather than a stronger assertion for its own sake.** An earlier version stored only
/// `&get_or_build(...)` and checked the contents after `join()`. Taking an address is
/// pointer arithmetic on the `optional`: it loads nothing from the payload, so a
/// sanitizer sees no access to instrument, and `join()` supplies its own happens-before
/// for everything after it. That harness reported no races whether the publication was
/// correctly ordered or not -- measured **0 detections in 35 runs** with the acquire and
/// release relaxed -- so it could not have failed. Summing the value inside the threaded
/// section is the whole difference; see `lazy_memo.hpp` for the counts.
///
/// The sums are checked against a closed form computed here rather than copied from a
/// run, so a wrong answer is a wrong answer and not a changed baseline.
void test_concurrent_first_calls_build_once() {
    constexpr int kThreads = 16;
    constexpr int kRounds = 200;
    constexpr std::size_t kWidth = 512;

    // sum(i for i in [0, kWidth)).
    constexpr std::int64_t kExpected =
        static_cast<std::int64_t>(kWidth) * (static_cast<std::int64_t>(kWidth) - 1) / 2;

    for (int round = 0; round < kRounds; ++round) {
        const LazyMemo<std::vector<std::int64_t>> memo;
        int builds = 0;  // written only under the memo's lock; see the file header
        std::atomic<int> waiting{0};
        std::vector<const std::vector<std::int64_t>*> seen(kThreads, nullptr);
        std::vector<std::int64_t> sums(kThreads, -1);

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                waiting.fetch_add(1, std::memory_order_acq_rel);
                while (waiting.load(std::memory_order_acquire) < kThreads) {
                    std::this_thread::yield();
                }
                const std::vector<std::int64_t>& value =
                    memo.get_or_build([&builds]() -> std::vector<std::int64_t> {
                        ++builds;
                        std::vector<std::int64_t> built(kWidth);
                        for (std::size_t i = 0; i < kWidth; ++i) {
                            built[i] = static_cast<std::int64_t>(i);
                        }
                        return built;
                    });
                seen[static_cast<std::size_t>(t)] = &value;
                std::int64_t total = 0;
                for (const std::int64_t entry : value) {
                    total += entry;
                }
                sums[static_cast<std::size_t>(t)] = total;
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }

        PANTR_CHECK(builds == 1);
        for (int t = 0; t < kThreads; ++t) {
            PANTR_CHECK(seen[static_cast<std::size_t>(t)] == seen[0]);
            PANTR_CHECK(sums[static_cast<std::size_t>(t)] == kExpected);
        }
    }
}

/// Copying a filled memo carries the value, and the copy is independent.
///
/// This is the behaviour the plain `mutable std::optional` member had implicitly, and
/// it is preserved on purpose: a grid's spatial index is a pure function of the grid,
/// so a copy that dropped it would silently turn one build into two.
void test_a_copy_carries_a_built_value() {
    const LazyMemo<int> source;
    (void)source.get_or_build([] { return 5; });

    const LazyMemo<int> copy(source);
    PANTR_CHECK(copy.is_filled());
    int builds = 0;
    PANTR_CHECK(copy.get_or_build([&builds] { return ++builds + 100; }) == 5);
    PANTR_CHECK(builds == 0);
    PANTR_CHECK(&copy.get_or_build([] { return 0; }) != &source.get_or_build([] { return 0; }));
}

/// Copying an empty memo copies the emptiness, not a value.
void test_a_copy_of_an_empty_memo_is_empty() {
    const LazyMemo<int> source;
    const LazyMemo<int> copy(source);
    PANTR_CHECK(!copy.is_filled());
    PANTR_CHECK(copy.get_or_build([] { return 9; }) == 9);
    PANTR_CHECK(!source.is_filled());
}

/// A move takes the value and leaves the source empty.
void test_a_move_empties_the_source() {
    LazyMemo<std::vector<int>> source;
    (void)source.get_or_build([] { return std::vector<int>{4, 5}; });

    const LazyMemo<std::vector<int>> moved(std::move(source));
    PANTR_CHECK(moved.is_filled());
    PANTR_CHECK(moved.get_or_build([] { return std::vector<int>{}; }) == std::vector<int>({4, 5}));
    PANTR_CHECK(!source.is_filled());  // NOLINT(bugprone-use-after-move) -- the point
}

/// Assignment replaces the target's contents in both directions.
void test_assignment_replaces_the_target() {
    LazyMemo<int> filled;
    (void)filled.get_or_build([] { return 11; });

    LazyMemo<int> target;
    target = filled;
    PANTR_CHECK(target.is_filled());
    PANTR_CHECK(target.get_or_build([] { return 0; }) == 11);

    LazyMemo<int> empty;
    target = empty;
    PANTR_CHECK(!target.is_filled());

    target = std::move(filled);
    PANTR_CHECK(target.is_filled());
    PANTR_CHECK(target.get_or_build([] { return 0; }) == 11);
    PANTR_CHECK(!filled.is_filled());  // NOLINT(bugprone-use-after-move) -- the point
}

/// Self-assignment leaves a filled memo alone rather than clearing it.
void test_self_assignment_is_a_no_op() {
    LazyMemo<int> memo;
    (void)memo.get_or_build([] { return 13; });

    LazyMemo<int>& alias = memo;
    memo = alias;
    PANTR_CHECK(memo.is_filled());
    PANTR_CHECK(memo.get_or_build([] { return 0; }) == 13);

    memo = std::move(alias);
    PANTR_CHECK(memo.is_filled());
    PANTR_CHECK(memo.get_or_build([] { return 0; }) == 13);
}

/// A move-only value works: the memo never requires copyability to be usable.
void test_a_move_only_value_is_supported() {
    const LazyMemo<std::unique_ptr<int>> memo;
    PANTR_CHECK(*memo.get_or_build([] { return std::make_unique<int>(21); }) == 21);
    PANTR_CHECK(memo.is_filled());
}

}  // namespace

int main() {
    test_a_fresh_memo_is_not_filled();
    test_the_first_call_builds();
    test_later_calls_reuse_the_same_object();
    test_a_throwing_build_leaves_the_memo_retryable();
    test_concurrent_first_calls_build_once();
    test_a_copy_carries_a_built_value();
    test_a_copy_of_an_empty_memo_is_empty();
    test_a_move_empties_the_source();
    test_assignment_replaces_the_target();
    test_self_assignment_is_a_no_op();
    test_a_move_only_value_is_supported();
    return pantr::test::summary("test_lazy_memo");
}
