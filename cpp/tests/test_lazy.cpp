/// \file
/// `pantr::LazySlot`: built once, published atomically, cold after a copy.
///
/// ## What can and cannot be checked here
///
/// Three of the four properties are ordinary single-threaded facts and are checked
/// as such: the builder runs at most once, a throwing builder leaves the slot cold
/// so the next caller retries, and copying or assigning leaves the target cold
/// rather than carrying a value that describes the source's state.
///
/// The fourth -- that concurrent first touches are a data race under the
/// unsynchronised spelling and are not under this one -- **cannot be checked by
/// asserting a value**. It was measured: eight threads hammering a bare
/// `mutable std::optional` produced 4 reports under g++ 14.4 `-fsanitize=thread`,
/// and **60 of 60 correct answers without the sanitizer**. So a test that runs the
/// threads and compares the total passes on the broken design, every time.
///
/// What the concurrency case below is for, therefore, is to be *the thing a
/// sanitizer build runs*. It asserts the cheap invariants it can -- one build, one
/// address, the right value from every thread -- and its real job is to put eight
/// threads through `get` simultaneously so that `ctest --preset gcc-tsan` has
/// something to look at. Reading a clean TSan run as evidence needs one further
/// check, by analogy with the devirtualisation trap: confirm the threads really do
/// contend, which the build counter below does by construction, since a builder
/// that ran before the threads started would leave the slot already filled and the
/// barrier unused.
///
/// ## The threaded section must READ the value, not just take its address
///
/// This is the part that is easy to lose, and it was learned the expensive way on a
/// second copy of this type that has since been deleted (FELIGN/pantr#429). That copy's
/// concurrency case stored `&slot.get(...)` in a per-thread slot and compared the
/// contents **after `join()`**. Taking an address is pointer arithmetic on the
/// `optional`: it loads nothing from the payload, so a sanitizer has no access to
/// instrument, and `join()` then supplies its own happens-before for everything that
/// follows it.
///
/// The consequence is not that the test was weak. It is that **the test could not fail**.
/// Measured on that harness, with the `acquire` load and the `release` store relaxed and
/// the mutex and flag left alone: **0 detections in 35 runs**. Rewriting it to sum the
/// value inside the threaded section, and nothing else, took it to red on 10 of 10
/// standalone runs and 5 of 5 through `ctest --preset gcc-tsan`, and green on the same
/// counts once the orderings were restored.
///
/// So `check_concurrent_first_touch` below sums every slot it touches **before any
/// join**, and that loop is load-bearing rather than a stronger assertion for its own
/// sake. A refactor that drops it leaves a test that still passes, still looks like
/// coverage, and no longer checks the acquire/release pair at all -- which is then back
/// to resting on `lazy.hpp`'s argument alone.
///
/// One caveat on the mechanism, stated at its real strength: that the two harnesses
/// differ, and in which direction, is measured and reproducible. *Why* address-of is
/// invisible to the sanitizer is inferred from that behaviour rather than read out of its
/// instrumentation, so treat it as well supported and not as established.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/core/lazy.hpp"

namespace {

using pantr::LazySlot;

/// The slot builds on first use and never again.
void check_built_once() {
    LazySlot<std::vector<int>> slot;
    int builds = 0;

    PANTR_CHECK(!slot.filled());
    const std::vector<int>& first = slot.get([&builds] {
        ++builds;
        return std::vector<int>{1, 2, 3};
    });
    PANTR_CHECK(slot.filled());
    PANTR_CHECK(builds == 1);
    PANTR_CHECK(first.size() == 3);

    const std::vector<int>& second = slot.get([&builds] {
        ++builds;
        return std::vector<int>{9};
    });
    PANTR_CHECK_MSG(builds == 1, "the second get must not run its builder");
    PANTR_CHECK_MSG(&second == &first, "the second get must return the same object");
}

/// A builder that throws leaves the slot cold, so the next caller retries.
///
/// The alternative -- publishing before the value exists, or marking the slot
/// filled in a destructor -- would hand the next caller a half-built value with
/// nothing raising, which is worse than the exception it swallowed.
void check_a_throwing_builder_leaves_it_cold() {
    LazySlot<int> slot;
    bool threw = false;
    try {
        (void)slot.get([]() -> int { throw std::runtime_error("no"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    PANTR_CHECK(threw);
    PANTR_CHECK_MSG(!slot.filled(), "a failed build must not publish");
    PANTR_CHECK(slot.get([] { return 7; }) == 7);
}

/// A copy or a move starts cold, and assignment clears the target.
///
/// The assignment case is the one that matters: the slot's value describes its
/// owner's state, and assignment is what replaces that state underneath it. A slot
/// that kept its value across an assignment would be a memo of something that is no
/// longer there, and no value test on the owner would show it.
void check_copies_start_cold() {
    LazySlot<int> filled;
    PANTR_CHECK(filled.get([] { return 5; }) == 5);
    PANTR_CHECK(filled.filled());

    const LazySlot<int> copied(filled);
    PANTR_CHECK_MSG(!copied.filled(), "a copy must start cold");
    PANTR_CHECK(copied.get([] { return 6; }) == 6);

    LazySlot<int> moved(std::move(filled));
    PANTR_CHECK_MSG(!moved.filled(), "a move must start cold");

    LazySlot<int> target;
    PANTR_CHECK(target.get([] { return 1; }) == 1);
    target = copied;
    PANTR_CHECK_MSG(!target.filled(), "assignment must clear the target");
    PANTR_CHECK_MSG(target.get([] { return 2; }) == 2,
                    "a cleared slot rebuilds from its new owner's state");
}

/// Many slots raced by many threads. The sanitizer's target.
///
/// ## Why this is not one slot
///
/// One slot touched once per thread is the obvious shape and it is the wrong one.
/// Every thread then either builds under the lock or blocks on it and reads after
/// acquiring it, and `std::mutex` already establishes that happens-before edge on
/// its own -- so the whole case passes with the atomics made fully `relaxed`, and
/// the **outer, lock-free fast path** is reached only if a straggler's check
/// happens to observe `ready_ == true` while still racing the writer. Measured on
/// exactly that shape: with `lazy.hpp` patched to `memory_order_relaxed`
/// throughout, the sanitizer caught the injected race in roughly one run in three
/// and reported nothing in the others, so a single `ctest --preset gcc-tsan` had a
/// real chance of passing on a broken acquire/release pair. That is the line the
/// design note calls load-bearing, and it was the line least covered.
///
/// So: many independent slots, and every thread walks all of them starting from a
/// different offset. At any moment some threads are building slot `i` while others
/// are reading a slot already published, which is what puts traffic through the
/// fast path rather than through the mutex. The second wave then reads every slot
/// again, when all of them are published and nothing takes the lock at all.
void check_concurrent_first_touch() {
    constexpr int num_threads = 8;
    constexpr int num_slots = 64;
    constexpr std::size_t width = 512;

    std::vector<LazySlot<std::vector<std::int64_t>>> slots(num_slots);
    std::atomic<int> builds{0};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::int64_t> sums(static_cast<std::size_t>(num_threads) * 2, -1);

    const auto build_slot = [&builds](int slot) {
        return [&builds, slot] {
            builds.fetch_add(1, std::memory_order_relaxed);
            std::vector<std::int64_t> built(width);
            for (std::size_t i = 0; i < width; ++i) {
                built[i] = static_cast<std::int64_t>(i) + slot;
            }
            return built;
        };
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            // A spin barrier rather than a sleep: the window this is meant to open
            // is the one between the outer load and the release store, and a sleep
            // makes it a race between one thread and nothing.
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
            }
            for (int wave = 0; wave < 2; ++wave) {
                std::int64_t total = 0;
                for (int step = 0; step < num_slots; ++step) {
                    // Each thread starts at its own offset, so the threads are
                    // spread across the slots rather than queued on one mutex.
                    const int slot = (t * (num_slots / num_threads) + step) % num_slots;
                    const auto& value =
                        slots[static_cast<std::size_t>(slot)].get(build_slot(slot));
                    for (const std::int64_t entry : value) {
                        total += entry;
                    }
                }
                sums[static_cast<std::size_t>(t * 2 + wave)] = total;
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < num_threads) {
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Per slot `s`: `sum(i + s) for i in [0, width)`, summed over every slot.
    // Computed here rather than copied from a run.
    constexpr std::int64_t w = static_cast<std::int64_t>(width);
    constexpr std::int64_t n = static_cast<std::int64_t>(num_slots);
    constexpr std::int64_t expected = n * (w * (w - 1) / 2) + w * (n * (n - 1) / 2);

    PANTR_CHECK_MSG(builds.load() == num_slots,
                    "each slot must be built exactly once, whatever the interleaving");
    for (std::size_t i = 0; i < sums.size(); ++i) {
        PANTR_CHECK_MSG(sums[i] == expected,
                        "thread " + std::to_string(i / 2) + " wave " + std::to_string(i % 2)
                            + " read a wrong total");
    }
}

}  // namespace

int main() {
    check_built_once();
    check_a_throwing_builder_leaves_it_cold();
    check_copies_start_cold();
    check_concurrent_first_touch();
    return pantr::test::summary("test_lazy");
}
