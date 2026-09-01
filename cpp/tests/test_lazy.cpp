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

/// Eight threads first-touching one slot at once. The sanitizer's target.
void check_concurrent_first_touch() {
    constexpr int num_threads = 8;
    constexpr std::size_t width = 4096;

    LazySlot<std::vector<std::int64_t>> slot;
    std::atomic<int> builds{0};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<const std::vector<std::int64_t>*> seen(num_threads, nullptr);
    std::vector<std::int64_t> totals(num_threads, -1);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            // A spin barrier rather than a sleep: the window this is meant to open
            // is the one between the first `acquire` load and the `release` store,
            // and a sleep makes it a race between one thread and nothing.
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
            }
            const std::vector<std::int64_t>& value = slot.get([&builds] {
                builds.fetch_add(1, std::memory_order_relaxed);
                std::vector<std::int64_t> built(width);
                for (std::size_t i = 0; i < width; ++i) {
                    built[i] = static_cast<std::int64_t>(i);
                }
                return built;
            });
            seen[static_cast<std::size_t>(t)] = &value;
            std::int64_t total = 0;
            for (const std::int64_t entry : value) {
                total += entry;
            }
            totals[static_cast<std::size_t>(t)] = total;
        });
    }

    while (ready.load(std::memory_order_acquire) < num_threads) {
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    // `width * (width - 1) / 2`, computed here rather than copied from a run.
    constexpr std::int64_t expected =
        static_cast<std::int64_t>(width) * (static_cast<std::int64_t>(width) - 1) / 2;
    PANTR_CHECK_MSG(builds.load() == 1, "the value must be built exactly once");
    for (int t = 0; t < num_threads; ++t) {
        const auto index = static_cast<std::size_t>(t);
        PANTR_CHECK_MSG(seen[index] == seen[0], "every thread must see one object");
        PANTR_CHECK_MSG(totals[index] == expected,
                        "thread " + std::to_string(t) + " read a wrong total");
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
