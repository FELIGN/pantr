#pragma once

/// \file
/// A derived value computed at most once, from any number of threads.
///
/// ## The contract this exists to make true
///
/// > Every non-mutating public accessor on a C++-owned pantr domain type is safe
/// > to call concurrently from any number of threads on the same object, with no
/// > external locking. A lazy memo behind such an accessor is filled at most once
/// > and published atomically.
///
/// That sentence is `design/bspline_derived_caches.md`'s, and it is the reason
/// this is a type rather than three lines copied into each owner. The port has at
/// least five of these coming -- a grid's bounding-volume hierarchy, a 1D space's
/// derived knot arrays, a THB space's contribution table, a multi-level
/// extraction's per-level spaces -- and the shape that gets copied by hand is the
/// bare `mutable std::optional`, which is a **data race** rather than a benign
/// lost update.
///
/// ## Why not the two obvious spellings
///
/// **A bare `mutable std::optional`, filled on first `const` access.** Measured
/// under g++ 14.4 `-fsanitize=thread` with 8 threads first-touching one memo: 4
/// data races, every frame in the unsynchronised accessor -- one on the
/// `optional`'s engaged flag and the rest on the contained vector's pointer
/// triple. **And measured without the sanitizer: 60 runs of 8 threads produced
/// the correct answer 60 times.** That is the finding, not the exoneration: two
/// threads assigning one `std::optional<std::vector<double>>` can leave a torn
/// pointer triple, double-free the loser's buffer, or leak it, and no test that
/// checks a value will ever see it.
///
/// **`std::call_once`.** Correct, idiomatic, and measured at one to a few
/// microseconds on its *first* call, size-independently -- roughly ten times the
/// entire construction of a small B-spline space. `strace` on a minimal program
/// shows glibc's `pthread_once` issuing `futex(..., FUTEX_WAKE_PRIVATE, INT_MAX)`
/// on that first call with zero waiters. Double-checked locking is the same
/// guarantee without that syscall: measured within noise of computing eagerly on
/// first use, and 3 to 8 ns per access thereafter.
///
/// ## Reading it correctly in a hot loop
///
/// The accessor looks free and is not. A sweep over cells or intervals should
/// hoist the `get` out of the loop into a local reference **once** and take a span
/// before the loop rather than per element; a few nanoseconds per element over an
/// `n * (p + 1)` sweep is a measurable tax for no reason, and the fix is a local
/// rather than a mechanism.

#include <atomic>
#include <mutex>
#include <optional>
#include <utility>

namespace pantr {

/// A `T` built on first use and then never again.
///
/// Copy and move leave the target **cold** rather than carrying the source's
/// value across. That is not laziness about implementing it: the memo is a
/// function of its owner's state, the owner is what is being copied, and a memo
/// that travelled would have to be proved still to describe the target. Assignment
/// clears the target for the same reason, which matters because assignment is what
/// replaces the owner's state underneath it.
///
/// **The slot is not itself thread-safe against assignment.** `get` is safe
/// against any number of concurrent `get`s; assigning or destroying the slot while
/// another thread reads it is a data race like any other write to a shared object,
/// and the owning types here are immutable after construction so it does not
/// arise. `cpp/include/pantr/grid/tags.hpp` says the same of the one mutable
/// member the domain layer keeps.
///
/// \tparam T The derived value. Must be movable; it is constructed by the builder
///         and moved into the slot.
template <class T>
class LazySlot {
  public:
    /// Start cold.
    LazySlot() noexcept = default;

    /// Copy the owner, not the memo: the new slot starts cold.
    LazySlot(const LazySlot& /*other*/) noexcept {}

    /// Move the owner, not the memo: the new slot starts cold and the source is
    /// left exactly as it was, which is legal for a moved-from object. Leaving it
    /// alone is the class-level semantics rather than an exception-safety
    /// necessity -- clearing the source would be `noexcept` too.
    LazySlot(LazySlot&& /*other*/) noexcept {}

    /// Clear this slot: the value it held described the state being overwritten.
    LazySlot& operator=(const LazySlot& other) noexcept {
        if (this != &other) {
            clear();
        }
        return *this;
    }

    /// Clear this slot, for the reason copy assignment does.
    LazySlot& operator=(LazySlot&& other) noexcept {
        if (this != &other) {
            clear();
        }
        return *this;
    }

    ~LazySlot() = default;

    /// The value, building it on the first call.
    ///
    /// The `acquire` load and the `release` store are the pair that makes this
    /// correct: the release store publishes every write the builder made, and the
    /// acquire load is what a reader synchronises with. Dropping either, or making
    /// both `relaxed`, restores the race with the ceremony still visible. The
    /// second load is inside the lock and is `relaxed` because the lock already
    /// orders it.
    ///
    /// A builder that throws leaves the slot cold and the lock released, so the
    /// next caller retries rather than reading a half-built value.
    ///
    /// \param build A callable returning the value; invoked at most once.
    /// \return The stored value, valid for the lifetime of this slot.
    template <class F>
    [[nodiscard]] const T& get(F&& build) const {
        if (!ready_.load(std::memory_order_acquire)) {
            const std::lock_guard<std::mutex> guard(mutex_);
            if (!ready_.load(std::memory_order_relaxed)) {
                value_.emplace(std::forward<F>(build)());
                ready_.store(true, std::memory_order_release);
            }
        }
        return *value_;
    }

    /// Whether the value has been built.
    ///
    /// For tests and diagnostics. A caller that branches on this is asking a
    /// question whose answer can change before it is used.
    ///
    /// \return `true` once `get` has returned at least once.
    [[nodiscard]] bool filled() const noexcept { return ready_.load(std::memory_order_acquire); }

  private:
    /// Drop the value and mark the slot cold.
    void clear() noexcept {
        value_.reset();
        ready_.store(false, std::memory_order_release);
    }

    mutable std::mutex mutex_;                ///< Held only while the value is built.
    mutable std::atomic<bool> ready_{false};  ///< Whether `value_` has been published.
    mutable std::optional<T> value_;          ///< The value, once built.
};

}  // namespace pantr
