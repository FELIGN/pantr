#pragma once

/// \file
/// A once-filled memo that any number of threads may reach through a `const` accessor.
///
/// ## Why this is a type rather than three lines in each class that needs it
///
/// A derived quantity that allocates is computed lazily here: eagerly is what the
/// scalars get, because they cost nothing, and `design/bspline_derived_caches.md`
/// measured that the allocating ones do not. That leaves a memo filled on first use
/// behind a `const` accessor, and the naive spelling of it -- a bare
/// `mutable std::optional<T>` tested for `has_value()` -- is a **data race**: two
/// threads constructing a `T` into one `std::optional` is undefined behaviour, not a
/// write one of them wins. The same note measured that shape under a thread sanitizer
/// and recorded both halves of the trap: the sanitizer reports races, and every run
/// without it answers correctly. **A passing value test cannot see this**, which is
/// exactly why it belongs in a type checked once rather than in a pattern re-typed per
/// class.
///
/// The reading the C++ shape was borrowed from is the *Python* one, where the
/// interpreter makes the assignment atomic so the losing thread's object is merely
/// discarded. That reason does not transfer, and pantr's C++ is written for a consumer
/// with no interpreter above it.
///
/// ## The mechanism, and which parts of it are load-bearing
///
/// Double-checked locking over an `std::atomic<bool>`, which
/// `design/bspline_derived_caches.md` chose over the alternatives on measurement:
/// `std::call_once` costs a `FUTEX_WAKE` on its first call even with no waiters, and an
/// atomic pointer published with `compare_exchange` forces the value onto the heap and
/// adds an indirection to every later read. The note carries the figures, the machine
/// and the commands; they are deliberately not repeated here, because a timing in a
/// comment rots while reading as current.
///
/// Three details are the correctness rather than the ceremony:
///
///  - **The `acquire` load pairs with the `release` store.** The store publishes every
///    write the build made; the load is what a reader synchronises with. Relaxing
///    either restores the race with the machinery still visible.
///  - **The second load, inside the lock, is `relaxed` and correct as such** -- the
///    mutex already orders it.
///  - **The value is read outside the lock on the fast path**, which is sound for the
///    same pairing: the reading thread's acquire load synchronises with the filling
///    thread's release store, so the write to the value happens-before the read.
///
/// ## The contract
///
/// > `get_or_build` and `is_filled` are safe to call concurrently on one memo from any
/// > number of threads. The build runs at most once and its result is published
/// > atomically. Every **mutating** operation -- construction, assignment, destruction
/// > -- is not, and needs the caller's own exclusion.
///
/// The build callable must not itself reach `get_or_build` on the same memo: the mutex
/// is not recursive, so that is a deadlock rather than a diagnostic.
///
/// ## Rule of zero, restored one level down
///
/// A `std::mutex` member deletes its enclosing class's copy and move operations, which
/// would push the special members of every class holding a memo out of the implicit
/// path -- exactly where a hand-written one forgets a member. Defining them **here**,
/// once, lets `GridBase` and its successors stay at rule of zero. A copy carries the
/// value across if the source already holds one, which is what the plain
/// `std::optional` member did before; a move leaves the source empty.

#include <atomic>
#include <concepts>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace pantr {

/// A lazily filled, once-only memo, safe to fill concurrently through a `const` handle.
///
/// \tparam T The memoised value. Must be move-constructible, so that a build result can
///           be placed into the memo and a move of the memo can carry it out.
template <class T>
    requires std::move_constructible<T>
class LazyMemo {
  public:
    /// Construct an empty memo.
    LazyMemo() = default;

    /// Copy a memo, carrying the value across if the source already holds one.
    ///
    /// \param other The memo to copy. Its lock is taken for the read, so this is safe
    ///        against a concurrent `get_or_build` on `other`.
    LazyMemo(const LazyMemo& other)
        requires std::copy_constructible<T>
        : value_(other.locked_copy()) {
        ready_.store(value_.has_value(), std::memory_order_relaxed);
    }

    /// Move a memo, leaving the source empty.
    ///
    /// The source's lock is **not** taken: a move mutates the source, so the caller
    /// already owes exclusion on it, and taking the lock would buy nothing the caller
    /// has not already had to provide.
    ///
    /// \param other The memo to move from; left empty.
    LazyMemo(LazyMemo&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        : value_(std::move(other.value_)) {
        ready_.store(value_.has_value(), std::memory_order_relaxed);
        other.value_.reset();
        other.ready_.store(false, std::memory_order_relaxed);
    }

    /// Replace this memo's contents with a copy of another's.
    ///
    /// The source's lock is released before this memo's is taken, so two memos assigned
    /// to each other concurrently cannot deadlock.
    ///
    /// \param other The memo to copy from.
    /// \return This memo.
    LazyMemo& operator=(const LazyMemo& other)
        requires std::copy_constructible<T>
    {
        if (this != &other) {
            std::optional<T> incoming = other.locked_copy();
            const std::lock_guard<std::mutex> guard(mutex_);
            value_ = std::move(incoming);
            ready_.store(value_.has_value(), std::memory_order_release);
        }
        return *this;
    }

    /// Replace this memo's contents with another's, leaving the source empty.
    ///
    /// \param other The memo to move from; left empty.
    /// \return This memo.
    LazyMemo& operator=(LazyMemo&& other) {
        if (this != &other) {
            std::optional<T> incoming = std::move(other.value_);
            other.value_.reset();
            other.ready_.store(false, std::memory_order_relaxed);
            const std::lock_guard<std::mutex> guard(mutex_);
            value_ = std::move(incoming);
            ready_.store(value_.has_value(), std::memory_order_release);
        }
        return *this;
    }

    /// Return the memoised value, building it on the first call.
    ///
    /// \tparam Build A callable of no arguments returning something a `T` can be
    ///         assigned from.
    /// \param build Runs at most once, however many threads arrive together. If it
    ///        throws, the memo stays empty and the next call tries again.
    /// \return A reference to the stored value, valid for as long as this memo is and
    ///         never replaced once filled.
    template <class Build>
        requires std::invocable<Build> && std::assignable_from<T&, std::invoke_result_t<Build>>
    [[nodiscard]] const T& get_or_build(Build&& build) const {
        if (!ready_.load(std::memory_order_acquire)) {
            const std::lock_guard<std::mutex> guard(mutex_);
            if (!ready_.load(std::memory_order_relaxed)) {
                value_ = std::forward<Build>(build)();
                ready_.store(true, std::memory_order_release);
            }
        }
        return *value_;
    }

    /// Report whether the value has been built.
    ///
    /// For a caller that wants to know whether the cost has been paid, and for the
    /// tests that assert laziness. Not a permission to skip `get_or_build`.
    ///
    /// \return `true` once the build has completed.
    [[nodiscard]] bool is_filled() const noexcept {
        return ready_.load(std::memory_order_acquire);
    }

  private:
    /// Copy the stored value under this memo's lock.
    ///
    /// \return The value if one is held, an empty optional otherwise.
    [[nodiscard]] std::optional<T> locked_copy() const
        requires std::copy_constructible<T>
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return value_;
    }

    mutable std::mutex mutex_;              ///< Serialises the build; never held by a reader.
    mutable std::atomic<bool> ready_{false};  ///< Publishes `value_`; acquire/release paired.
    mutable std::optional<T> value_;        ///< The memoised value, written once.
};

}  // namespace pantr
