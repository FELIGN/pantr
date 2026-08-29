#pragma once

/// \file
/// Sparse named integer tags over a grid's cells and over its local facets.
///
/// Ports `src/pantr/grid/_tags.py`, whose classes stay as the Python backend under
/// `PANTR_BACKEND=python`. Ownership moves under `design/cross_backend_types.md`'s
/// 2026-08-27 amendment: there is one registry, it is this one, and
/// `pantr.grid.CellTags` wraps it.
///
/// ## The accumulating container, and why it is allowed to mutate
///
/// `CLAUDE.md` says an instance is immutable once created and that a `fit()` which
/// mutates `self` is the anti-pattern. These two are the reasoned exception it
/// names: a tag registry *is* an accumulating container, `set` and `remove` are its
/// whole purpose, and the grid holds one as a member precisely so that a C++
/// program with no interpreter can tag cells as it classifies them.
///
/// ## Four things the port has to get right
///
/// **1. A handed-out view must outlive a replacement.** `get` returns the stored
/// arrays, and the binding hands them to numpy as zero-copy views. If those viewed
/// a `std::vector` owned directly by the registry, a later `set` on the same name
/// would free the storage under a live numpy array -- a use-after-free the oracle
/// cannot have, because Python's own reference counting keeps the replaced arrays
/// alive. So each tag is held behind a `std::shared_ptr<const Tag>`: `set`
/// *replaces the pointer* rather than the buffer, and any view that took a copy of
/// the pointer keeps its own storage alive. Nothing here is thread-safe; the
/// shared pointer buys lifetime, not concurrency.
///
/// **2. `set` on an existing name keeps that name's position.** `__iter__` and
/// `names` are public, so the order is a contract and both backends have to agree
/// on it. Python's `dict` replaces in place, so the entries are held in a vector
/// and a replacement overwrites the entry it found rather than appending. A
/// `std::unordered_map` plus a separate order vector would work too and was not
/// chosen: a registry holds a handful of tags, so a linear scan over a contiguous
/// vector is both faster and impossible to get out of step with itself.
///
/// **3. A missing name is a `KeyError`, and that is raised in the Python wrapper.**
/// `pantr/core/error.hpp` records the rule for the whole port: nanobind 2.14.0's
/// default translator maps `std::out_of_range` to `IndexError` and has no path to
/// `KeyError` at all. The methods below throw `std::out_of_range`, which is right
/// for a C++ caller; the wrapper checks `contains` first, so no Python caller ever
/// reaches it.
///
/// **4. `scatter` writes into the caller's destination integer type**, which is
/// what the oracle's `to_dense(dtype=...)` parameter means. The *kind* check --
/// that the destination dtype is an integer at all -- is a `TypeError` and lives in
/// the wrapper, because nanobind cannot raise one. The *range* check is an
/// `OverflowError` and lives here, in the template, where the destination type is
/// known. Two places, two different questions; the docstrings on both sides say so
/// rather than claiming validation happens in one place.
///
/// ## What `scatter` deliberately does not do
///
/// It does not fill. The oracle's `to_dense` allocates with `numpy.full(n, fill,
/// dtype)` and then scatters, and `numpy.full` is itself the thing that decides
/// what an out-of-range `fill` does -- it raises `OverflowError` with a message
/// this side would have to reproduce by hand. So the wrapper keeps the allocation
/// and the fill, which is the project's Layer 2 / `out` convention anyway, and this
/// method owns only the part that needs the tag: writing the stored values at the
/// stored positions.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "pantr/core/mdspan.hpp"

namespace pantr::grid {

namespace detail {

/// A facet key is the pair `(cell_id, local_facet_id)`; stored as `(M, 2)` rows.
inline constexpr std::size_t kFacetKeyWidth = 2;

/// A valid, never-dereferenced address for a zero-length array's data pointer.
///
/// `std::vector::data()` on an empty vector may return `nullptr`, and nanobind
/// treats a null data pointer as "no array" rather than as an empty one. An empty
/// tag is legal -- `_tags.py`'s own tests set one -- so the binding needs an
/// address it can hand over for a shape of `(0,)`.
inline constexpr std::int64_t kEmptyStorage = 0;

/// Render a tag name the way Python's `repr` renders a `str`.
///
/// The oracle's duplicate-key messages interpolate `{name!r}`, so reproducing them
/// means reproducing that quoting. Python prefers single quotes and switches to
/// double quotes only for a string that contains a single quote and no double one;
/// backslash, the delimiter and the three common control characters are escaped.
///
/// **The limit is deliberate and worth stating:** a name containing some *other*
/// control or non-ASCII character prints as itself here, where Python would print a
/// `\xNN` or `\uNNNN` escape. Tag names come from user code and are identifiers in
/// every use in this repository, so the divergence needs a name nobody writes; the
/// alternative is a transliteration of CPython's `unicode_repr`, which is a lot of
/// code for an error message.
///
/// \param name The tag name.
/// \return Its Python-style quoted form, delimiters included.
[[nodiscard]] inline std::string quote_name(std::string_view name) {
    const bool has_single = name.find('\'') != std::string_view::npos;
    const bool has_double = name.find('"') != std::string_view::npos;
    const char delim = (has_single && !has_double) ? '"' : '\'';
    std::string out(1, delim);
    for (const char c : name) {
        if (c == '\\' || c == delim) {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    out += delim;
    return out;
}

/// numpy's name for the destination integer type, as `numpy.dtype.__repr__` shows it.
///
/// The oracle's overflow message interpolates `{out_dtype!r}`, which is
/// `dtype('int8')` and not `int8`, so the quoting is part of the message rather
/// than decoration.
///
/// \tparam IntT The destination integer type.
/// \return The dtype name, without the surrounding `dtype('...')`.
template <class IntT>
[[nodiscard]] consteval std::string_view numpy_dtype_name() {
    if constexpr (std::is_same_v<IntT, std::int8_t>) {
        return "int8";
    } else if constexpr (std::is_same_v<IntT, std::int16_t>) {
        return "int16";
    } else if constexpr (std::is_same_v<IntT, std::int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<IntT, std::int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<IntT, std::uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<IntT, std::uint16_t>) {
        return "uint16";
    } else if constexpr (std::is_same_v<IntT, std::uint32_t>) {
        return "uint32";
    } else {
        static_assert(std::is_same_v<IntT, std::uint64_t>,
                      "scatter's destination must be one of numpy's eight integer types");
        return "uint64";
    }
}

/// Reject stored values the destination integer type cannot hold.
///
/// **The `sizeof(IntT) < 8` guard reproduces the oracle exactly, including where the
/// oracle is wrong.** `_tags.py` runs its check only when
/// `out_dtype.itemsize < 8`, and its docstring states that limit as the contract:
/// "only raised when `dtype` is narrower than `int64`". That is sound for `int64`,
/// whose range contains every stored value, and unsound for `uint64`, which is
/// eight bytes wide and cannot hold a negative one -- so a negative tag value
/// scattered into a `uint64` destination wraps silently, in the oracle and
/// therefore here. Reported as a contract gap rather than fixed: closing it changes
/// documented behaviour in both backends at once, which is not a port's decision.
///
/// \tparam IntT The destination integer type.
/// \param values The stored values.
/// \param who The registry name, for the message.
/// \throws std::overflow_error If some value is outside `IntT`'s range.
template <class IntT>
void require_representable(std::span<const std::int64_t> values, const char* who) {
    (void)who;
    if constexpr (sizeof(IntT) < 8) {
        if (values.empty()) {
            return;
        }
        const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        const std::int64_t vmin = *min_it;
        const std::int64_t vmax = *max_it;
        constexpr auto lo = static_cast<std::int64_t>(std::numeric_limits<IntT>::min());
        constexpr auto hi = static_cast<std::int64_t>(std::numeric_limits<IntT>::max());
        if (vmin < lo || vmax > hi) {
            throw std::overflow_error(
                "dtype dtype('" + std::string(numpy_dtype_name<IntT>())
                + "') cannot represent all tag values without truncation; value range ["
                + std::to_string(vmin) + ", " + std::to_string(vmax) + "] exceeds dtype range ["
                + std::to_string(lo) + ", " + std::to_string(hi) + "].");
        }
    }
}

}  // namespace detail

/// Sparse named integer tags over a grid's cells.
///
/// Each tag named `name` is a pair of parallel `int64` arrays `(ids, values)`
/// sorted by `ids`; a cell not listed in `ids` is untagged under `name`. Distinct
/// tag names are independent, and the names keep their insertion order under
/// iteration -- including across a replacement, which is a contract rather than an
/// accident. See the file comment for the four decisions this reproduces.
class CellTags {
  public:
    /// One tag's stored arrays.
    ///
    /// Held behind a `std::shared_ptr<const Tag>` so that a view handed to a caller
    /// survives a later `set` on the same name; see the file comment.
    ///
    /// Attributes are public because this is a value aggregate with no invariant of
    /// its own: `CellTags::set` establishes the sorting and the uniqueness before
    /// one is ever built, and nothing can reach a `Tag` except through a
    /// `shared_ptr<const Tag>`.
    struct Tag {
        std::vector<std::int64_t> ids;     ///< Cell ids, ascending and unique.
        std::vector<std::int64_t> values;  ///< The value at each id, same length.
    };

    /// Create an empty cell-tag registry.
    ///
    /// \param num_cells Number of cells in the owning grid, `>= 0`.
    /// \throws std::invalid_argument If `num_cells` is negative.
    explicit CellTags(std::int64_t num_cells) : num_cells_(num_cells) {
        if (num_cells < 0) {
            throw std::invalid_argument("num_cells must be >= 0; got "
                                        + std::to_string(num_cells) + ".");
        }
    }

    /// The number of cells in the owning grid.
    ///
    /// \return The cell count; valid cell ids are `[0, num_cells)`.
    [[nodiscard]] std::int64_t num_cells() const noexcept { return num_cells_; }

    /// The number of registered tags.
    ///
    /// \return Count of distinct tag names.
    [[nodiscard]] std::size_t size() const noexcept { return tags_.size(); }

    /// The registered tag names, in insertion order.
    ///
    /// \return The names; a replaced tag keeps the position it first took.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(tags_.size());
        for (const Entry& entry : tags_) {
            out.push_back(entry.name);
        }
        return out;
    }

    /// Whether a tag named `name` exists.
    ///
    /// \param name Candidate tag name.
    /// \return `true` when `name` is registered.
    [[nodiscard]] bool contains(std::string_view name) const noexcept {
        return find(name) != tags_.end();
    }

    /// Create or replace the tag `name` with the association `ids -> values`.
    ///
    /// A replacement keeps the name's position among `names()`; see the file
    /// comment for why that is a contract.
    ///
    /// \param name Tag name.
    /// \param ids Cell ids, each in `[0, num_cells)` and unique. Order is free;
    ///        the stored pair is sorted by id.
    /// \param values The value for each id, same length as `ids`.
    /// \throws std::invalid_argument If an id is out of range, if two ids are
    ///         equal, or if the two lengths differ.
    void set(std::string_view name, std::span<const std::int64_t> ids,
             std::span<const std::int64_t> values) {
        require_ids_in_range(ids);
        const std::vector<std::size_t> order = sorted_order(ids);
        require_unique_ids(name, ids, order);
        if (values.size() != ids.size()) {
            throw std::invalid_argument("values must be a scalar or have length "
                                        + std::to_string(ids.size()) + "; got length "
                                        + std::to_string(values.size()) + ".");
        }
        auto tag = std::make_shared<Tag>();
        tag->ids.reserve(ids.size());
        tag->values.reserve(values.size());
        for (const std::size_t k : order) {
            tag->ids.push_back(ids[k]);
            tag->values.push_back(values[k]);
        }
        store(name, std::move(tag));
    }

    /// The stored arrays for tag `name`.
    ///
    /// \param name Tag name.
    /// \return A shared handle to the pair, which keeps its storage alive
    ///         independently of any later `set` or `remove`.
    /// \throws std::out_of_range If no tag named `name` exists. Reaches Python as
    ///         `IndexError`, which is why the wrapper checks `contains` first and
    ///         raises `KeyError` itself.
    [[nodiscard]] std::shared_ptr<const Tag> get(std::string_view name) const {
        const auto it = find(name);
        if (it == tags_.end()) {
            throw std::out_of_range(std::string(name));
        }
        return it->tag;
    }

    /// Delete the tag `name`.
    ///
    /// \param name Tag name.
    /// \throws std::out_of_range If no tag named `name` exists.
    void remove(std::string_view name) {
        const auto it = find(name);
        if (it == tags_.end()) {
            throw std::out_of_range(std::string(name));
        }
        tags_.erase(it);
    }

    /// Write tag `name`'s values into `out` at its ids.
    ///
    /// Does **not** fill: every entry not named by the tag is left as the caller
    /// left it. See the file comment for why the fill stays with the caller.
    ///
    /// \tparam IntT Destination integer type.
    /// \param name Tag name.
    /// \param out Destination, length `num_cells`.
    /// \throws std::out_of_range If no tag named `name` exists.
    /// \throws std::invalid_argument If `out` is not `num_cells` long.
    /// \throws std::overflow_error If a stored value is outside `IntT`'s range,
    ///         subject to the width limit `detail::require_representable` documents.
    template <class IntT>
    void scatter(std::string_view name, std::span<IntT> out) const {
        const std::shared_ptr<const Tag> tag = get(name);
        if (out.size() != static_cast<std::size_t>(num_cells_)) {
            throw std::invalid_argument("scatter: out must have length "
                                        + std::to_string(num_cells_) + "; got "
                                        + std::to_string(out.size()) + ".");
        }
        detail::require_representable<IntT>(tag->values, "CellTags");
        for (std::size_t k = 0; k < tag->ids.size(); ++k) {
            out[static_cast<std::size_t>(tag->ids[k])] = static_cast<IntT>(tag->values[k]);
        }
    }

    /// A compact representation naming the cell count and the registered tags.
    ///
    /// \return `"CellTags(num_cells=..., tags=[...])"`.
    [[nodiscard]] std::string to_string() const {
        std::string text = "CellTags(num_cells=" + std::to_string(num_cells_) + ", tags=[";
        for (std::size_t k = 0; k < tags_.size(); ++k) {
            if (k != 0) {
                text += ", ";
            }
            text += detail::quote_name(tags_[k].name);
        }
        return text + "])";
    }

  private:
    /// One registered tag: its name and its shared storage.
    struct Entry {
        std::string name;                ///< The tag name.
        std::shared_ptr<const Tag> tag;  ///< Its arrays, shared with any live view.
    };

    /// Locate a tag by name.
    ///
    /// \param name The name to look for.
    /// \return An iterator to the entry, or `tags_.end()`.
    [[nodiscard]] std::vector<Entry>::const_iterator find(std::string_view name) const noexcept {
        return std::find_if(tags_.begin(), tags_.end(),
                            [name](const Entry& e) { return e.name == name; });
    }

    /// Insert a tag, or replace one in place without moving its position.
    ///
    /// \param name The tag name.
    /// \param tag The storage to adopt.
    void store(std::string_view name, std::shared_ptr<const Tag> tag) {
        const auto it = find(name);
        if (it == tags_.end()) {
            tags_.push_back(Entry{std::string(name), std::move(tag)});
            return;
        }
        tags_[static_cast<std::size_t>(it - tags_.begin())].tag = std::move(tag);
    }

    /// Reject a cell id outside `[0, num_cells)`.
    ///
    /// \param ids The ids to check.
    /// \throws std::invalid_argument If any id is out of range.
    void require_ids_in_range(std::span<const std::int64_t> ids) const {
        if (ids.empty()) {
            return;
        }
        const auto [min_it, max_it] = std::minmax_element(ids.begin(), ids.end());
        if (*min_it < 0 || *max_it >= num_cells_) {
            throw std::invalid_argument("cell ids must be in [0, " + std::to_string(num_cells_)
                                        + "); got range [" + std::to_string(*min_it) + ", "
                                        + std::to_string(*max_it) + "].");
        }
    }

    /// The permutation that sorts `ids` ascending, ties in increasing index order.
    ///
    /// \param ids The ids to order.
    /// \return Indices into `ids`.
    [[nodiscard]] static std::vector<std::size_t> sorted_order(
        std::span<const std::int64_t> ids) {
        std::vector<std::size_t> order(ids.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [ids](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
        return order;
    }

    /// Reject a repeated cell id.
    ///
    /// Checked on the sorted permutation, so equal ids are adjacent.
    ///
    /// \param name The tag name, for the message.
    /// \param ids The ids.
    /// \param order The permutation sorting them.
    /// \throws std::invalid_argument If two ids are equal.
    static void require_unique_ids(std::string_view name, std::span<const std::int64_t> ids,
                                   const std::vector<std::size_t>& order) {
        for (std::size_t k = 1; k < order.size(); ++k) {
            if (ids[order[k]] == ids[order[k - 1]]) {
                throw std::invalid_argument("cell ids for tag " + detail::quote_name(name)
                                            + " must be unique; got duplicates.");
            }
        }
    }

    std::int64_t num_cells_;    ///< The owning grid's cell count.
    std::vector<Entry> tags_;   ///< The registered tags, in insertion order.
};

/// Sparse named integer tags over a grid's local facets.
///
/// Each facet is addressed by a `(cell_id, local_facet_id)` key with
/// `local_facet_id` in `[0, facets_per_cell)`. Each tag is a pair `(keys, values)`
/// where `keys` is an `(M, 2)` block of `(cell_id, local_facet_id)` rows sorted
/// lexicographically, and `values` is the length-`M` companion. The same four
/// decisions as `CellTags` apply; see the file comment.
class FacetTags {
  public:
    /// One tag's stored arrays.
    ///
    /// As `CellTags::Tag`, with the keys held flat in row-major `(M, 2)` order.
    struct Tag {
        std::vector<std::int64_t> keys;    ///< `(M, 2)` rows, flattened, sorted.
        std::vector<std::int64_t> values;  ///< The value at each key, length `M`.

        /// The number of tagged facets.
        ///
        /// \return `M`, the row count of `keys`.
        [[nodiscard]] std::size_t rows() const noexcept { return values.size(); }
    };

    /// Create an empty facet-tag registry.
    ///
    /// \param num_cells Number of cells in the owning grid, `>= 0`.
    /// \param facets_per_cell Number of local facets per cell, `>= 1`.
    /// \throws std::invalid_argument If `num_cells` is negative or
    ///         `facets_per_cell` is below one.
    FacetTags(std::int64_t num_cells, std::int64_t facets_per_cell)
        : num_cells_(num_cells), facets_per_cell_(facets_per_cell) {
        if (num_cells < 0) {
            throw std::invalid_argument("num_cells must be >= 0; got "
                                        + std::to_string(num_cells) + ".");
        }
        if (facets_per_cell < 1) {
            throw std::invalid_argument("facets_per_cell must be >= 1; got "
                                        + std::to_string(facets_per_cell) + ".");
        }
    }

    /// The number of cells in the owning grid.
    ///
    /// \return The cell count.
    [[nodiscard]] std::int64_t num_cells() const noexcept { return num_cells_; }

    /// The number of local facets per cell.
    ///
    /// \return `2 * ndim` for an axis-aligned box grid.
    [[nodiscard]] std::int64_t facets_per_cell() const noexcept { return facets_per_cell_; }

    /// The number of registered tags.
    ///
    /// \return Count of distinct tag names.
    [[nodiscard]] std::size_t size() const noexcept { return tags_.size(); }

    /// The registered tag names, in insertion order.
    ///
    /// \return The names; a replaced tag keeps the position it first took.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(tags_.size());
        for (const Entry& entry : tags_) {
            out.push_back(entry.name);
        }
        return out;
    }

    /// Whether a tag named `name` exists.
    ///
    /// \param name Candidate tag name.
    /// \return `true` when `name` is registered.
    [[nodiscard]] bool contains(std::string_view name) const noexcept {
        return find(name) != tags_.end();
    }

    /// Create or replace the tag `name` with the association `keys -> values`.
    ///
    /// \param name Tag name.
    /// \param keys `(M, 2)` rows of `(cell_id, local_facet_id)`, each row unique,
    ///        with `cell_id` in `[0, num_cells)` and `local_facet_id` in
    ///        `[0, facets_per_cell)`.
    /// \param values The value for each row, length `M`.
    /// \throws std::invalid_argument If `keys` does not have two columns, if a key
    ///         component is out of range, if two rows are equal, or if the lengths
    ///         disagree.
    void set(std::string_view name, span2d<const std::int64_t> keys,
             std::span<const std::int64_t> values) {
        if (keys.extent(1) != detail::kFacetKeyWidth) {
            throw std::invalid_argument("keys must have shape (M, 2); got shape ("
                                        + std::to_string(keys.extent(0)) + ", "
                                        + std::to_string(keys.extent(1)) + ").");
        }
        const std::size_t rows = keys.extent(0);
        require_keys_in_range(keys);
        const std::vector<std::size_t> order = sorted_order(keys);
        require_unique_keys(name, keys, order);
        if (values.size() != rows) {
            throw std::invalid_argument("values must be a scalar or have length "
                                        + std::to_string(rows) + "; got length "
                                        + std::to_string(values.size()) + ".");
        }
        auto tag = std::make_shared<Tag>();
        tag->keys.reserve(rows * detail::kFacetKeyWidth);
        tag->values.reserve(rows);
        for (const std::size_t k : order) {
            tag->keys.push_back(at(keys, k, 0));
            tag->keys.push_back(at(keys, k, 1));
            tag->values.push_back(values[k]);
        }
        store(name, std::move(tag));
    }

    /// The stored arrays for tag `name`.
    ///
    /// \param name Tag name.
    /// \return A shared handle to the pair, which keeps its storage alive
    ///         independently of any later `set` or `remove`.
    /// \throws std::out_of_range If no tag named `name` exists.
    [[nodiscard]] std::shared_ptr<const Tag> get(std::string_view name) const {
        const auto it = find(name);
        if (it == tags_.end()) {
            throw std::out_of_range(std::string(name));
        }
        return it->tag;
    }

    /// Delete the tag `name`.
    ///
    /// \param name Tag name.
    /// \throws std::out_of_range If no tag named `name` exists.
    void remove(std::string_view name) {
        const auto it = find(name);
        if (it == tags_.end()) {
            throw std::out_of_range(std::string(name));
        }
        tags_.erase(it);
    }

    /// Write tag `name`'s values into `out` at its keys.
    ///
    /// Does **not** fill; see `CellTags::scatter` and the file comment.
    ///
    /// \tparam IntT Destination integer type.
    /// \param name Tag name.
    /// \param out Destination, shape `(num_cells, facets_per_cell)`.
    /// \throws std::out_of_range If no tag named `name` exists.
    /// \throws std::invalid_argument If `out` does not have the grid's shape.
    /// \throws std::overflow_error If a stored value is outside `IntT`'s range,
    ///         subject to the width limit `detail::require_representable` documents.
    template <class IntT>
    void scatter(std::string_view name, span2d<IntT> out) const {
        const std::shared_ptr<const Tag> tag = get(name);
        if (out.extent(0) != static_cast<std::size_t>(num_cells_)
            || out.extent(1) != static_cast<std::size_t>(facets_per_cell_)) {
            throw std::invalid_argument("scatter: out must be (" + std::to_string(num_cells_)
                                        + ", " + std::to_string(facets_per_cell_) + "); got ("
                                        + std::to_string(out.extent(0)) + ", "
                                        + std::to_string(out.extent(1)) + ").");
        }
        detail::require_representable<IntT>(tag->values, "FacetTags");
        for (std::size_t k = 0; k < tag->rows(); ++k) {
            const auto cid = static_cast<std::size_t>(tag->keys[k * detail::kFacetKeyWidth]);
            const auto lfid = static_cast<std::size_t>(tag->keys[k * detail::kFacetKeyWidth + 1]);
            at(out, cid, lfid) = static_cast<IntT>(tag->values[k]);
        }
    }

    /// A compact representation, matching the oracle's `__repr__`.
    ///
    /// \return `"FacetTags(num_cells=..., facets_per_cell=..., tags=[...])"`.
    [[nodiscard]] std::string to_string() const {
        std::string text = "FacetTags(num_cells=" + std::to_string(num_cells_)
                           + ", facets_per_cell=" + std::to_string(facets_per_cell_) + ", tags=[";
        for (std::size_t k = 0; k < tags_.size(); ++k) {
            if (k != 0) {
                text += ", ";
            }
            text += detail::quote_name(tags_[k].name);
        }
        return text + "])";
    }

  private:
    /// One registered tag: its name and its shared storage.
    struct Entry {
        std::string name;                ///< The tag name.
        std::shared_ptr<const Tag> tag;  ///< Its arrays, shared with any live view.
    };

    /// Locate a tag by name.
    ///
    /// \param name The name to look for.
    /// \return An iterator to the entry, or `tags_.end()`.
    [[nodiscard]] std::vector<Entry>::const_iterator find(std::string_view name) const noexcept {
        return std::find_if(tags_.begin(), tags_.end(),
                            [name](const Entry& e) { return e.name == name; });
    }

    /// Insert a tag, or replace one in place without moving its position.
    ///
    /// \param name The tag name.
    /// \param tag The storage to adopt.
    void store(std::string_view name, std::shared_ptr<const Tag> tag) {
        const auto it = find(name);
        if (it == tags_.end()) {
            tags_.push_back(Entry{std::string(name), std::move(tag)});
            return;
        }
        tags_[static_cast<std::size_t>(it - tags_.begin())].tag = std::move(tag);
    }

    /// Reject a key component outside its own range.
    ///
    /// The two components are checked in the oracle's order -- every cell id, then
    /// every local facet id -- because a key violating both reports the first.
    ///
    /// \param keys The `(M, 2)` keys.
    /// \throws std::invalid_argument If a component is out of range.
    void require_keys_in_range(span2d<const std::int64_t> keys) const {
        const std::size_t rows = keys.extent(0);
        if (rows == 0) {
            return;
        }
        std::int64_t cid_min = at(keys, 0, 0);
        std::int64_t cid_max = cid_min;
        std::int64_t lfid_min = at(keys, 0, 1);
        std::int64_t lfid_max = lfid_min;
        for (std::size_t k = 1; k < rows; ++k) {
            cid_min = std::min(cid_min, at(keys, k, 0));
            cid_max = std::max(cid_max, at(keys, k, 0));
            lfid_min = std::min(lfid_min, at(keys, k, 1));
            lfid_max = std::max(lfid_max, at(keys, k, 1));
        }
        if (cid_min < 0 || cid_max >= num_cells_) {
            throw std::invalid_argument("facet cell ids must be in [0, "
                                        + std::to_string(num_cells_) + "); got range ["
                                        + std::to_string(cid_min) + ", "
                                        + std::to_string(cid_max) + "].");
        }
        if (lfid_min < 0 || lfid_max >= facets_per_cell_) {
            throw std::invalid_argument("local facet ids must be in [0, "
                                        + std::to_string(facets_per_cell_) + "); got range ["
                                        + std::to_string(lfid_min) + ", "
                                        + std::to_string(lfid_max) + "].");
        }
    }

    /// The permutation sorting the keys lexicographically by `(cell_id, facet_id)`.
    ///
    /// \param keys The `(M, 2)` keys.
    /// \return Row indices into `keys`.
    [[nodiscard]] static std::vector<std::size_t> sorted_order(span2d<const std::int64_t> keys) {
        std::vector<std::size_t> order(keys.extent(0));
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [keys](std::size_t a, std::size_t b) {
            if (at(keys, a, 0) != at(keys, b, 0)) {
                return at(keys, a, 0) < at(keys, b, 0);
            }
            return at(keys, a, 1) < at(keys, b, 1);
        });
        return order;
    }

    /// Reject a repeated key.
    ///
    /// \param name The tag name, for the message.
    /// \param keys The `(M, 2)` keys.
    /// \param order The permutation sorting them, so equal rows are adjacent.
    /// \throws std::invalid_argument If two rows are equal.
    static void require_unique_keys(std::string_view name, span2d<const std::int64_t> keys,
                                    const std::vector<std::size_t>& order) {
        for (std::size_t k = 1; k < order.size(); ++k) {
            if (at(keys, order[k], 0) == at(keys, order[k - 1], 0)
                && at(keys, order[k], 1) == at(keys, order[k - 1], 1)) {
                throw std::invalid_argument("facet keys for tag " + detail::quote_name(name)
                                            + " must be unique; got duplicates.");
            }
        }
    }

    std::int64_t num_cells_;       ///< The owning grid's cell count.
    std::int64_t facets_per_cell_; ///< Local facets per cell.
    std::vector<Entry> tags_;      ///< The registered tags, in insertion order.
};

}  // namespace pantr::grid
