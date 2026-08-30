# Type-level tests

Modules here are checked by **mypy**, not by running them, and they are the only
place a type error is an *expected result*. `tests/test_grid_typing.py` runs mypy over
`cases/` and compares the errors it reports against the lines marked

```python
some_expression  # expect-error: arg-type
```

The comparison is a set equality, so an expected error that stops being reported and an
unexpected one that starts both fail. The marker names the mypy **error code**, which is
what stops a case from passing on the wrong error.

`mypy.ini` excludes this directory from the repo-wide run. It has to: these modules do
not type-check by construction, and `make mypy` would go red on them. The exclusion only
suppresses directory *discovery* -- mypy still checks a file named explicitly on the
command line, which is how the harness reaches them, and it means both runs share one
cache and one set of flags.

## Why this exists

`Grid` is a `typing.Protocol`, and a protocol is enforced by a type checker rather than
at run time: no `isinstance` answers it, and nothing in the ordinary suite can observe a
class failing to satisfy it. Ticket #386 asked for a type-level test for that reason,
and there was no harness in the repo; this is the smallest one that works.
