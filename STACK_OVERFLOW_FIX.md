# R3.3.1 Windows Stack-Overflow Fix

## Root cause
`Worker` previously embedded the entire `HistoryTables` object inline. The continuation-history table makes `HistoryTables` about 1.53 MiB, and the complete `Worker` object was about 1.67 MiB.

A local/automatic `Worker` therefore exceeded the default Windows thread stack budget and `search_smoke_test` terminated with a segmentation fault before the search smoke test could complete.

Reproduction on a 1 MiB stack:

```text
old R3.3 search_smoke_test -> Segmentation fault (RC 139)
```

## Fix
`HistoryTables` is now heap-owned by `Worker` using `std::unique_ptr`. The `Worker` object drops from about 1.67 MiB to about 141 KiB.

This changes storage lifetime only; it does not change the search algorithm or SPSA parameters.

## Validation

```text
Release build
perft              PASS
search_smoke       PASS
position_integrity PASS

ASan + UBSan
perft              PASS
search_smoke       PASS
position_integrity PASS

100% tests passed, 0 failures
```
