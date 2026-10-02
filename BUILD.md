# ZERO Phase 1 Build Guide

## Linux / WSL / MinGW-style CMake toolchains

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## UCI smoke test

```text
uci
isready
position startpos
go depth 5
quit
```

Expected final move from the Phase 1 baseline run: `e2e4`.

## Design rule

The build contains only active Phase 1 modules. Disabled historical TT/history/killer/zobrist/qsearch implementations are not kept in the active source tree. The original MVP remains a separate frozen reference archive.
