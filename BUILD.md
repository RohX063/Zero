# ZERO 2.6 R3.5 QS-L0 — Build Guide

This guide explains how to build ZERO on Linux, macOS and WSL using CMake.

## Requirements

* CMake 3.16 or newer
* A C++20-compatible compiler (GCC or Clang)
* Python 3 is optional and only required for certain tuning scripts.

ZERO has no external runtime library dependencies.

## Release build

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON

cmake --build build --parallel 2
```

The executable will be generated at:

```text
build/zero
```

## Run the test suite

```bash
ctest --test-dir build --output-on-failure
```

The current CMake test suite contains seven tests:

* `perft`
* `search_smoke`
* `quiet_checks`
* `repetition_search`
* `position_integrity`
* `search_time`
* `mistake_punisher`

## UCI smoke test

Launch the engine:

```bash
./build/zero
```

Enter the following commands individually:

```text
uci
isready
position startpos
go depth 5
```

Wait for the engine to return `bestmove`, then enter:

```text
quit
```

The engine should complete the UCI handshake, acknowledge readiness and return a legal move.

## Deterministic benchmark

From the repository root, run:

```text
bench 7 tuning/bench_positions.epd
```

The benchmark evaluates the fixed position suite and prints a signature.

Use the same executable settings, benchmark file and depth when comparing builds.

## Debug build

To build with debugging information:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON

cmake --build build-debug --parallel 2

ctest --test-dir build-debug --output-on-failure
```

## Notes

* Release is recommended for engine matches and performance measurements.
* Debug builds are useful for diagnosing correctness issues.
* Keep benchmark configurations identical when comparing versions.
* The tuning scripts are optional and are not required to compile ZERO.
