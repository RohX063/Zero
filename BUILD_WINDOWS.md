# ZERO 2.6 R3.5 QS-L0 — Windows Build Guide

This guide explains how to build ZERO on Windows using CMake and Microsoft Visual Studio.

## Requirements

* Windows 10 or newer
* CMake 3.16 or newer
* Visual Studio 2022 with the Desktop Development with C++ workload
* MSVC C++17-compatible compiler

## 1. Open the project

Open PowerShell or the Visual Studio Developer PowerShell.

Navigate to the repository root (the directory containing `CMakeLists.txt`).

## 2. Configure the project

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
```

CMake will generate a Visual Studio build configuration when using the Visual Studio generator.

## 3. Build ZERO

```powershell
cmake --build build --config Release --parallel 4
```

The Release executable will normally be generated at:

```text
build\Release\zero.exe
```

## 4. Run the test suite

```powershell
ctest --test-dir build -C Release --output-on-failure
```

The current CMake test suite contains seven tests:

* `perft`
* `search_smoke`
* `quiet_checks`
* `repetition_search`
* `position_integrity`
* `search_time`
* `mistake_punisher`

## 5. Run ZERO

```powershell
.\build\Release\zero.exe
```

Test the UCI interface by entering these commands individually:

```text
uci
isready
position startpos
go depth 5
```

Wait for the `bestmove` response before entering:

```text
quit
```

## 6. Debug configuration

To build a Debug version:

```powershell
cmake --build build --config Debug --parallel 4
```

Run its tests:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

## Important: Visual Studio build configurations

Visual Studio uses a multi-configuration CMake generator.

Unlike single-configuration generators, `Release` and `Debug` are selected during the build and test commands using `--config`.

Do not rely on `CMAKE_BUILD_TYPE` when using the Visual Studio generator.

## Notes

* Use Release for engine matches and performance testing.
* Use Debug for development and debugging.
* Keep compiler, build configuration and engine options consistent when comparing versions.
* Python is not required to compile or run the chess engine.
