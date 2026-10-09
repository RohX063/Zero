# ZERO R3.3 — Windows build

From the project root (the directory containing `CMakeLists.txt`):

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config RelWithDebInfo -j 4
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

Important: this project uses a Visual Studio/MSVC generator on Windows, so `RelWithDebInfo` is a configuration selected at build/test time rather than a CMake single-config build type.
