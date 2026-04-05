# LocalFaceFilter Native

LocalFaceFilter Native is a lightweight C++ webcam filter app designed for smooth realtime effects on lower-end machines. It uses a native OpenCV pipeline with manual face lock, optical-flow tracking, and simple overlay rendering instead of a heavy neural face-swap path.

## Why This Version Exists

The original Python prototype was useful for experimentation, but it was the wrong architecture for Snapchat-style smoothness on light hardware. This native project is optimized for:

- lower latency
- lower CPU overhead
- latest-frame rendering instead of laggy backlog
- simple, stable AR-style effects

## Features

- Native C++20 desktop app
- Realtime webcam preview
- Single-face optimized pipeline
- Manual face lock for instant startup on a smaller OpenCV build
- Optical-flow tracking after lock
- Built-in `hat` and `glasses` overlays
- Selfie-style mirrored preview by default
- Low-latency camera capture using DirectShow on Windows

## Requirements

- Windows
- Visual Studio Build Tools with C++ tools installed
- The Visual Studio bundled CMake, Ninja, and `vcpkg` toolchain
- A webcam

This repo is now native-first. Python is no longer required for normal use.

## Quick Start

Build the app:

```powershell
.\build-native.ps1
```

Build and run immediately:

```powershell
.\build-native.ps1 -Run
```

The first build may take a long time because `vcpkg` may need to download and compile OpenCV.
This repo is tuned to reduce that cost by building a smaller OpenCV feature set and using a release-only `vcpkg` triplet.

## Run

```powershell
.\build\native\local_face_filter.exe
```

Example:

```powershell
.\build\native\local_face_filter.exe --overlay=glasses --detect-interval=5
```

## Runtime Controls

- `q` quit
- `space` lock face using the on-screen guide box
- `c` clear the current face lock
- `1` switch to hat overlay
- `2` switch to glasses overlay
- `d` toggle debug face rectangle

## Command Line Options

- `--camera=<index>` camera index, default `0`
- `--width=<pixels>` capture width, default `640`
- `--height=<pixels>` capture height, default `480`
- `--fps=<value>` requested camera FPS, default `30`
- `--detect-interval=<frames>` tracked frames before reseeding points, default `4`
- `--overlay=hat|glasses` choose the active overlay
- `--backend=dshow|any` camera backend, default `dshow`
- `--mirror=true|false` mirror the preview, default `true`
- `--debug=true|false` show debug rectangle, default `false`

## Performance Design

This project is built around a few rules that matter on weaker devices:

1. Capture and render the latest frame, not every frame.
2. Lock once, then track cheaply in between refreshes.
3. Render cheap overlays instead of running a heavy identity-swap model.
4. Keep the effect stable with lightweight smoothing.

Core pipeline:

1. Camera thread continuously grabs frames into a latest-frame buffer.
2. Main loop reads only the freshest frame.
3. The user locks the face once with the guide box.
4. `calcOpticalFlowPyrLK` tracks face motion after lock.
5. The tracked box is smoothed and the overlay is rendered.

## Low-End Device Tips

If you want the smoothest experience:

- keep capture at `640x480`
- use one face only
- increase `--detect-interval` to reduce reseeding cost
- keep debug mode off
- close other camera-heavy apps

Example low-load launch:

```powershell
.\build\native\local_face_filter.exe --width=640 --height=480 --fps=24 --detect-interval=6 --overlay=hat
```

## Manual Build

If you want the raw CMake flow instead of the helper script:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$toolchain = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
$ninja = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$triplets = (Resolve-Path ".\vcpkg-triplets").Path

& $cmake -S . -B build\native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DVCPKG_OVERLAY_TRIPLETS="$triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-release
& $cmake --build build\native --config Release
```

## Project Layout

- `CMakeLists.txt` native build definition
- `vcpkg.json` native dependency manifest
- `vcpkg-triplets/x64-windows-release.cmake` release-only vcpkg triplet to cut dependency build time
- `build-native.ps1` one-command configure/build helper
- `native/src/main.cpp` main application
## Notes

- This app targets smooth AR-style overlays, not full neural face swap.
- The first native build is the heaviest step. After dependencies are built, rebuilds are much faster.
- OpenCV is intentionally trimmed to the smallest useful Windows feature set for this app.
- Manual face lock is a deliberate tradeoff to avoid heavier detection dependencies and keep startup/build time lower.
- If you later want even better face quality, the next upgrade should be a stronger landmark detector, not a return to CPU-only Python face swap.
