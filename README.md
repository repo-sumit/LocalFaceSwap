# LocalFaceSwap

LocalFaceSwap now includes two paths:

- A Python live face-swap app that uses the same model-based idea as Deep-Live-Cam: it reads the newest JPG/PNG from `uploads/` and swaps your live webcam face so your expression follows the camera.
- A native C++ app that stays lighter and easier to run locally, but is still a heuristic lightweight effect rather than a full neural deepfake.

If your goal is "look like the uploaded person while keeping my live expression", use the Python app first.

## What It Does

- Creates an `uploads/` folder for your source image
- Auto-loads the newest JPG/PNG found in `uploads/`
- Lets you lock onto your live face with the on-screen guide box
- Tracks your face in real time with optical flow
- Replaces your live face with the uploaded face in realtime
- Can also switch to a portrait/avatar overlay mode
- Supports quick scale and offset tuning while the app is running

## Python Live Face Swap

This is the real model-based face-swap path inspired by the Deep-Live-Cam reference project. It uses `insightface` for face analysis and the `inswapper_128.onnx` model for the actual swap, which means your live face pose and expression drive the output.

### Python Setup

From PowerShell in the project folder:

```powershell
.\setup-python.ps1
```

Create the environment, install dependencies, and launch immediately:

```powershell
.\setup-python.ps1 -Run
```

The first run will also download the swap model to `models/`, and `insightface` will download its face-analysis model pack.

### Python Run

```powershell
.\.venv\Scripts\python.exe .\python\live_face_swap.py
```

Useful options:

- `--execution-provider=auto|cpu|cuda|directml`
- `--camera=0`
- `--width=640 --height=480`
- `--swap-all-faces`
- `--no-mirror`

Runtime keys:

- `q` quit
- `r` rescan `uploads/` immediately
- `m` toggle mirror

## Native C++ Lightweight

The native app is still here for a faster and lighter local experience. It is useful when you want smoother low-overhead tracking, but it does not reproduce Deep-Live-Cam quality or expression transfer.

This is much lighter than a full neural body-swap pipeline, so it is easier to run locally and stays smoother on weaker hardware.

## Best Results

Use an uploaded image that:

- shows one person clearly
- has the head and body visible
- is centered in the frame
- is a PNG with transparency if possible

JPG images also work. For non-transparent images, the app tries to estimate the person cutout automatically.

## Project Layout

- `native/src/main.cpp` native app
- `uploads/` place your source image here
- `build-native.ps1` one-command build helper
- `CMakeLists.txt` native build definition
- `vcpkg.json` dependency manifest
- `vcpkg-triplets/x64-windows-release.cmake` release-only dependency triplet

## Native Build

From PowerShell in the project folder:

```powershell
.\build-native.ps1
```

Build and launch immediately:

```powershell
.\build-native.ps1 -Run
```

The first build can take time because OpenCV may need to be built once through `vcpkg`.

## Native Run

```powershell
.\build\native\local_face_filter.exe
```

You can also choose the uploads folder explicitly:

```powershell
.\build\native\local_face_filter.exe --uploads-dir=uploads
```

## How To Use

1. Put a JPG or PNG image in `uploads/`
2. Start the app
3. Center your face inside the guide box
4. Press `Space` to lock onto your face
5. The newest uploaded image will auto-load
6. Press `3` for face swap or `4` for avatar mode
7. Press `r` if you want to force an immediate rescan

## Runtime Controls

- `q` quit
- `space` lock face using the guide box
- `c` clear the current lock
- `r` force a rescan of the uploaded image from `uploads/`
- `1` hat overlay
- `2` glasses overlay
- `3` uploaded face swap
- `4` uploaded avatar swap
- `[` shrink avatar
- `]` enlarge avatar
- `i` move avatar up
- `k` move avatar down
- `j` move avatar left
- `l` move avatar right
- `d` toggle debug box

## Command Line Options

- `--camera=<index>` camera index, default `0`
- `--width=<pixels>` capture width, default `640`
- `--height=<pixels>` capture height, default `480`
- `--fps=<value>` requested camera FPS, default `30`
- `--detect-interval=<frames>` tracked frames before point refresh, default `6`
- `--overlay=face|avatar|hat|glasses` starting overlay mode, default `face`
- `--backend=dshow|any` camera backend, default `dshow`
- `--mirror=true|false` mirror preview, default `true`
- `--debug=true|false` show debug box, default `false`
- `--uploads-dir=<path>` folder containing your uploaded image, default `uploads`

## Performance Notes

This native version is designed to stay responsive by:

1. using a latest-frame-only capture loop
2. tracking instead of re-detecting constantly
3. avoiding heavy neural inference
4. rendering a lightweight 2D face or avatar overlay

For smoother performance:

- keep the camera at `640x480`
- use `--fps=24` on slower systems
- use a clean source image with one centered person
- prefer PNG images with transparency

Example lower-load run:

```powershell
.\build\native\local_face_filter.exe --fps=24 --detect-interval=8 --overlay=face
```

## Manual Build

If you want the raw CMake flow:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$toolchain = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
$ninja = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$triplets = (Resolve-Path ".\vcpkg-triplets").Path

& $cmake -S . -B build\native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DVCPKG_OVERLAY_TRIPLETS="$triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-release
& $cmake --build build\native --config Release
```

## Important Limitation

This is still a lightweight heuristic swap, not a landmark-accurate neural deepfake pipeline. The new default mode replaces your face with the uploaded face crop, and the avatar mode overlays the uploaded portrait on your tracked head position. It is fast and local, but it will not perfectly match every pose, expression, or head turn.
