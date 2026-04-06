# LocalFaceSwap

LocalFaceSwap is a Windows-first live face-swap project with two modes:

- A Python app for real model-based face swap from a folder image to your live webcam feed
- A native C++ app for a lighter local effect when you want lower overhead and easier native execution

If your goal is "use my webcam in Google Meet or Zoom, but make my face look like the person in the uploaded image", use the Python app first.

## What This Project Can Do

- Watch the `uploads/` folder and automatically use the newest JPG, PNG, JPEG, BMP, or WEBP image
- Detect the main face in the uploaded image and use it as the source identity
- Read your live webcam feed and swap your face so the live pose and expression drive the output
- Show the result in a local preview window
- Run in the background and publish the swapped feed to a virtual camera for Google Meet, Zoom, Teams, and similar apps
- Output through `OBS Virtual Camera` or `Unity Capture` on Windows
- Mirror or un-mirror the local preview
- Swap only the main face or all detected faces in a frame
- Blend the result with the original frame using opacity control
- Use different ONNX execution providers such as CPU, CUDA, or DirectML when available
- Keep a separate native C++ fallback app for a lighter non-neural effect

## Recommended Path

For most people, this is the best workflow:

1. Install Python 3.11.
2. Install OBS Studio if you want the swapped video inside Google Meet or another webcam app.
3. Put one face image into `uploads/`.
4. Run the Python setup once.
5. Start the Python app in virtual camera mode.
6. In Google Meet, choose `OBS Virtual Camera` instead of your real webcam.

## Quick Start

### Requirements

- Windows
- Python 3.11
- A webcam
- PowerShell
- OBS Studio if you want a virtual camera for Google Meet, Zoom, Teams, or similar apps

Optional but useful:

- A GPU with a supported ONNX Runtime provider for lower latency
- A clean portrait image in `uploads/`

### Install OBS Studio For Google Meet

If you only want the local preview window, you can skip OBS Studio.

If you want the swapped output to appear as a camera inside Google Meet:

1. Download and install OBS Studio from `https://obsproject.com/download`.
2. Finish the installation normally.
3. Fully close Chrome, Edge, and Meet after installation.
4. Reopen them only after the LocalFaceSwap virtual camera is running.

LocalFaceSwap reads your real webcam and publishes the processed result to a virtual camera. Because of that, Google Meet should use `OBS Virtual Camera`, not your physical camera.

### First-Time Setup

Open PowerShell in the project folder and run:

```powershell
.\setup-python.ps1
```

This does the first-time Python setup:

- creates `.venv/`
- upgrades `pip`
- installs all Python packages
- keeps the environment ready for later runs

On the first actual launch, the app will also download:

- the `inswapper_128.onnx` face-swap model into `models/`
- the InsightFace `buffalo_l` analysis models into `models/buffalo_l/`

### First-Time Setup And Run For Google Meet

If you want to set everything up and immediately start the virtual camera:

```powershell
.\setup-python.ps1 -Run -VirtualCamera -NoPreview
```

What to expect:

- the app loads the newest source image from `uploads/`
- the app opens your real webcam
- the app starts the swap pipeline
- the app publishes the result to `OBS Virtual Camera`
- the terminal should print a line like:

```text
[python-swap] Virtual camera ready: OBS Virtual Camera
```

### Regular Run After Setup

After the first setup, these are the main commands you will use.

Run with a local preview window:

```powershell
.\.venv\Scripts\python.exe .\python\live_face_swap.py
```

Run with a local preview and virtual camera:

```powershell
.\setup-python.ps1 -Run -VirtualCamera
```

Run in the background for Google Meet or Zoom without a local preview window:

```powershell
.\setup-python.ps1 -Run -VirtualCamera -NoPreview
```

Equivalent direct command:

```powershell
.\.venv\Scripts\python.exe .\python\live_face_swap.py --virtual-camera --virtual-camera-backend=obs --no-preview
```

### Everyday Google Meet Steps

Use this order each time:

1. Put your source image into `uploads/`.
2. Start LocalFaceSwap:

```powershell
.\setup-python.ps1 -Run -VirtualCamera -NoPreview
```

3. Wait for:

```text
[python-swap] Virtual camera ready: OBS Virtual Camera
```

4. Open Google Meet after that.
5. In Meet, choose `OBS Virtual Camera` as the camera.
6. Do not choose `HP FHD Camera` or your physical webcam.

If Meet only shows the physical webcam:

1. Fully close Chrome or Edge.
2. Start LocalFaceSwap first.
3. Reopen the browser.
4. Recheck the camera list in Meet.

If needed, join the meeting first, then open:

- `More options -> Settings -> Video`

Meet sometimes refreshes the device list better there.

## Python App Capabilities

- Auto-watches the `uploads/` folder
- Uses the newest supported image automatically
- Detects the best face in the source image
- Detects faces in the live webcam feed
- Swaps the source identity onto the live face
- Keeps live pose and expression from the webcam frame
- Supports local preview mode
- Supports background virtual camera mode
- Supports OBS or Unity Capture backends for virtual camera output
- Supports `auto`, `cpu`, `cuda`, `directml`, `openvino`, `coreml`, and `tensorrt` provider selection when available
- Supports single-face or all-face swap
- Supports adjustable opacity from `0.0` to `1.0`
- Supports mirrored or non-mirrored local preview
- Supports camera index selection
- Supports custom capture resolution, FPS, and detector input size

## Python Commands And Options

### Main Commands

Setup only:

```powershell
.\setup-python.ps1
```

Setup and run with preview:

```powershell
.\setup-python.ps1 -Run
```

Setup and run with OBS virtual camera:

```powershell
.\setup-python.ps1 -Run -VirtualCamera
```

Setup and run with OBS virtual camera in background:

```powershell
.\setup-python.ps1 -Run -VirtualCamera -NoPreview
```

### Useful Direct CLI Options

- `--uploads-dir=uploads`
- `--camera=0`
- `--width=640`
- `--height=480`
- `--fps=30`
- `--det-size=640`
- `--execution-provider=auto|cpu|cuda|directml|openvino|coreml|tensorrt`
- `--backend=dshow|any`
- `--virtual-camera`
- `--virtual-camera-backend=auto|obs|unitycapture`
- `--virtual-camera-device=<exact device name>`
- `--preview`
- `--no-preview`
- `--mirror`
- `--no-mirror`
- `--swap-all-faces`
- `--opacity=1.0`

### Runtime Keys

These keys are available when the local preview window is enabled:

- `q` quit
- `r` rescan `uploads/` immediately
- `m` toggle preview mirror

## Best Results

Use an uploaded image that:

- shows one face clearly
- is reasonably front-facing
- has good lighting
- is not heavily blurred
- does not contain multiple competing faces
- is tightly framed around the face and upper head area when possible

Good source images usually improve results more than any code setting.

## Lower-Latency Tips

If the app feels slow, start with these:

```powershell
.\.venv\Scripts\python.exe .\python\live_face_swap.py --width=320 --height=240 --fps=24 --det-size=320 --virtual-camera --no-preview
```

For smoother runtime:

- keep capture size at `640x480` or lower
- reduce `--det-size` to `320`
- reduce `--fps` to `24`
- keep `--swap-all-faces` off
- use `--no-preview` when you only need Meet or Zoom output
- use `directml` or another hardware provider if available

Example with DirectML:

```powershell
.\.venv\Scripts\python.exe .\python\live_face_swap.py --execution-provider=directml --virtual-camera --no-preview --width=640 --height=480 --fps=24 --det-size=320
```

## How The Python Pipeline Works

The Python app works like this:

1. It watches the `uploads/` folder and finds the newest supported image.
2. It reads that image and detects the best face inside it.
3. It opens the webcam with a latest-frame-only capture loop to reduce backlog.
4. For each fresh frame, it detects the live face or faces.
5. It runs the `inswapper_128.onnx` model to place the uploaded identity onto the live face.
6. It optionally blends the result with the original frame using the configured opacity.
7. It either shows the result in a local OpenCV preview window, sends it to a virtual camera, or both.
8. Google Meet or Zoom reads the virtual camera instead of the physical webcam.

Important behavior:

- The real webcam is owned by LocalFaceSwap.
- Meet should read the virtual camera output, not the real webcam.
- The source image can be changed by dropping a newer image into `uploads/`.

## Libraries Used In The Python App

### `opencv-python`

Used for:

- webcam capture
- image loading
- preview rendering
- text overlay
- frame transforms such as mirroring
- frame blending

Why it is used:

- it gives a simple and fast camera plus image-processing layer
- it is the easiest way to manage preview windows on Windows for this project

### `numpy`

Used for:

- frame storage
- array operations
- efficient image data movement between OpenCV, InsightFace, and pyvirtualcam

Why it is used:

- almost every image-processing library in this stack expects NumPy arrays

### `onnxruntime`

Used for:

- running the ONNX models used by the face analysis and swap pipeline

Why it is used:

- it supports CPU and multiple hardware backends
- it is a lightweight inference runtime compared to a full training framework

### `insightface`

Used for:

- face detection
- face recognition embeddings
- landmark extraction
- loading the `inswapper_128.onnx` model

Why it is used:

- it provides the full face analysis plus swap flow needed for a Deep-Live-Cam-style result

### `pyvirtualcam`

Used for:

- sending processed frames to a Windows virtual camera device such as `OBS Virtual Camera`

Why it is used:

- it is the bridge that lets Google Meet, Zoom, and Teams see the processed video as a normal camera

## Project Layout

- `python/live_face_swap.py` Python live face-swap app
- `python/requirements.txt` Python dependencies
- `setup-python.ps1` PowerShell helper for Python setup and run
- `uploads/` drop source images here
- `models/` downloaded swap and analysis models
- `native/src/main.cpp` native C++ lightweight app
- `build-native.ps1` one-command native build helper
- `CMakeLists.txt` native build definition
- `vcpkg.json` native dependency manifest
- `vcpkg-triplets/x64-windows-release.cmake` native release triplet

## Native C++ Lightweight App

The native app is still included for a lighter local effect path.

Use the C++ app when:

- you want a simpler native executable
- you care more about lightweight local performance than neural realism
- you want a non-Python fallback

Do not expect the same identity transfer quality as the Python app. The C++ version is not a full Deep-Live-Cam neural port.

### Native Capabilities

- opens the webcam
- tracks the face locally
- watches the `uploads/` folder
- auto-loads the newest uploaded image
- supports face and avatar overlay modes
- supports lightweight runtime controls for scale and offset
- stays smoother on weaker systems than the model-based Python path

### Native Build

First build:

```powershell
.\build-native.ps1
```

Build and run:

```powershell
.\build-native.ps1 -Run
```

Direct run:

```powershell
.\build\native\local_face_filter.exe
```

Lower-load native example:

```powershell
.\build\native\local_face_filter.exe --fps=24 --detect-interval=8 --overlay=face
```

### Native Runtime Controls

- `q` quit
- `space` lock face using the guide box
- `c` clear the current lock
- `r` rescan `uploads/`
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

### Native Command Line Options

- `--camera=<index>`
- `--width=<pixels>`
- `--height=<pixels>`
- `--fps=<value>`
- `--detect-interval=<frames>`
- `--overlay=face|avatar|hat|glasses`
- `--backend=dshow|any`
- `--mirror=true|false`
- `--debug=true|false`
- `--uploads-dir=<path>`

### Native Performance Notes

The C++ path stays lighter because it:

- uses a latest-frame-only capture loop
- relies on local tracking more than heavy neural inference
- renders a lightweight 2D effect instead of running the full model-based swap pipeline

## Technical Notes

### Model Download Behavior

The Python app downloads models automatically if they are missing.

Files involved:

- `models/inswapper_128.onnx`
- `models/buffalo_l/...`

This means the first launch is slower than later launches.

### Folder Watch Behavior

The Python app polls `uploads/` periodically and always prefers the newest matching file. If you add a new file, it becomes the new source image automatically.

### Preview Versus Virtual Camera

- Preview mode shows a local OpenCV window
- Virtual camera mode publishes frames to OBS Virtual Camera or Unity Capture
- You can use both at the same time
- `--no-preview` is useful when you want the app to run quietly in the background

### CPU Versus GPU

If the app logs `CPUExecutionProvider`, it is running fully on CPU. That works, but latency will be higher.

If your system supports it, try a hardware execution provider such as `directml` on Windows Intel or AMD GPUs.

## Troubleshooting

### Google Meet Only Shows The Real Camera

Try this order:

1. Close Meet and fully close the browser.
2. Start LocalFaceSwap first with:

```powershell
.\setup-python.ps1 -Run -VirtualCamera -NoPreview
```

3. Wait for:

```text
[python-swap] Virtual camera ready: OBS Virtual Camera
```

4. Reopen the browser and Meet.
5. Check the camera list again.

If still needed:

1. Join the meeting first.
2. Open `More options -> Settings -> Video`.
3. Select `OBS Virtual Camera` there.

If the device still does not appear, reboot Windows once and try again.

### Meet Says "Camera Unavailable"

That usually means Meet is trying to use the physical webcam while LocalFaceSwap already owns it.

Fix:

- keep LocalFaceSwap running
- switch Meet from the physical webcam to `OBS Virtual Camera`

### The App Is Slow

Try:

- `--width=320 --height=240`
- `--fps=24`
- `--det-size=320`
- `--no-preview`
- `--execution-provider=directml` if supported

### The Wrong Face Is Used

Use an uploaded image with:

- one clear face
- good lighting
- minimal background clutter

The app picks the best face in the uploaded image, so crowded photos are a bad source.

### No Face Found In The Uploaded Image

Try a different source photo with:

- a larger face
- brighter lighting
- less blur
- less extreme head angle

## Manual Native Build

If you want the raw CMake flow for the native app:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$toolchain = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
$ninja = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$triplets = (Resolve-Path ".\vcpkg-triplets").Path

& $cmake -S . -B build\native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DVCPKG_OVERLAY_TRIPLETS="$triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-release
& $cmake --build build\native --config Release
```

## Limitations

- The Python app is the real face-swap path, but it still depends on source image quality and model limitations.
- Latency depends heavily on whether you are running on CPU or a hardware execution provider.
- Google Meet and Chrome sometimes cache device lists, so virtual camera visibility can require a full browser restart.
- The native C++ app is intentionally lighter and does not match the Python app's neural swap quality.
