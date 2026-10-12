# Face-Swap Android (Option B) — Full Project Scaffold

This repository is an on-device face-swap Android app using:
- OpenCV (native & Java) for image processing
- Google ML Kit Face Detection for landmarks & contours
- Kotlin + Jetpack Compose UI (dark black + red accents)
- Optional mobile-ffmpeg for on-device video frame extraction/reassembly
- A simple desktop CLI (OpenCV + MediaPipe) for quick testing

What’s included
- Image face-swap with Delaunay triangulation per‑triangle affine warps + blending
- Video pipeline: extract frames (FFmpeg), process per-frame, reassemble (FFmpeg)
- Compose UI with per-face selection overlays and progress bar
- CLI for desktop quick swaps
- GitHub Actions workflow to build debug APK

Important setup steps
1. OpenCV Android SDK
   - Download OpenCV Android SDK from https://opencv.org/releases/
   - Copy native libs: `OpenCV-android-sdk/sdk/native/libs/<abi>/libopencv_java4.so` into `app/src/main/jniLibs/<abi>/`
     (e.g., `armeabi-v7a`, `arm64-v8a`)
   - Copy the Java wrapper jar: `OpenCV-android-sdk/sdk/java/bin/opencv-4xx.jar` into `app/libs/` (rename to `opencv-4xx.jar`)
   - In Android Studio, confirm `app/libs/opencv-4xx.jar` is added (app/build.gradle uses `implementation files('libs/opencv-4xx.jar')`).

2. FFmpeg (optional, for on-device video)
   - In `app/build.gradle` uncomment the mobile-ffmpeg dependency line to include FFmpeg.
   - Warning: this increases APK size dramatically.

3. Gradle wrapper
   - If you don't include Gradle wrapper files, generate them locally:
     ./gradlew wrapper
   - Then open the project in Android Studio.

4. Build & install (dev)
   - From Android Studio: Run app on a device (recommended).
   - From terminal:
     ./gradlew :app:assembleDebug
     ./install.sh

5. Desktop CLI (optional)
   - Requires Python 3 and packages:
     pip install opencv-python mediapipe numpy
   - Usage:
     python cli/swap.py --source source.jpg --target target.jpg --out out.jpg

Ethics & usage
- Use only with explicit permission of people whose faces you process.
- The app is local-first; data is processed on-device by default.
- No output watermark is added by default.

License
- Apache-2.0 (see LICENSE in repo root if you add one)
- Respect third-party libraries’ licenses (OpenCV, ML Kit, mobile-ffmpeg).