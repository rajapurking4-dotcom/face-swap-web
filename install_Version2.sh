#!/usr/bin/env bash
set -e
# Quick helper: build debug APK and install to connected device (adb required)
./gradlew :app:assembleDebug

APK_PATH=$(find ./app/build/outputs/apk -name "*debug.apk" | head -n 1)
if [ -z "$APK_PATH" ]; then
  echo "APK not found. Build failed."
  exit 1
fi

echo "Installing $APK_PATH to device..."
adb install -r "$APK_PATH"

echo "Installed. Launch the app on the device."