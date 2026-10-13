package com.example.faceswap.processing

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Handles video face-swap by:
 * 1. Extracting frames from the input video with FFmpegKit.
 * 2. Running the existing FaceSwapProcessor on each frame.
 * 3. Re-encoding the processed frames back into a video with FFmpegKit.
 * 4. Saving the finished video to the Gallery (Movies/FaceSwap).
 *
 * This is a straightforward frame-by-frame pipeline; it trades some speed
 * for simplicity and reliability.
 */
class VideoProcessor(private val context: Context) {

    data class Progress(val stage: String, val fraction: Float)

    suspend fun swapFacesInVideo(
        sourceBitmap: Bitmap,
        sourceFaceId: Int,
        videoUri: Uri,
        targetFaceIds: List<Int>,
        faceSwapProcessor: FaceSwapProcessor,
        onProgress: (String, Float) -> Unit
    ): File? = withContext(Dispatchers.IO) {
        val workDir = File(context.cacheDir, "video_swap_${System.currentTimeMillis()}")
        val framesDir = File(workDir, "frames")
        val outFramesDir = File(workDir, "out_frames")
        framesDir.mkdirs()
        outFramesDir.mkdirs()

        try {
            onProgress("Copying video...", 0.02f)
            val inputVideoFile = copyUriToFile(videoUri, File(workDir, "input.mp4"))
            if (inputVideoFile == null) {
                onProgress("Could not read the selected video", 0f)
                return@withContext null
            }

            onProgress("Extracting frames...", 0.05f)
            // 15 fps and max 720px wide keeps this feasible on a phone.
            val fps = 15
            val extractCmd = "-y -i \"${inputVideoFile.absolutePath}\" -vf \"fps=$fps,scale=720:-2\" \"${framesDir.absolutePath}/frame_%05d.png\""
            val extractSession = FFmpegKit.execute(extractCmd)
            if (!ReturnCode.isSuccess(extractSession.returnCode)) {
                onProgress("Frame extraction failed: ${logTail(extractSession.allLogsAsString)}", 0f)
                return@withContext null
            }

            val frameFiles = framesDir.listFiles { f -> f.extension == "png" }?.sortedBy { it.name } ?: emptyList()
            if (frameFiles.isEmpty()) {
                onProgress("No frames extracted", 0f)
                return@withContext null
            }

            val total = frameFiles.size
            frameFiles.forEachIndexed { index, frameFile ->
                val bmp = BitmapFactory.decodeFile(frameFile.absolutePath)
                if (bmp != null) {
                    val targetFaces = faceSwapProcessor.detectFaces(bmp)
                    val matchingTargetIds = targetFaces.map { it.id }
                    val swapped = if (matchingTargetIds.isNotEmpty()) {
                        try {
                            faceSwapProcessor.swapFacesWithDelaunay(
                                sourceBitmap,
                                sourceFaceId,
                                bmp,
                                matchingTargetIds
                            ) { _, _ -> }
                        } catch (e: Exception) {
                            null
                        }
                    } else {
                        null
                    }
                    val finalBmp = swapped ?: bmp
                    val outFile = File(outFramesDir, frameFile.name)
                    outFile.outputStream().use { out ->
                        finalBmp.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                }
                val frac = 0.1f + 0.8f * (index + 1).toFloat() / total.toFloat()
                onProgress("Processing frame ${index + 1}/$total", frac)
            }

            onProgress("Re-encoding video...", 0.92f)
            val outputFile = File(workDir, "output.mp4")
            // mpeg4 is built into FFmpeg (libx264 is NOT in this FFmpegKit package).
            val encodeCmd = "-y -framerate $fps -i \"${outFramesDir.absolutePath}/frame_%05d.png\" -i \"${inputVideoFile.absolutePath}\" -map 0:v -map 1:a? -c:v mpeg4 -q:v 3 -pix_fmt yuv420p -c:a aac -shortest \"${outputFile.absolutePath}\""
            val encodeSession = FFmpegKit.execute(encodeCmd)
            if (!ReturnCode.isSuccess(encodeSession.returnCode)) {
                onProgress("Video encoding failed: ${logTail(encodeSession.allLogsAsString)}", 0f)
                return@withContext null
            }

            val saved = saveToGallery(outputFile)
            onProgress(if (saved) "Done - saved to Gallery (Movies/FaceSwap)" else "Done (could not copy to Gallery)", 1f)
            outputFile
        } catch (e: Exception) {
            onProgress("Error: ${e.message}", 0f)
            null
        }
    }

    private fun logTail(logs: String?): String {
        if (logs.isNullOrBlank()) return ""
        return logs.trim().takeLast(200)
    }

    private fun saveToGallery(file: File): Boolean {
        return try {
            val name = "FaceSwap_${System.currentTimeMillis()}.mp4"
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, name)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/FaceSwap")
                }
                val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                    ?: return false
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    file.inputStream().use { it.copyTo(out) }
                }
                true
            } else {
                val dir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: return false
                dir.mkdirs()
                file.copyTo(File(dir, name), overwrite = true)
                true
            }
        } catch (e: Exception) {
            false
        }
    }

    private fun copyUriToFile(uri: Uri, destFile: File): File? {
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            } ?: return null
            destFile
        } catch (e: Exception) {
            null
        }
    }
}
