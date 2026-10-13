package com.example.faceswap.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
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
            val inputVideoFile = copyUriToFile(videoUri, File(workDir, "input.mp4")) ?: return@withContext null

            onProgress("Extracting frames...", 0.05f)
            val fps = 24
            val extractCmd = "-i \"${inputVideoFile.absolutePath}\" -vf fps=$fps \"${framesDir.absolutePath}/frame_%05d.png\""
            val extractSession = FFmpegKit.execute(extractCmd)
            if (!ReturnCode.isSuccess(extractSession.returnCode)) {
                onProgress("Frame extraction failed", 0f)
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
                        faceSwapProcessor.swapFacesWithDelaunay(
                            sourceBitmap,
                            sourceFaceId,
                            bmp,
                            matchingTargetIds
                        ) { _, _ -> }
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
            val encodeCmd = "-framerate $fps -i \"${outFramesDir.absolutePath}/frame_%05d.png\" -i \"${inputVideoFile.absolutePath}\" -map 0:v -map 1:a? -c:v libx264 -pix_fmt yuv420p -c:a copy -shortest \"${outputFile.absolutePath}\""
            val encodeSession = FFmpegKit.execute(encodeCmd)
            if (!ReturnCode.isSuccess(encodeSession.returnCode)) {
                onProgress("Video encoding failed", 0f)
                return@withContext null
            }

            onProgress("Done", 1f)
            outputFile
        } catch (e: Exception) {
            onProgress("Error: ${e.message}", 0f)
            null
        }
    }

    private fun copyUriToFile(uri: Uri, destFile: File): File? {
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            destFile
        } catch (e: Exception) {
            null
        }
    }
}
