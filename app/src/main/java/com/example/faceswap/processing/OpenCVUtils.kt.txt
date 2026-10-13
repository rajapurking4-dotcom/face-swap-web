package com.example.faceswap.processing

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat

object OpenCVUtils {
    fun initOpenCV(context: Context) {
        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OpenCV failed to initialize. Ensure native libs are added in app/src/main/jniLibs/")
        }
    }

    fun bitmapToMat(bmp: Bitmap): Mat {
        val mat = Mat(bmp.height, bmp.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bmp, mat)
        return mat
    }

    fun matToBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }
}
