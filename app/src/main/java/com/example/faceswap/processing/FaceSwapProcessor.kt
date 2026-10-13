package com.example.faceswap.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Subdiv2D
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

/** A source face prepared once (RGB image + detected face), reusable for many target frames. */
class PreparedSource(val mat: Mat, val info: FaceInfo)

class FaceSwapProcessor(private val context: Context) {

    private val detector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
        FaceDetection.getClient(options)
    }

    suspend fun detectFaces(bitmap: Bitmap, samplingPoints: Int = 80): List<FaceInfo> {
        val image = InputImage.fromBitmap(bitmap, 0)
        val faces = detector.process(image).await()
        val out = mutableListOf<FaceInfo>()
        for ((idx, f) in faces.withIndex()) {
            val bbox = f.boundingBox
            val lands = mutableListOf<PointF>()
            val lmIds = listOf(
                com.google.mlkit.vision.face.FaceLandmark.LEFT_EYE,
                com.google.mlkit.vision.face.FaceLandmark.RIGHT_EYE,
                com.google.mlkit.vision.face.FaceLandmark.NOSE_BASE,
                com.google.mlkit.vision.face.FaceLandmark.MOUTH_LEFT,
                com.google.mlkit.vision.face.FaceLandmark.MOUTH_RIGHT
            )
            for (lid in lmIds) {
                val l = f.getLandmark(lid)?.position
                if (l != null) lands.add(PointF(l.x, l.y))
            }

            val contourPts = mutableListOf<PointF>()
            fun addContour(c: FaceContour?) {
                if (c != null) {
                    for (p in c.points) contourPts.add(PointF(p.x, p.y))
                }
            }
            // Fixed order + fixed point count per contour => source/target points correspond 1-to-1.
            // Index layout when complete (133 points):
            //  0-35 face oval | 36-40 L brow top | 41-45 L brow bottom | 46-61 L eye
            //  62-66 R brow top | 67-71 R brow bottom | 72-87 R eye | 88-89 nose bridge
            //  90-92 nose bottom | 93-103 upper lip top | 104-112 upper lip bottom
            //  113-121 lower lip top | 122-130 lower lip bottom | 131 L cheek | 132 R cheek
            addContour(f.getContour(FaceContour.FACE))
            addContour(f.getContour(FaceContour.LEFT_EYEBROW_TOP))
            addContour(f.getContour(FaceContour.LEFT_EYEBROW_BOTTOM))
            addContour(f.getContour(FaceContour.LEFT_EYE))
            addContour(f.getContour(FaceContour.RIGHT_EYEBROW_TOP))
            addContour(f.getContour(FaceContour.RIGHT_EYEBROW_BOTTOM))
            addContour(f.getContour(FaceContour.RIGHT_EYE))
            addContour(f.getContour(FaceContour.NOSE_BRIDGE))
            addContour(f.getContour(FaceContour.NOSE_BOTTOM))
            addContour(f.getContour(FaceContour.UPPER_LIP_TOP))
            addContour(f.getContour(FaceContour.UPPER_LIP_BOTTOM))
            addContour(f.getContour(FaceContour.LOWER_LIP_TOP))
            addContour(f.getContour(FaceContour.LOWER_LIP_BOTTOM))
            addContour(f.getContour(FaceContour.LEFT_CHEEK))
            addContour(f.getContour(FaceContour.RIGHT_CHEEK))
            if (contourPts.isEmpty()) {
                contourPts.add(PointF(bbox.left.toFloat(), bbox.top.toFloat()))
                contourPts.add(PointF(bbox.right.toFloat(), bbox.top.toFloat()))
                contourPts.add(PointF(bbox.right.toFloat(), bbox.bottom.toFloat()))
                contourPts.add(PointF(bbox.left.toFloat(), bbox.bottom.toFloat()))
            }
            out.add(FaceInfo(idx, bbox, lands, contourPts))
        }
        return out
    }

    private fun toCvPoints(list: List<PointF>): List<Point> {
        return list.map { Point(it.x.toDouble(), it.y.toDouble()) }
    }

    /** Bitmap->Mat gives 4-channel RGBA. The warp/blend code works on 3-channel images. */
    private fun rgbaToRgb(mat: Mat): Mat {
        val out = Mat()
        Imgproc.cvtColor(mat, out, Imgproc.COLOR_RGBA2RGB)
        return out
    }

    private fun rgbToRgba(mat: Mat): Mat {
        val out = Mat()
        Imgproc.cvtColor(mat, out, Imgproc.COLOR_RGB2RGBA)
        return out
    }

    private fun warpWithDelaunay(srcMat: Mat, srcPts: List<Point>, tgtPts: List<Point>, targetSize: Size): Mat {
        val w = targetSize.width.roundToInt()
        val h = targetSize.height.roundToInt()
        val warped = Mat.zeros(h, w, srcMat.type())

        val subdiv = Subdiv2D(Rect(0, 0, w, h))
        for (p in tgtPts) subdiv.insert(p)

        val tMat = MatOfFloat6()
        subdiv.getTriangleList(tMat)
        val tArr = tMat.toArray()

        fun findNearestIndex(x: Double, y: Double): Int {
            var best = -1
            var bestDist = Double.MAX_VALUE
            for ((i, p) in tgtPts.withIndex()) {
                val dx = p.x - x
                val dy = p.y - y
                val d = dx * dx + dy * dy
                if (d < bestDist) { bestDist = d; best = i }
            }
            return best
        }

        val triIndexSet = mutableSetOf<Triple<Int, Int, Int>>()
        var i = 0
        while (i + 5 < tArr.size) {
            val x1 = tArr[i].toDouble(); val y1 = tArr[i+1].toDouble()
            val x2 = tArr[i+2].toDouble(); val y2 = tArr[i+3].toDouble()
            val x3 = tArr[i+4].toDouble(); val y3 = tArr[i+5].toDouble()
            i += 6
            if (x1 < 0 || x1 > w || y1 < 0 || y1 > h) continue
            if (x2 < 0 || x2 > w || y2 < 0 || y2 > h) continue
            if (x3 < 0 || x3 > w || y3 < 0 || y3 > h) continue
            val idx1 = findNearestIndex(x1, y1)
            val idx2 = findNearestIndex(x2, y2)
            val idx3 = findNearestIndex(x3, y3)
            if (idx1 != idx2 && idx2 != idx3 && idx1 != idx3) {
                val tri = Triple(idx1, idx2, idx3)
                triIndexSet.add(tri)
            }
        }

        for (tri in triIndexSet) {
            val idxA = tri.first; val idxB = tri.second; val idxC = tri.third
            val srcTriShifted = MatOfPoint2f(Point(srcPts[idxA].x, srcPts[idxA].y), Point(srcPts[idxB].x, srcPts[idxB].y), Point(srcPts[idxC].x, srcPts[idxC].y))
            val dstTriPts = arrayOf(Point(tgtPts[idxA].x, tgtPts[idxA].y), Point(tgtPts[idxB].x, tgtPts[idxB].y), Point(tgtPts[idxC].x, tgtPts[idxC].y))
            val dstRect = Imgproc.boundingRect(MatOfPoint(*dstTriPts))

            if (dstRect.width <= 0 || dstRect.height <= 0) continue

            val srcRect = Imgproc.boundingRect(MatOfPoint(*srcTriShifted.toArray()))
            val srcRectSafe = Rect(
                maxOf(0, srcRect.x),
                maxOf(0, srcRect.y),
                min(srcMat.cols() - srcRect.x, srcRect.width),
                min(srcMat.rows() - srcRect.y, srcRect.height)
            )
            if (srcRectSafe.width <= 0 || srcRectSafe.height <= 0) continue
            val srcROI = Mat(srcMat, srcRectSafe)

            val srcTriRel = MatOfPoint2f(
                Point(srcPts[idxA].x - srcRectSafe.x, srcPts[idxA].y - srcRectSafe.y),
                Point(srcPts[idxB].x - srcRectSafe.x, srcPts[idxB].y - srcRectSafe.y),
                Point(srcPts[idxC].x - srcRectSafe.x, srcPts[idxC].y - srcRectSafe.y)
            )

            val dstTriRel = MatOfPoint2f(
                Point(tgtPts[idxA].x - dstRect.x, tgtPts[idxA].y - dstRect.y),
                Point(tgtPts[idxB].x - dstRect.x, tgtPts[idxB].y - dstRect.y),
                Point(tgtPts[idxC].x - dstRect.x, tgtPts[idxC].y - dstRect.y)
            )

            val warpMat = Imgproc.getAffineTransform(srcTriRel, dstTriRel)
            val warpedTriangle = Mat.zeros(dstRect.height, dstRect.width, srcMat.type())
            Imgproc.warpAffine(srcROI, warpedTriangle, warpMat, warpedTriangle.size(), Imgproc.INTER_LINEAR, Core.BORDER_REFLECT)

            val mask = Mat.zeros(dstRect.height, dstRect.width, CvType.CV_8UC1)
            val dstTriForMask = MatOfPoint(*dstTriRel.toArray().map { Point(it.x, it.y) }.toTypedArray())
            Imgproc.fillConvexPoly(mask, dstTriForMask, Scalar(255.0))

            val roiDst = Rect(dstRect.x, dstRect.y, dstRect.width, dstRect.height)
            if (roiDst.x < 0 || roiDst.y < 0 || roiDst.x + roiDst.width > warped.cols() || roiDst.y + roiDst.height > warped.rows()) continue
            val dstSub = Mat(warped, roiDst)

            val mask3c = Mat()
            Imgproc.cvtColor(mask, mask3c, Imgproc.COLOR_GRAY2BGR)
            val invertedMask = Mat()
            Core.bitwise_not(mask, invertedMask)
            val inv3c = Mat()
            Imgproc.cvtColor(invertedMask, inv3c, Imgproc.COLOR_GRAY2BGR)

            val dstBg = Mat()
            Core.bitwise_and(dstSub, inv3c, dstBg)
            val fg = Mat()
            Core.bitwise_and(warpedTriangle, mask3c, fg)
            Core.add(dstBg, fg, dstSub)
        }

        return warped
    }

    /** Gaussian blur that only averages over pixels inside the mask (normalized convolution). */
    private fun maskedBlur(ch: Mat, maskF: Mat, maskBlur: Mat, sigma: Double): Mat {
        val prod = Mat()
        Core.multiply(ch, maskF, prod)
        val blurred = Mat()
        Imgproc.GaussianBlur(prod, blurred, Size(0.0, 0.0), sigma)
        val out = Mat()
        Core.divide(blurred, maskBlur, out)
        return out
    }

    private fun blurredMask(maskF: Mat, sigma: Double): Mat {
        val mb = Mat()
        Imgproc.GaussianBlur(maskF, mb, Size(0.0, 0.0), sigma)
        Core.max(mb, Scalar(1e-3), mb)
        return mb
    }

    /**
     * Mask (inside the ROI crop) of the facial features that are NOT skin:
     * eyes, eyebrows and mouth. Needs the full 133-point contour layout.
     */
    private fun buildExclusionMask(pts: List<Point>, roi: Rect, faceWidth: Int): Mat {
        val excl = Mat.zeros(roi.height, roi.width, CvType.CV_8UC1)
        if (pts.size != 133) return excl

        fun fillHull(from: Int, toInclusive: Int) {
            val shifted = (from..toInclusive).map { Point(pts[it].x - roi.x, pts[it].y - roi.y) }
            val m = MatOfPoint(*shifted.toTypedArray())
            val hullIdx = MatOfInt()
            Imgproc.convexHull(m, hullIdx)
            val hp = hullIdx.toArray().map { shifted[it] }
            Imgproc.fillConvexPoly(excl, MatOfPoint(*hp.toTypedArray()), Scalar(255.0))
        }
        try {
            fillHull(36, 45)    // left eyebrow
            fillHull(46, 61)    // left eye
            fillHull(62, 71)    // right eyebrow
            fillHull(72, 87)    // right eye
            fillHull(93, 130)   // mouth
        } catch (e: Exception) {
            return Mat.zeros(roi.height, roi.width, CvType.CV_8UC1)
        }
        val d = maxOf(1, (faceWidth * 0.035).toInt())
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * d + 1, 2.0 * d + 1))
        val dil = Mat()
        Imgproc.dilate(excl, dil, k)
        return dil
    }

    /**
     * Blends the warped source face into the target using frequency separation in LAB:
     *  - fine detail (eyes, lips, nose shape) comes from the source face
     *  - overall lighting, shadows and skin color (low frequencies) come from the target
     * The low frequencies are measured on SKIN ONLY (eyes, eyebrows, mouth excluded), so
     * the dark features themselves do not turn into dark halos around the eyes.
     * Works only on a crop around the face. All Mats are 3-channel 8-bit RGB.
     */
    private fun blendWithColorTransfer(warpedSrc: Mat, tgtMat: Mat, faceMask: Mat, faceRect: Rect, tgtPts: List<Point>): Mat {
        val pad = (faceRect.width * 0.15).toInt()
        val x0 = maxOf(0, faceRect.x - pad)
        val y0 = maxOf(0, faceRect.y - pad)
        val x1 = min(tgtMat.cols(), faceRect.x + faceRect.width + pad)
        val y1 = min(tgtMat.rows(), faceRect.y + faceRect.height + pad)
        val roi = Rect(x0, y0, x1 - x0, y1 - y0)
        if (roi.width <= 4 || roi.height <= 4) return tgtMat.clone()

        val src = Mat(warpedSrc, roi)
        val tgt = Mat(tgtMat, roi)
        val msk = Mat(faceMask, roi)
        val faceWidth = faceRect.width

        // Where the warped source really has pixels (triangles can leave tiny gaps).
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)
        val nonBlack = Mat()
        Imgproc.threshold(gray, nonBlack, 0.0, 255.0, Imgproc.THRESH_BINARY)
        val valid = Mat()
        Core.bitwise_and(nonBlack, msk, valid)
        val closeK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(valid, valid, Imgproc.MORPH_CLOSE, closeK)
        Core.bitwise_and(valid, msk, valid)

        // Shrink a little so the blend stays inside the face (away from hair/background).
        val er = maxOf(2, (faceWidth * 0.03).toInt())
        val erodeK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * er + 1, 2.0 * er + 1))
        val inner = Mat()
        Imgproc.erode(valid, inner, erodeK)
        val region = if (Core.countNonZero(inner) >= 100) inner else valid
        if (Core.countNonZero(region) < 100) return tgtMat.clone()

        // Skin-only mask used to measure the low-frequency lighting.
        val excl = buildExclusionMask(tgtPts, roi, faceWidth)
        val notExcl = Mat()
        Core.bitwise_not(excl, notExcl)
        val skinTmp = Mat()
        Core.bitwise_and(region, notExcl, skinTmp)
        val skin = if (Core.countNonZero(skinTmp) >= 100) skinTmp else region

        // ---- Frequency separation in LAB ----
        val srcLab8 = Mat()
        Imgproc.cvtColor(src, srcLab8, Imgproc.COLOR_RGB2Lab)
        val tgtLab8 = Mat()
        Imgproc.cvtColor(tgt, tgtLab8, Imgproc.COLOR_RGB2Lab)
        val sCh = ArrayList<Mat>()
        Core.split(srcLab8, sCh)
        val tCh = ArrayList<Mat>()
        Core.split(tgtLab8, tCh)

        val maskF = Mat()
        skin.convertTo(maskF, CvType.CV_32F, 1.0 / 255.0)
        val sigmaL = maxOf(2.0, faceWidth * 0.18)   // lightness: broad shading comes from the target
        val sigmaC = maxOf(3.0, faceWidth * 0.30)   // color: almost fully from the target skin
        val maskBlurL = blurredMask(maskF, sigmaL)
        val maskBlurC = blurredMask(maskF, sigmaC)

        val outCh = ArrayList<Mat>()
        for (c in 0..2) {
            val sigma = if (c == 0) sigmaL else sigmaC
            val mb = if (c == 0) maskBlurL else maskBlurC
            val sF = Mat()
            sCh[c].convertTo(sF, CvType.CV_32F)
            val tF = Mat()
            tCh[c].convertTo(tF, CvType.CV_32F)
            val sLow = maskedBlur(sF, maskF, mb, sigma)
            val tLow = maskedBlur(tF, maskF, mb, sigma)
            val res = Mat()
            Core.subtract(sF, sLow, res)      // source detail
            Core.add(res, tLow, res)          // + target lighting/color
            val res8 = Mat()
            res.convertTo(res8, CvType.CV_8U) // saturates to 0..255
            outCh.add(res8)
        }
        val labOut = Mat()
        Core.merge(outCh, labOut)
        val corrected = Mat()
        Imgproc.cvtColor(labOut, corrected, Imgproc.COLOR_Lab2RGB)

        // ---- Feathered alpha blend ----
        var featherK = (faceWidth * 0.08).toInt()
        if (featherK % 2 == 0) featherK += 1
        if (featherK < 5) featherK = 5
        val alpha8 = Mat()
        Imgproc.GaussianBlur(region, alpha8, Size(featherK.toDouble(), featherK.toDouble()), 0.0)
        val alphaF = Mat()
        alpha8.convertTo(alphaF, CvType.CV_32F, 1.0 / 255.0)
        val alpha3 = Mat()
        Core.merge(listOf(alphaF, alphaF, alphaF), alpha3)

        val srcF = Mat()
        corrected.convertTo(srcF, CvType.CV_32FC3)
        val tgtF = Mat()
        tgt.convertTo(tgtF, CvType.CV_32FC3)
        val ones = Mat(alpha3.size(), CvType.CV_32FC3, Scalar(1.0, 1.0, 1.0))
        val invAlpha = Mat()
        Core.subtract(ones, alpha3, invAlpha)
        val fg = Mat()
        Core.multiply(srcF, alpha3, fg)
        val bg = Mat()
        Core.multiply(tgtF, invAlpha, bg)
        val outF = Mat()
        Core.add(fg, bg, outF)
        val blended = Mat()
        outF.convertTo(blended, CvType.CV_8UC3)

        // Put the blended crop back into a copy of the full target image.
        val result = tgtMat.clone()
        blended.copyTo(Mat(result, roi))
        return result
    }

    private fun applyDelaunayWarpAndBlend(srcMat: Mat, srcFaceInfo: FaceInfo, tgtMat: Mat, tgtFaceInfo: FaceInfo): Mat {
        var srcPts = toCvPoints(srcFaceInfo.contourPoints)
        var tgtPts = toCvPoints(tgtFaceInfo.contourPoints)
        // Keep all points inside their images (Subdiv2D requires this).
        val tMaxX = tgtMat.cols() - 1.0
        val tMaxY = tgtMat.rows() - 1.0
        val sMaxX = srcMat.cols() - 1.0
        val sMaxY = srcMat.rows() - 1.0
        tgtPts = tgtPts.map { Point(it.x.coerceIn(0.0, tMaxX), it.y.coerceIn(0.0, tMaxY)) }
        srcPts = srcPts.map { Point(it.x.coerceIn(0.0, sMaxX), it.y.coerceIn(0.0, sMaxY)) }

        val warpedSrc = warpWithDelaunay(srcMat, srcPts, tgtPts, Size(tgtMat.cols().toDouble(), tgtMat.rows().toDouble()))

        // Proper convex hull of the target face points as the face mask.
        val mask = Mat.zeros(tgtMat.rows(), tgtMat.cols(), CvType.CV_8UC1)
        var maskRect = Rect(0, 0, tgtMat.cols(), tgtMat.rows())
        try {
            val allPts = MatOfPoint(*tgtPts.toTypedArray())
            val hullIdx = MatOfInt()
            Imgproc.convexHull(allPts, hullIdx)
            val hullPts = hullIdx.toArray().map { tgtPts[it] }
            val hullMat = MatOfPoint(*hullPts.toTypedArray())
            Imgproc.fillConvexPoly(mask, hullMat, Scalar(255.0))
            maskRect = Imgproc.boundingRect(hullMat)
        } catch (ex: Exception) {
            Imgproc.rectangle(mask, Point(0.0, 0.0), Point(tgtMat.cols().toDouble(), tgtMat.rows().toDouble()), Scalar(255.0))
            maskRect = Rect(0, 0, tgtMat.cols(), tgtMat.rows())
        }

        return try {
            blendWithColorTransfer(warpedSrc, tgtMat, mask, maskRect, tgtPts)
        } catch (e: Exception) {
            tgtMat.clone()
        }
    }

    /** Detects + converts the source photo once. Returns null if no face was found. */
    suspend fun prepareSource(sourceBitmap: Bitmap, sourceFaceIndex: Int): PreparedSource? {
        val infos = detectFaces(sourceBitmap)
        if (infos.isEmpty()) return null
        val info = infos.getOrNull(sourceFaceIndex) ?: infos.first()
        return PreparedSource(rgbaToRgb(OpenCVUtils.bitmapToMat(sourceBitmap)), info)
    }

    /**
     * Swaps the prepared source face onto the given (already detected) target faces.
     * Faces whose contour layout does not match the source are left untouched.
     */
    fun swapPrepared(source: PreparedSource, targetBitmap: Bitmap, targets: List<FaceInfo>): Bitmap {
        var output = rgbaToRgb(OpenCVUtils.bitmapToMat(targetBitmap))
        for (t in targets) {
            if (t.contourPoints.size != source.info.contourPoints.size) continue
            output = applyDelaunayWarpAndBlend(source.mat, source.info, output, t)
        }
        return OpenCVUtils.matToBitmap(rgbToRgba(output))
    }

    suspend fun swapFacesWithDelaunay(
        sourceBitmap: Bitmap,
        sourceFaceIndex: Int,
        targetBitmap: Bitmap,
        selectedTargetIndices: List<Int>,
        progressCallback: (String, Float) -> Unit
    ): Bitmap? {
        progressCallback("Detecting source faces...", 0.02f)
        val prepared = prepareSource(sourceBitmap, sourceFaceIndex)
        if (prepared == null) {
            progressCallback("No face found in source image", 0f)
            return null
        }

        progressCallback("Detecting target faces...", 0.08f)
        val targetInfos = detectFaces(targetBitmap)
        if (targetInfos.isEmpty()) {
            progressCallback("No faces in target image", 0f)
            return null
        }

        val targets = selectedTargetIndices.mapNotNull { idx -> targetInfos.getOrNull(idx) }
        if (targets.isEmpty()) {
            progressCallback("No target faces selected", 0f)
            return null
        }
        val usable = targets.filter { it.contourPoints.size == prepared.info.contourPoints.size }
        if (usable.isEmpty()) {
            progressCallback("Face details not detected clearly - try a clearer, front-facing photo", 0f)
            return null
        }

        progressCallback("Swapping ${usable.size} face(s)...", 0.3f)
        val bmp = swapPrepared(prepared, targetBitmap, usable)
        progressCallback("Done", 1.0f)
        return bmp
    }
}
