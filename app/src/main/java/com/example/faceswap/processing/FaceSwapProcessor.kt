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
            addContour(f.getContour(FaceContour.FACE))
            addContour(f.getContour(FaceContour.LEFT_EYEBROW_TOP))
            addContour(f.getContour(FaceContour.LEFT_EYE))
            addContour(f.getContour(FaceContour.RIGHT_EYEBROW_TOP))
            addContour(f.getContour(FaceContour.RIGHT_EYE))
            addContour(f.getContour(FaceContour.UPPER_LIP_TOP))
            addContour(f.getContour(FaceContour.LOWER_LIP_BOTTOM))
            addContour(f.getContour(FaceContour.NOSE_BOTTOM))
            if (contourPts.isEmpty()) {
                contourPts.add(PointF(bbox.left.toFloat(), bbox.top.toFloat()))
                contourPts.add(PointF(bbox.right.toFloat(), bbox.top.toFloat()))
                contourPts.add(PointF(bbox.right.toFloat(), bbox.bottom.toFloat()))
                contourPts.add(PointF(bbox.left.toFloat(), bbox.bottom.toFloat()))
            }
            // Keep the raw ML Kit contour points: each contour type has a fixed number of
            // points in a fixed order, so source and target points correspond 1-to-1.
            out.add(FaceInfo(idx, bbox, lands, contourPts))
        }
        return out
    }

    private fun resampleContour(points: List<PointF>, n: Int): List<PointF> {
        if (points.isEmpty()) return emptyList()
        if (points.size == 1) return List(n) { points.first() }
        val dists = DoubleArray(points.size)
        dists[0] = 0.0
        for (i in 1 until points.size) {
            val dx = (points[i].x - points[i - 1].x).toDouble()
            val dy = (points[i].y - points[i - 1].y).toDouble()
            dists[i] = dists[i - 1] + hypot(dx, dy)
        }
        val total = dists.last()
        if (total <= 0.0) return List(n) { points.first() }
        val resampled = mutableListOf<PointF>()
        val step = total / n.toDouble()
        var idx = 0
        for (i in 0 until n) {
            val target = i * step
            while (idx < dists.size - 1 && dists[idx + 1] < target) idx++
            val t0 = dists[idx]
            val t1 = dists[min(idx + 1, dists.size - 1)]
            if (t1 - t0 == 0.0) {
                resampled.add(points[idx])
            } else {
                val ratio = ((target - t0) / (t1 - t0)).toFloat()
                val p0 = points[idx]
                val p1 = points[min(idx + 1, points.size - 1)]
                val x = p0.x + (p1.x - p0.x) * ratio
                val y = p0.y + (p1.y - p0.y) * ratio
                resampled.add(PointF(x, y))
            }
        }
        return resampled
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

    /**
     * Matches the warped source face to the target's skin tone / lighting (Reinhard color
     * transfer in LAB space, measured only inside the face), then blends it in with a
     * feathered (soft-edged) mask. All Mats are 3-channel 8-bit RGB.
     */
    private fun blendWithColorTransfer(warpedSrc: Mat, tgtMat: Mat, faceMask: Mat, faceWidth: Int): Mat {
        // Where the warped source really has pixels (triangles can leave tiny gaps).
        val gray = Mat()
        Imgproc.cvtColor(warpedSrc, gray, Imgproc.COLOR_RGB2GRAY)
        val nonBlack = Mat()
        Imgproc.threshold(gray, nonBlack, 0.0, 255.0, Imgproc.THRESH_BINARY)
        val valid = Mat()
        Core.bitwise_and(nonBlack, faceMask, valid)
        val closeK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(valid, valid, Imgproc.MORPH_CLOSE, closeK)
        Core.bitwise_and(valid, faceMask, valid)

        // Shrink a little so the blend stays inside the face (away from hair/background).
        val er = maxOf(2, (faceWidth * 0.03).toInt())
        val erodeK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * er + 1, 2.0 * er + 1))
        val inner = Mat()
        Imgproc.erode(valid, inner, erodeK)
        val region = if (Core.countNonZero(inner) >= 100) inner else valid

        var corrected = warpedSrc
        if (Core.countNonZero(region) >= 100) {
            val srcLab = Mat()
            Imgproc.cvtColor(warpedSrc, srcLab, Imgproc.COLOR_RGB2Lab)
            val tgtLab = Mat()
            Imgproc.cvtColor(tgtMat, tgtLab, Imgproc.COLOR_RGB2Lab)

            val sMean = MatOfDouble(); val sStd = MatOfDouble()
            val tMean = MatOfDouble(); val tStd = MatOfDouble()
            Core.meanStdDev(srcLab, sMean, sStd, region)
            Core.meanStdDev(tgtLab, tMean, tStd, region)
            val sm = sMean.toArray(); val ss = sStd.toArray()
            val tm = tMean.toArray(); val ts = tStd.toArray()

            val chans = ArrayList<Mat>()
            Core.split(srcLab, chans)
            for (c in 0..2) {
                val scale = (ts[c] / maxOf(ss[c], 1.0)).coerceIn(0.7, 1.4)
                val shift = tm[c] - sm[c] * scale
                chans[c].convertTo(chans[c], CvType.CV_8U, scale, shift)
            }
            val labAdj = Mat()
            Core.merge(chans, labAdj)
            val rgbAdj = Mat()
            Imgproc.cvtColor(labAdj, rgbAdj, Imgproc.COLOR_Lab2RGB)
            corrected = rgbAdj
        }

        // Feathered alpha mask.
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
        tgtMat.convertTo(tgtF, CvType.CV_32FC3)
        val ones = Mat(alpha3.size(), CvType.CV_32FC3, Scalar(1.0, 1.0, 1.0))
        val invAlpha = Mat()
        Core.subtract(ones, alpha3, invAlpha)
        val fg = Mat()
        Core.multiply(srcF, alpha3, fg)
        val bg = Mat()
        Core.multiply(tgtF, invAlpha, bg)
        val outF = Mat()
        Core.add(fg, bg, outF)
        val out = Mat()
        outF.convertTo(out, CvType.CV_8UC3)
        return out
    }

    private fun applyDelaunayWarpAndBlend(srcMat: Mat, srcFaceInfo: FaceInfo, tgtMat: Mat, tgtFaceInfo: FaceInfo): Mat {
        var srcPts = toCvPoints(srcFaceInfo.contourPoints)
        var tgtPts = toCvPoints(tgtFaceInfo.contourPoints)
        if (srcPts.size != tgtPts.size) {
            // Some contour was missing on one face: fall back to equal-size resampling.
            srcPts = toCvPoints(resampleContour(srcFaceInfo.contourPoints, 100))
            tgtPts = toCvPoints(resampleContour(tgtFaceInfo.contourPoints, 100))
        }
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
            blendWithColorTransfer(warpedSrc, tgtMat, mask, maskRect.width)
        } catch (e: Exception) {
            tgtMat.clone()
        }
    }

    suspend fun swapFacesWithDelaunay(
        sourceBitmap: Bitmap,
        sourceFaceIndex: Int,
        targetBitmap: Bitmap,
        selectedTargetIndices: List<Int>,
        progressCallback: (String, Float) -> Unit
    ): Bitmap? {
        progressCallback("Detecting source faces...", 0.02f)
        val sourceInfos = detectFaces(sourceBitmap, samplingPoints = 100)
        if (sourceInfos.isEmpty()) {
            progressCallback("No face found in source image", 0f)
            return null
        }
        val srcInfo = sourceInfos.getOrNull(sourceFaceIndex) ?: sourceInfos.first()

        progressCallback("Detecting target faces...", 0.08f)
        val targetInfos = detectFaces(targetBitmap, samplingPoints = 100)
        if (targetInfos.isEmpty()) {
            progressCallback("No faces in target image", 0f)
            return null
        }

        // Convert to 3-channel RGB: the warp/blend code (masks, color transfer) needs 3 channels.
        val tgtMat = rgbaToRgb(OpenCVUtils.bitmapToMat(targetBitmap))
        val srcMat = rgbaToRgb(OpenCVUtils.bitmapToMat(sourceBitmap))

        var output = tgtMat.clone()
        val targets = selectedTargetIndices.mapNotNull { idx -> targetInfos.getOrNull(idx) }
        if (targets.isEmpty()) {
            progressCallback("No target faces selected", 0f)
            return null
        }

        val total = targets.size
        for ((i, tgtInfo) in targets.withIndex()) {
            val stageBase = 0.1f + (i.toFloat() / total) * 0.7f
            progressCallback("Warping face ${i+1}/$total", stageBase)
            val blended = applyDelaunayWarpAndBlend(srcMat, srcInfo, output, tgtInfo)
            output = blended
            progressCallback("Completed ${i+1}/$total", stageBase + 0.7f * ((i + 1).toFloat() / total))
        }
        progressCallback("Finalizing", 0.98f)
        val bmp = OpenCVUtils.matToBitmap(rgbToRgba(output))
        progressCallback("Done", 1.0f)
        return bmp
    }
}
