package com.example.faceswap.processing

import android.graphics.PointF
import android.graphics.Rect

data class FaceInfo(
    val id: Int,
    val boundingBox: Rect,
    val landmarks: List<PointF>,
    val contourPoints: List<PointF>
)
