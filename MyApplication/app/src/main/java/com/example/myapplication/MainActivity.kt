package com.example.myapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter


import java.io.ByteArrayOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import androidx.activity.compose.rememberLauncherForActivityResult

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        interpreter = Interpreter(loadModelFile(this))

        setContent {
            CameraScreen(interpreter)
        }
    }

    // =========================
    // LOAD MODEL
    // =========================
    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }
}

// =========================
// COMPOSABLE
// =========================
@Composable
fun CameraScreen(interpreter: Interpreter) {

    val context = androidx.compose.ui.platform.LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current

    var label by remember { mutableStateOf("Detecting...") }

    val executor = remember { Executors.newSingleThreadExecutor() }

    val classNames = listOf("depan", "miring_kanan", "miring_kiri", "nunduk")

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) {
        hasPermission = it
    }

    LaunchedEffect(Unit) {
        if (!hasPermission) {
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {

        if (hasPermission) {
            AndroidView(
                factory = { ctx ->

                    val previewView = PreviewView(ctx)

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                    cameraProviderFuture.addListener({

                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalyzer = ImageAnalysis.Builder()
                            .setTargetResolution(Size(224, 224))
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()

                        imageAnalyzer.setAnalyzer(executor) { imageProxy ->

                            val bitmap = imageProxyToBitmap(imageProxy)

                            if (bitmap != null) {
                                val input = preprocess(bitmap)

                                val output = Array(1) { FloatArray(4) }

                                interpreter.run(input, output)

                                val maxIdx = output[0].indices.maxByOrNull { output[0][it] } ?: 0
                                val confidence = output[0][maxIdx]

                                label = "${classNames[maxIdx]} (%.2f)".format(confidence)
                            }

                            imageProxy.close()
                        }

                        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalyzer
                        )

                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Text("Camera permission required")
        }

        // =========================
        // LABEL DISPLAY
        // =========================
        Text(
            text = label,
            modifier = Modifier
                .padding(16.dp)
        )
    }
}

// =========================
// PREPROCESS - FIXED
// Input harus 0-255 karena preprocess_input ada di dalam model
// =========================
fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
    val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

    for (y in 0 until 224) {
        for (x in 0 until 224) {
            val pixel = resized.getPixel(x, y)

            // ✅ Kirim nilai 0-255 mentah, JANGAN dinormalisasi
            // karena preprocess_input sudah ada di dalam model TFLite
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8)  and 0xFF).toFloat()
            val b = ((pixel)        and 0xFF).toFloat()

            input[0][y][x][0] = r
            input[0][y][x][1] = g
            input[0][y][x][2] = b
        }
    }

    return input
}

// =========================
// CONVERT ImageProxy → Bitmap + ROTATE - FIXED
// =========================
fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
    val planes = image.planes
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)

    val imageBytes = out.toByteArray()
    val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        ?: return null

    // ✅ FIX: Rotasi frame kamera agar orientasi benar
    val rotationDegrees = image.imageInfo.rotationDegrees
    return if (rotationDegrees != 0) {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    } else {
        bitmap
    }
}