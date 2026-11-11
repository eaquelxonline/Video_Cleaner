// Video_Cleaner — kodlayıcı yetenekleri ve bit hızı kalibrasyonu (cihaz üstü).
//
// f_encoderCapabilities: her cihazda koşar; yalnızca raporlar ("VCCAPS" satırları), hiçbir şeyi zorunlu tutmaz.
// g_bitrateCalibration: yalnızca `-Pandroid.testInstrumentationRunnerArguments.calibrate=true` ile koşar
//   (.github/workflows/bitrate-calibration.yml). Her kalibrasyon klibini aday ayarlarla kodlar, çıktıları
//   Movies/VideoCleanerCalib altına koyar ve her kodlamayı "VCCAL" satırı olarak yazar. Kalite ölçümü
//   (PSNR/SSIM/VMAF) bilgisayarda yapılır: .github/scripts/calibration_metrics.py

package com.videocleaner.engine

import android.content.ContentValues
import android.content.Context
import android.media.MediaCodecInfo
import android.media.MediaCodecInfo.EncoderCapabilities
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters

@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class EncoderCalibrationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app: Context = instrumentation.targetContext
    private val testAssets = instrumentation.context.assets

    // ---- 1. Kodlayıcı yetenekleri ---------------------------------------------------

    @Test
    fun f_encoderCapabilities() {
        val types = listOf(MediaFormat.MIMETYPE_VIDEO_AVC, MediaFormat.MIMETYPE_VIDEO_HEVC, MediaFormat.MIMETYPE_VIDEO_AV1, MediaFormat.MIMETYPE_VIDEO_VP9)
        for (info in MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos) {
            if (!info.isEncoder || info.isAlias) continue
            for (type in info.supportedTypes) {
                if (types.none { it.equals(type, ignoreCase = true) }) continue
                val caps = runCatching { info.getCapabilitiesForType(type) }.getOrNull() ?: continue
                val enc = caps.encoderCapabilities ?: continue
                val video = caps.videoCapabilities
                val modes = MODES.filter { (mode, _) -> enc.isBitrateModeSupported(mode) }.joinToString("/") { it.second }
                val qpBounds = Build.VERSION.SDK_INT >= 31 && caps.isFeatureSupported("qp-bounds")
                Log.i(
                    CAPS,
                    "ENCODER name=${info.name} type=$type hw=${info.isHardwareAccelerated} sw=${info.isSoftwareOnly} vendor=${info.isVendor} " +
                        "modes=${modes.ifEmpty { "-" }} quality=${enc.qualityRange} complexity=${enc.complexityRange} " +
                        "bitrate=${video?.bitrateRange} qpBounds=$qpBounds intraRefresh=${caps.isFeatureSupported(MediaCodecInfo.CodecCapabilities.FEATURE_IntraRefresh)} " +
                        "1080p30=${video?.areSizeAndRateSupported(1920, 1080, 30.0)} 1080p60=${video?.areSizeAndRateSupported(1920, 1080, 60.0)} " +
                        "maxInstances=${caps.maxSupportedInstances}",
                )
            }
        }
    }

    // ---- 2. Bit hızı kalibrasyonu ---------------------------------------------------

    /** Bir aday: kodlayıcı türü + bit hızı/mod hesabı. */
    private class Candidate(
        val id: String,
        val hevc: Boolean,
        val mode: Int?,
        val cqFraction: Float? = null,
        val gopFromSource: Boolean = false,
        val bitrate: (Clip) -> Long,
    )

    private class Clip(
        val name: String, val file: File, val width: Int, val height: Int, val fps: Double,
        val videoBps: Long, val declaredBps: Int, val avgGopSec: Double,
    ) {
        val pixelRate: Double get() = width.toDouble() * height * fps
        fun bpppf(value: Double): Long = (pixelRate * value).toLong()
    }

    @Test
    fun g_bitrateCalibration() {
        assumeTrue("yalnızca calibrate=true ile", InstrumentationRegistry.getArguments().getString("calibrate") == "true")
        val vbr = EncoderCapabilities.BITRATE_MODE_VBR
        val candidates = listOf(
            // Bugünkü sistem: kapsayıcı beyanı ×1.25 ile 0.14 bpppf tabanının büyüğü, mod verilmez.
            Candidate("avc_legacy", false, null) { c -> maxOf(c.declaredBps * 5L / 4, c.bpppf(0.14)) },
            Candidate("avc_f050", false, vbr) { c -> maxOf(c.videoBps, c.bpppf(0.05)) },
            Candidate("avc_f060", false, vbr) { c -> maxOf(c.videoBps, c.bpppf(0.06)) },
            Candidate("avc_f070", false, vbr) { c -> maxOf(c.videoBps, c.bpppf(0.07)) },
            Candidate("avc_s100", false, vbr) { c -> c.videoBps },
            Candidate("avc_s125", false, vbr) { c -> c.videoBps * 5 / 4 },
            Candidate("avc_s150", false, vbr) { c -> c.videoBps * 3 / 2 },
            Candidate("avc_s200", false, vbr) { c -> c.videoBps * 2 },
            Candidate("avc_s100_gopsrc", false, vbr, gopFromSource = true) { c -> c.videoBps },
            Candidate("hevc_legacy", true, null) { c -> maxOf(c.declaredBps * 5L / 4, c.bpppf(0.14)) },
            Candidate("hevc_s100", true, vbr) { c -> c.videoBps },
            Candidate("hevc_s085", true, vbr) { c -> c.videoBps * 85 / 100 },
            Candidate("hevc_s070", true, vbr) { c -> c.videoBps * 70 / 100 },
            Candidate("hevc_s055", true, vbr) { c -> c.videoBps * 55 / 100 },
            Candidate("hevc_cq50", true, EncoderCapabilities.BITRATE_MODE_CQ, cqFraction = 0.5f) { c -> c.videoBps },
            Candidate("hevc_cq70", true, EncoderCapabilities.BITRATE_MODE_CQ, cqFraction = 0.7f) { c -> c.videoBps },
            Candidate("hevc_cq90", true, EncoderCapabilities.BITRATE_MODE_CQ, cqFraction = 0.9f) { c -> c.videoBps },
            Candidate("avc_cq70", false, EncoderCapabilities.BITRATE_MODE_CQ, cqFraction = 0.7f) { c -> c.videoBps },
        )
        val only = InstrumentationRegistry.getArguments().getString("calibrateClips")?.split(',')?.toSet()
        val names = testAssets.list("calib")!!.sorted().filter { only == null || it.substringBefore('.') in only }
        for (name in names) {
            val clip = loadClip(name)
            Log.i(
                CAL,
                "SOURCE clip=${clip.name} size=${clip.width}x${clip.height} fps=${"%.3f".format(clip.fps)} videoBps=${clip.videoBps} " +
                    "declaredBps=${clip.declaredBps} gopSec=${"%.2f".format(clip.avgGopSec)} bytes=${clip.file.length()}",
            )
            for (candidate in candidates) {
                val mime = if (candidate.hevc) MediaFormat.MIMETYPE_VIDEO_HEVC else MediaFormat.MIMETYPE_VIDEO_AVC
                // VBR desteklenmiyorsa kodlayıcının varsayılan moduna düşülür (sonuç satırında mode=varsayılan görünür).
                val mode = if (candidate.mode == vbr && modeSupport(mime, vbr, clip.width, clip.height) == null) null else candidate.mode
                val supported = modeSupport(mime, mode, clip.width, clip.height)
                if (supported == null) {
                    Log.i(CAL, "RESULT clip=${clip.name} cand=${candidate.id} status=UNSUPPORTED")
                    continue
                }
                val cq = candidate.cqFraction?.let { f -> supported.lower + ((supported.upper - supported.lower) * f).toInt() }
                val gop = if (candidate.gopFromSource) clip.avgGopSec.coerceIn(1.0, 10.0).toFloat() else 1f
                val tuning = EncoderTuning(
                    bitrateBps = candidate.bitrate(clip).coerceIn(64_000L, Int.MAX_VALUE.toLong()).toInt(),
                    bitrateMode = mode, cqQuality = cq, iFrameIntervalSec = gop,
                )
                val output = File(app.cacheDir, "${clip.name}__${candidate.id}.mp4")
                output.delete()
                try {
                    val result = VideoJob(app, Uri.fromFile(clip.file), Uri.fromFile(output), JobOptions(preferHevc = candidate.hevc, tuning = tuning)).run()
                    val out = measure(output)
                    val s = result.encoderSettings
                    Log.i(
                        CAL,
                        "RESULT clip=${clip.name} cand=${candidate.id} status=OK encoder=${result.encoderName} mime=${result.outputMime} " +
                            "mode=${s.bitrateMode} cq=${s.cqQuality} targetBps=${s.bitrateBps} gopSec=${s.iFrameIntervalSec} " +
                            "outVideoBps=${out?.videoBitsPerSecond ?: 0} outBytes=${output.length()} srcBytes=${clip.file.length()} " +
                            "frames=${result.framesDecoded}/${result.framesEncoded} outSamples=${out?.samples ?: 0} outSync=${out?.syncSamples ?: 0} " +
                            "ms=${result.elapsedMs} size=${result.width}x${result.height} audio=\"${result.audio}\"",
                    )
                    publish(output, output.name)
                } catch (e: Exception) {
                    Log.i(CAL, "RESULT clip=${clip.name} cand=${candidate.id} status=FAIL error=\"${e.javaClass.simpleName}: ${e.message}\"")
                } finally {
                    output.delete()
                }
            }
            clip.file.delete()
        }
    }

    /** İlk uygun kodlayıcının bu modu destekleyip desteklemediği; destekliyorsa kalite aralığı (CQ için). */
    private fun modeSupport(mime: String, mode: Int?, width: Int, height: Int): android.util.Range<Int>? {
        for (info in MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos) {
            if (!info.isEncoder) continue
            val type = info.supportedTypes.firstOrNull { it.equals(mime, ignoreCase = true) } ?: continue
            val caps = runCatching { info.getCapabilitiesForType(type) }.getOrNull() ?: continue
            if (caps.videoCapabilities?.isSizeSupported(width, height) != true) continue
            val enc = caps.encoderCapabilities ?: continue
            if (mode == null || enc.isBitrateModeSupported(mode)) return enc.qualityRange
        }
        return null
    }

    private fun loadClip(name: String): Clip {
        val file = File(app.cacheDir, "calib_$name")
        testAssets.open("calib/$name").use { input -> file.outputStream().use { input.copyTo(it) } }
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.path)
            val track = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith("video/") }
            val format = extractor.getTrackFormat(track)
            val stats = measure(file)!!
            val fps = (stats.samples - 1) * 1_000_000.0 / (stats.durationUs - stats.durationUs / stats.samples)
            return Clip(
                name.substringBefore('.'), file, format.getInteger(MediaFormat.KEY_WIDTH), format.getInteger(MediaFormat.KEY_HEIGHT),
                fps, stats.videoBitsPerSecond, if (format.containsKey(MediaFormat.KEY_BIT_RATE)) format.getInteger(MediaFormat.KEY_BIT_RATE) else 0,
                stats.averageGopUs / 1e6,
            )
        } finally {
            extractor.release()
        }
    }

    /** Video izinin örnek tablosundan gerçek bit hızı (VideoJob'daki ölçümle aynı yöntem). */
    private fun measure(file: File): SourceVideoStats? {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.path)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith("video/") } ?: return null
            extractor.selectTrack(track)
            var bytes = 0L
            var samples = 0
            var sync = 0
            var first = Long.MAX_VALUE
            var last = Long.MIN_VALUE
            while (extractor.sampleTrackIndex >= 0) {
                bytes += extractor.sampleSize.coerceAtLeast(0)
                first = minOf(first, extractor.sampleTime)
                last = maxOf(last, extractor.sampleTime)
                if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) sync++
                samples++
                if (!extractor.advance()) break
            }
            if (samples < 2) return null
            return SourceVideoStats(bytes, samples, sync, (last - first) + (last - first) / (samples - 1))
        } finally {
            extractor.release()
        }
    }

    private fun publish(file: File, name: String) {
        runCatching {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/VideoCleanerCalib")
            }
            val uri = app.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: return
            app.contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
        }
    }

    private companion object {
        const val CAPS = "VCCAPS"
        const val CAL = "VCCAL"
        val MODES = listOf(
            EncoderCapabilities.BITRATE_MODE_CQ to "CQ",
            EncoderCapabilities.BITRATE_MODE_VBR to "VBR",
            EncoderCapabilities.BITRATE_MODE_CBR to "CBR",
            3 to "CBR_FD",
        )
    }
}
