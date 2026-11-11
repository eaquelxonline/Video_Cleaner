// Video_Cleaner — kodlayıcı yetenekleri ve bit hızı kalibrasyonu (cihaz üstü).
//
// f_encoderCapabilities: her cihazda koşar; yalnızca raporlar ("VCCAPS" satırları), hiçbir şeyi zorunlu tutmaz.
// g_bitrateCalibration: yalnızca `-Pandroid.testInstrumentationRunnerArguments.calibrate=true` ile koşar
//   (.github/workflows/bitrate-calibration.yml). Her kalibrasyon klibini aday ayarlarla kodlar, çıktıları
//   Movies/VideoCleanerCalib altına koyar ve her kodlamayı "VCCAL" satırı olarak yazar. Kalite ölçümü
//   (PSNR/SSIM/VMAF) bilgisayarda yapılır: .github/scripts/calibration_metrics.py

package com.video.cleaner

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
                        "widths=${video?.supportedWidths} heights=${video?.supportedHeights} align=${video?.widthAlignment}x${video?.heightAlignment} " +
                        "sizeOk(640x360,1280x720,1920x1080,576x1024)=${listOf(640 to 360, 1280 to 720, 1920 to 1080, 576 to 1024).map { (w, h) -> video?.isSizeSupported(w, h) }} " +
                        "profiles=${caps.profileLevels.joinToString(",") { "${it.profile}/${it.level}" }} maxInstances=${caps.maxSupportedInstances}",
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
        val qpMax: Int? = null,
        /** Doluysa ayar verilmez; üretimdeki BitratePolicy bu kalite ayarıyla çalışır. */
        val policyQuality: Float? = null,
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
        // Doğrulama turu: nihai üretim politikası (Standart ×2.0, Yüksek ×3.0, kaynak GOP, VBR) eski kurala karşı.
        // Önceki turlar (kaynak katı 1.0–3.0, GOP, QP sınırı, CQ, HEVC) için commit geçmişine bakın.
        val all = listOf(
            Candidate("avc_legacy", false, null) { c -> maxOf(c.declaredBps * 5L / 4, c.bpppf(0.14)) },
            Candidate("policy_std", false, null, policyQuality = 1f) { 0 },
            Candidate("policy_high", false, null, policyQuality = 2f) { 0 },
            Candidate("policy_std_hevc", true, null, policyQuality = 1f) { 0 },
        )
        // Gerçek cihazda (Firebase Test Lab) yalnızca üretim profilleri koşturulur: calibrateCandidates=policy_std,policy_high,...
        val wanted = InstrumentationRegistry.getArguments().getString("calibrateCandidates")?.split(',', '+')?.toSet()
        val candidates = all.filter { wanted == null || it.id in wanted }
        val only = InstrumentationRegistry.getArguments().getString("calibrateClips")?.split(',', '+')?.toSet()
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
                val supported = modeSupport(mime, mode, clip.width, clip.height, candidate.qpMax != null)
                if (supported == null) {
                    Log.i(CAL, "RESULT clip=${clip.name} cand=${candidate.id} status=UNSUPPORTED reason=\"${unsupportedReason(mime, mode, clip.width, clip.height, candidate.qpMax != null)}\"")
                    continue
                }
                val cq = candidate.cqFraction?.let { f -> supported.lower + ((supported.upper - supported.lower) * f).toInt() }
                val gop = if (candidate.gopFromSource) clip.avgGopSec.coerceIn(1.0, 10.0).toFloat() else 1f
                val tuning = EncoderTuning(
                    bitrateBps = candidate.bitrate(clip).coerceIn(64_000L, Int.MAX_VALUE.toLong()).toInt(),
                    bitrateMode = mode, cqQuality = cq, iFrameIntervalSec = gop, qpMax = candidate.qpMax,
                )
                val output = File(app.cacheDir, "${clip.name}__${candidate.id}.mp4")
                output.delete()
                try {
                    val options = candidate.policyQuality?.let { JobOptions(preferHevc = candidate.hevc, quality = it) }
                        ?: JobOptions(preferHevc = candidate.hevc, tuning = tuning)
                    val result = VideoJob(app, Uri.fromFile(clip.file), Uri.fromFile(output), options).run()
                    val out = measure(output)
                    val s = result.encoderSettings
                    Log.i(
                        CAL,
                        "RESULT clip=${clip.name} cand=${candidate.id} status=OK encoder=${result.encoderName} mime=${result.outputMime} " +
                            "mode=${s.bitrateMode} cq=${s.cqQuality} qpMax=${s.qpMax} targetBps=${s.bitrateBps} gopSec=${s.iFrameIntervalSec} " +
                            "outVideoBps=${out?.videoBitsPerSecond ?: 0} outBytes=${output.length()} srcBytes=${clip.file.length()} " +
                            "frames=${result.framesDecoded}/${result.framesEncoded} outSamples=${out?.samples ?: 0} outSync=${out?.syncSamples ?: 0} reason=\"${s.reason}\" " +
                            "ms=${result.elapsedMs} size=${result.width}x${result.height} audio=\"${result.audio}\"",
                    )
                    Log.i(CAL, "CHECK clip=${clip.name} cand=${candidate.id} ${verify(clip.file, output, result)}")
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

    // ---- 3. Çıktı doğrulaması (cihazda): kare kare PTS, ses paketleri, çözünürlük, en-boy, yön ----------

    private class TrackDump(
        val format: MediaFormat,
        val times: List<Long>,
        /** Her örneğin (paketin) MD5'i; yalnızca ses için doldurulur. */
        val hashes: List<String>,
        val firstPts: Long,
        val endUs: Long,
    )

    private fun dump(file: File, prefix: String, withHashes: Boolean): TrackDump? {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.path)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true } ?: return null
            val format = extractor.getTrackFormat(track)
            extractor.selectTrack(track)
            val buffer = java.nio.ByteBuffer.allocate(4 shl 20)
            val md = java.security.MessageDigest.getInstance("MD5")
            val times = ArrayList<Long>()
            val hashes = ArrayList<String>()
            while (extractor.sampleTrackIndex >= 0) {
                times += extractor.sampleTime
                if (withHashes) {
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    md.reset()
                    md.update(buffer.array(), 0, maxOf(size, 0))
                    hashes += md.digest().joinToString("") { "%02x".format(it) }
                }
                if (!extractor.advance()) break
            }
            val sorted = times.sorted()
            val frame = if (sorted.size > 1) (sorted.last() - sorted.first()) / (sorted.size - 1) else 0L
            return TrackDump(format, sorted, hashes, sorted.firstOrNull() ?: 0L, (sorted.lastOrNull() ?: 0L) + frame)
        } finally {
            extractor.release()
        }
    }

    private fun MediaFormat.int(key: String): Int? = if (containsKey(key)) getInteger(key) else null

    /** Kaynak ile çıktıyı karşılaştırıp tek satırlık denetim sonucu üretir (PASS/FAIL + ölçümler). */
    private fun verify(source: File, output: File, result: JobResult): String {
        val problems = ArrayList<String>()
        val sv = dump(source, "video/", false)!!
        val ov = dump(output, "video/", false)!!
        // Kare kare PTS: ilk kareye göre göreli zamanlar birebir karşılaştırılır (MP4 zaman tabanı yuvarlaması dahil).
        var maxDiff = 0L
        if (sv.times.size != ov.times.size) problems += "kare sayısı ${sv.times.size}→${ov.times.size}"
        for (i in 0 until minOf(sv.times.size, ov.times.size)) {
            maxDiff = maxOf(maxDiff, kotlin.math.abs((ov.times[i] - ov.firstPts) - (sv.times[i] - sv.firstPts)))
        }
        if (maxDiff > 1000) problems += "PTS sapması ${maxDiff}µs"
        val sw = sv.format.int(MediaFormat.KEY_WIDTH)!!
        val sh = sv.format.int(MediaFormat.KEY_HEIGHT)!!
        if (result.width != (sw and 1.inv()) || result.height != (sh and 1.inv())) problems += "çözünürlük ${sw}x$sh→${result.width}x${result.height}"
        val sSar = "${sv.format.int("sar-width") ?: 1}:${sv.format.int("sar-height") ?: 1}"
        val oSar = "${ov.format.int("sar-width") ?: 1}:${ov.format.int("sar-height") ?: 1}"
        if (sSar != oSar) problems += "piksel en-boy $sSar→$oSar"
        val sRot = sv.format.int(MediaFormat.KEY_ROTATION) ?: 0
        val oRot = ov.format.int(MediaFormat.KEY_ROTATION) ?: 0
        if (sRot != oRot) problems += "yön $sRot→$oRot"
        // Ses
        val sa = dump(source, "audio/", true)
        val oa = dump(output, "audio/", true)
        var audio = "-"
        if (sa != null) {
            if (oa == null) {
                problems += "ses yok"
            } else {
                val sMime = sa.format.getString(MediaFormat.KEY_MIME)
                val oMime = oa.format.getString(MediaFormat.KEY_MIME)
                val copied = result.audio.contains("kopyalandı")
                val identical = sa.hashes == oa.hashes
                if (copied && !identical) problems += "ses 'kopyalandı' diyor ama paketler farklı (${sa.hashes.size}→${oa.hashes.size})"
                if (copied && sMime != oMime) problems += "ses kodeki değişti $sMime→$oMime"
                if (sa.format.int(MediaFormat.KEY_SAMPLE_RATE) != oa.format.int(MediaFormat.KEY_SAMPLE_RATE)) problems += "örnekleme hızı değişti"
                if (sa.format.int(MediaFormat.KEY_CHANNEL_COUNT) != oa.format.int(MediaFormat.KEY_CHANNEL_COUNT)) problems += "kanal sayısı değişti"
                // A/V senkron (paket düzeyi tutarlılık): sesin görüntüye göre başlangıç ofseti kaynakla aynı olmalı.
                // Kopyada ≤ 1 ms (MP4 zaman tabanı yuvarlaması). Yeniden kodlamada video, AAC ön dolgusu kadar
                // bilerek kaydırılır (result.audioDelayUs); kalan fark ≤ 1 Opus ön atlaması + yuvarlama (10 ms).
                // İçerik tabanlı asıl senkron ölçümü ffmpeg ile calibration_metrics.py içinde yapılır.
                val sOff = sa.firstPts - sv.firstPts
                val oOff = oa.firstPts - ov.firstPts + if (copied) 0L else result.audioDelayUs
                val syncTol = if (copied) 1_000L else 10_000L
                if (kotlin.math.abs(oOff - sOff) > syncTol) problems += "A/V ofseti ${sOff}µs→${oOff}µs"
                val sDur = sa.endUs - sa.firstPts
                val oDur = oa.endUs - oa.firstPts - if (copied) 0L else result.audioDelayUs
                if (kotlin.math.abs(oDur - sDur) > 50_000L) problems += "ses süresi ${sDur / 1000}ms→${oDur / 1000}ms"
                audio = "src=$sMime/${sa.format.int(MediaFormat.KEY_SAMPLE_RATE)}Hz/${sa.format.int(MediaFormat.KEY_CHANNEL_COUNT)}ch/${sa.hashes.size}pk " +
                    "out=$oMime/${oa.format.int(MediaFormat.KEY_SAMPLE_RATE)}Hz/${oa.format.int(MediaFormat.KEY_CHANNEL_COUNT)}ch/${oa.hashes.size}pk " +
                    "packets=${if (identical) "identical" else "different"} mode=${if (copied) "copy" else "transcode"} avOffsetUs=$sOff→$oOff durMs=${sDur / 1000}→${oDur / 1000}"
            }
        }
        val status = if (problems.isEmpty()) "PASS" else "FAIL"
        return "status=$status frames=${sv.times.size}→${ov.times.size} ptsMaxDiffUs=$maxDiff size=${sw}x$sh→${result.width}x${result.height} sar=$sSar→$oSar rot=$sRot→$oRot " +
            "audio=\"$audio\" problems=\"${problems.joinToString("; ")}\""
    }

    // ---- 4. Hangi ses kodekleri MP4'e yeniden kodlanmadan taşınabilir? -------------------------------

    @Test
    fun h_audioPassthroughProbe() {
        // Kaynak ses paketleri MediaMuxer (MP4) ile aynen yazılır, sonra geri okunur; paket MD5'leri karşılaştırılır.
        val sources = listOf("calib/c20_360p30_h264_opus.mp4", "calib/c14_360p30_vp9.webm", "videos/s4_h264_vorbis.mkv", "videos/s5_mpeg4_mp3.mp4", "videos/u1_360p.mp4")
        for (asset in sources) {
            val src = File(app.cacheDir, "probe_" + asset.substringAfterLast('/'))
            val out = File(app.cacheDir, "probe_out.mp4")
            testAssets.open(asset).use { input -> src.outputStream().use { input.copyTo(it) } }
            out.delete()
            var mime = "?"
            val verdict = try {
                val extractor = MediaExtractor()
                extractor.setDataSource(src.path)
                val track = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith("audio/") }
                val format = extractor.getTrackFormat(track)
                mime = format.getString(MediaFormat.KEY_MIME)!!
                extractor.selectTrack(track)
                val muxer = android.media.MediaMuxer(out.path, android.media.MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                val index = try {
                    muxer.addTrack(format)
                } catch (e: Exception) {
                    muxer.release()
                    extractor.release()
                    throw IllegalStateException("addTrack reddetti: ${e.message}")
                }
                muxer.start()
                val buffer = java.nio.ByteBuffer.allocateDirect(1 shl 20)
                val info = android.media.MediaCodec.BufferInfo()
                var last = Long.MIN_VALUE
                while (extractor.sampleTrackIndex >= 0) {
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) break
                    val pts = maxOf(extractor.sampleTime, 0L).let { if (last == Long.MIN_VALUE) it else maxOf(it, last + 1) }
                    last = pts
                    buffer.position(0)
                    buffer.limit(size)
                    info.set(0, size, pts, android.media.MediaCodec.BUFFER_FLAG_KEY_FRAME)
                    muxer.writeSampleData(index, buffer, info)
                    extractor.advance()
                }
                muxer.stop()
                muxer.release()
                extractor.release()
                val a = dump(src, "audio/", true)!!
                val b = dump(out, "audio/", true)
                when {
                    b == null -> "BOZUK (çıktıda ses izi okunamadı)"
                    b.format.getString(MediaFormat.KEY_MIME) != mime -> "BOZUK (tür ${b.format.getString(MediaFormat.KEY_MIME)})"
                    a.hashes == b.hashes -> "OK (${a.hashes.size} paket birebir)"
                    else -> "FARKLI (${a.hashes.size}→${b.hashes.size} paket)"
                }
            } catch (e: Exception) {
                "DESTEKLENMİYOR (${e.javaClass.simpleName}: ${e.message})"
            }
            Log.i(CAL, "AUDIOPROBE api=${Build.VERSION.SDK_INT} source=$asset mime=$mime result=\"$verdict\"")
            Log.i(CAPS, "AUDIOPROBE api=${Build.VERSION.SDK_INT} source=$asset mime=$mime result=\"$verdict\"")
            src.delete()
            out.delete()
        }
    }

    /** İlk uygun kodlayıcının bu modu destekleyip desteklemediği; destekliyorsa kalite aralığı (CQ için). */
    private fun modeSupport(mime: String, mode: Int?, width: Int, height: Int, needQp: Boolean = false): android.util.Range<Int>? {
        for (info in MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos) {
            if (!info.isEncoder) continue
            val type = info.supportedTypes.firstOrNull { it.equals(mime, ignoreCase = true) } ?: continue
            val caps = runCatching { info.getCapabilitiesForType(type) }.getOrNull() ?: continue
            if (caps.videoCapabilities?.isSizeSupported(width, height) != true) continue
            if (needQp && (Build.VERSION.SDK_INT < 31 || !caps.isFeatureSupported("qp-bounds"))) continue
            val enc = caps.encoderCapabilities ?: continue
            if (mode == null || enc.isBitrateModeSupported(mode)) return enc.qualityRange
        }
        return null
    }

    private fun unsupportedReason(mime: String, mode: Int?, width: Int, height: Int, needQp: Boolean): String =
        MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder && it.supportedTypes.any { t -> t.equals(mime, ignoreCase = true) } }.joinToString("; ") { info ->
            val caps = runCatching { info.getCapabilitiesForType(mime) }.getOrNull()
            val v = caps?.videoCapabilities
            "${info.name}: size=${v?.isSizeSupported(width, height)} w=${v?.supportedWidths} h=${v?.supportedHeights} hFor$width=${runCatching { v?.getSupportedHeightsFor(width) }.getOrNull()} " +
                "mode=${mode?.let { caps?.encoderCapabilities?.isBitrateModeSupported(it) }} qp=${if (needQp) Build.VERSION.SDK_INT >= 31 && caps?.isFeatureSupported("qp-bounds") == true else "-"}"
        }.ifEmpty { "bu türde kodlayıcı yok" }

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
