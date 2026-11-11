// Video_Cleaner — cihaz üstü (instrumented) testler.
//
// Gerçek Android çalışma zamanında (ART) ve gerçek MediaCodec/MediaMuxer ile çalışır:
//   ./gradlew connectedDebugAndroidTest
// GitHub Actions'ta KVM'li emülatörde koşar (.github/workflows/android-test.yml).
// Her sonuç logcat'e "VCTEST" etiketiyle tek satır olarak yazılır.

package com.videocleaner.engine

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.ImageFormat
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.system.Os
import android.system.OsConstants
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.log10
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters

@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class VideoPipelineTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app: Context = instrumentation.targetContext
    private val testAssets = instrumentation.context.assets

    private fun log(line: String) {
        Log.i(TAG, line)
    }

    private fun asset(name: String): File {
        val target = File(app.cacheDir, name)
        testAssets.open("videos/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
        return target
    }

    private fun direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    // ---- 1. ART üzerinde yerel kitaplık ---------------------------------------

    @Test
    fun a_nativeCoreLoadsOnArt() {
        assertEquals(NativeCore.ABI_VERSION, NativeCore.nativeAbiVersion())
        val page = NativeCore.nativePageSize()
        assertEquals("çekirdeğin gördüğü sayfa boyutu işletim sistemininkiyle aynı olmalı", Os.sysconf(OsConstants._SC_PAGESIZE).toInt(), page)
        log(
            "DEVICE api=${Build.VERSION.SDK_INT} release=${Build.VERSION.RELEASE} abi=${Build.SUPPORTED_ABIS.joinToString("/")} " +
                "page=$page backend=\"${NativeCore.kernelBackendName()}\" model=\"${Build.MODEL}\"",
        )
        InstrumentationRegistry.getArguments().getString("expectPageSize")?.let { expected ->
            assertEquals("beklenen sayfa boyutu", expected.toInt(), page)
        }
        // Kullanıcının bildirdiği hata: nativeCreate(360x270) geçerli tutamağı hata sanıyordu.
        for ((w, h) in listOf(360 to 270, 1280 to 720, 1920 to 1080, 2 to 2, 128 to 96)) {
            CleanerSession(w, h).use { session ->
                // Çekirdek her yuvayı 64 bayta hizalar (Rust FrameRing: stride = round_up(kare, 64)).
                val stride = (w.toLong() * h * 3 / 2 + 63) / 64 * 64
                assertEquals(stride * 30, session.stats().poolBytes)
            }
        }
        // Geçersiz boyut istisna olarak, doğru hata koduyla dönmeli (çökme değil).
        try {
            CleanerSession(361, 270)
            throw AssertionError("tek sayılı genişlik kabul edildi")
        } catch (e: CoreException) {
            assertEquals(-1, e.code)
        }
    }

    // ---- 2. Halka + zamanlayıcı ART üzerinde bit düzeyinde -----------------------

    @Test
    fun b_coreIsBitExactOnArt() {
        val (w, h, total) = Triple(360, 270, 97)
        val bytes = w * h * 3 / 2
        val handle = NativeCore.nativeCreate(w, h)
        assertFalse("nativeCreate hata döndürdü: $handle", handle in -4096L..0L)
        try {
            assertEquals(0, NativeCore.nativeSetOption(handle, NativeCore.OPTION_AUTO_SCENE_CUT, 0))
            assertEquals(0, NativeCore.nativeSetOption(handle, NativeCore.OPTION_INTEGRITY_CHECK, 1))
            val info = direct(NativeCore.POLL_INFO_BYTES)
            val meta = direct(NativeCore.OUTPUT_META_BYTES)
            val frame = direct(bytes)
            val out = direct(bytes)
            fun pattern(seq: Int, i: Int): Byte = (seq * 31 + i * 7 + (i shr 8)).toByte()
            var pushed = 0
            var popped = 0
            var windows = 0
            loop@ while (true) {
                when (val step = NativeCore.nativePoll(handle, info)) {
                    NativeCore.STEP_NEED_FRAMES -> {
                        for (k in 0 until info.getInt(4)) {
                            if (pushed == total) {
                                assertEquals(0, NativeCore.nativeFinish(handle))
                                break
                            }
                            for (i in 0 until bytes) frame.put(i, pattern(pushed, i))
                            // Düzensiz ama artan zaman damgaları (değişken kare hızı).
                            assertEquals(pushed.toLong(), NativeCore.nativePushNv12(handle, frame, pushed * 16_667L + pushed % 7, 0))
                            pushed++
                        }
                    }
                    NativeCore.STEP_WINDOW -> {
                        windows++
                        assertEquals(0, NativeCore.nativeCompleteWindow(handle))
                    }
                    NativeCore.STEP_OUTPUT -> while (NativeCore.nativePopOutputNv12(handle, out, meta) == 1) {
                        assertEquals(popped.toLong(), meta.getLong(0))
                        assertEquals(popped * 16_667L + popped % 7, meta.getLong(8))
                        for (i in 0 until bytes) {
                            if (out.get(i) != pattern(popped, i)) throw AssertionError("kare $popped bayt $i değişmiş")
                        }
                        popped++
                    }
                    NativeCore.STEP_FINISHED -> break@loop
                    else -> throw AssertionError("beklenmeyen adım $step")
                }
            }
            assertEquals(total, pushed)
            assertEquals(total, popped)
            log("CORE frames=$total windows=$windows bitExact=true size=${w}x$h")
        } finally {
            NativeCore.nativeDestroy(handle)
        }
        assertEquals("yok edilmiş tutamak reddedilmeli", -12, NativeCore.nativeFinish(0))
    }

    // ---- 3. Gerçek videolar: çöz → çekirdek → kodla → MP4 ---------------------------

    private class Track(val mime: String, val times: List<Long>, val format: MediaFormat)

    private fun readTrack(file: File, prefix: String): Track? {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.path)
            val index = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true
            } ?: return null
            val format = extractor.getTrackFormat(index)
            extractor.selectTrack(index)
            val times = ArrayList<Long>()
            while (extractor.sampleTime >= 0) {
                times += extractor.sampleTime
                if (!extractor.advance()) break
            }
            return Track(format.getString(MediaFormat.KEY_MIME) ?: "", times.sorted(), format)
        } finally {
            extractor.release()
        }
    }

    private fun averageFps(times: List<Long>): Double =
        if (times.size > 1 && times.last() > times.first()) (times.size - 1) * 1_000_000.0 / (times.last() - times.first()) else 0.0

    private fun videoBytes(file: File): Long {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.path)
            val index = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true } ?: return 0
            extractor.selectTrack(index)
            var bytes = 0L
            while (extractor.sampleTrackIndex >= 0) {
                bytes += extractor.sampleSize.coerceAtLeast(0)
                if (!extractor.advance()) break
            }
            return bytes
        } finally {
            extractor.release()
        }
    }

    private fun canDecode(format: MediaFormat): Boolean {
        val probe = MediaFormat(format)
        probe.setString(MediaFormat.KEY_FRAME_RATE, null)     // findDecoderForFormat kare hızı ister-istemez
        return MediaCodecList(MediaCodecList.REGULAR_CODECS).findDecoderForFormat(probe) != null
    }

    private fun publish(file: File, name: String) {
        // Çıktıyı ortak depolamaya kopyala: test bitince uygulama kaldırılsa da `adb pull` ile alınabilir.
        runCatching {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/VideoCleanerTest")
            }
            val uri = app.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: return
            app.contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
        }
    }

    @Test
    fun c_everyVideoTranscodesWithoutFrameLoss() {
        val names = testAssets.list("videos")!!.sorted()
        assertTrue("test videosu yok", names.isNotEmpty())
        val failures = ArrayList<String>()
        var passed = 0
        var skipped = 0
        for (name in names) {
            val input = asset(name)
            val output = File(app.cacheDir, name.substringBeforeLast('.') + "_out.mp4")
            output.delete()
            val source = readTrack(input, "video/")
            if (source == null) {
                failures += "$name: video akışı okunamadı"
                continue
            }
            if (!canDecode(source.format)) {
                skipped++
                log("VIDEO name=$name status=SKIPPED reason=\"cihazda ${source.mime} çözücüsü yok\"")
                continue
            }
            val sourceAudio = readTrack(input, "audio/")
            val problems = ArrayList<String>()
            try {
                val result = VideoJob(app, Uri.fromFile(input), Uri.fromFile(output)).run()
                val video = readTrack(output, "video/")
                val audio = readTrack(output, "audio/")
                if (!result.intact) problems += "bütünlük bayrağı false"
                if (result.framesDecoded != source.times.size.toLong()) problems += "çözülen ${result.framesDecoded} ≠ kaynaktaki ${source.times.size}"
                if (video == null) {
                    problems += "çıktıda video akışı yok"
                } else {
                    if (video.times.size.toLong() != result.framesEncoded) problems += "çıktıdaki ${video.times.size} ≠ kodlanan ${result.framesEncoded}"
                    // Zaman damgaları: ilk kareye göre aralıklar aynı olmalı (MP4 zaman tabanı yuvarlaması ≤ 1 ms).
                    var worst = 0L
                    if (video.times.size == source.times.size) {
                        for (i in video.times.indices) {
                            worst = maxOf(worst, abs((video.times[i] - video.times[0]) - (source.times[i] - source.times[0])))
                        }
                        if (worst > 1000) problems += "zaman damgası sapması $worst µs"
                    }
                    if (sourceAudio != null && audio == null) problems += "ses kayboldu (${sourceAudio.mime}): ${result.audio}"
                    // Kaynağın kare hızı, çözünürlüğü ve yönü korunmalı (tek sayılı boyutta yalnızca son satır/sütun kırpılır).
                    val srcW = source.format.getInteger(MediaFormat.KEY_WIDTH)
                    val srcH = source.format.getInteger(MediaFormat.KEY_HEIGHT)
                    if (result.width != (srcW and 1.inv()) || result.height != (srcH and 1.inv())) problems += "çözünürlük değişti: ${srcW}x$srcH → ${result.width}x${result.height}"
                    val srcFps = averageFps(source.times)
                    val outFps = averageFps(video.times)
                    if (srcFps > 0 && abs(outFps - srcFps) / srcFps > 0.005) problems += "kare hızı değişti: ${"%.3f".format(srcFps)} → ${"%.3f".format(outFps)}"
                    val srcRot = if (source.format.containsKey(MediaFormat.KEY_ROTATION)) source.format.getInteger(MediaFormat.KEY_ROTATION) else 0
                    val outRot = if (video.format.containsKey(MediaFormat.KEY_ROTATION)) video.format.getInteger(MediaFormat.KEY_ROTATION) else 0
                    if (srcRot != outRot) problems += "yön değişti: $srcRot → $outRot"
                    // Bit hızı: kodlayıcıya İSTENEN hedef sınırlanır (politika). Çıktının gerçek boyutu kodlayıcıya göre
                    // değişir (bazı yazılım/donanım kodlayıcıları alt sınır uygular); o yalnızca raporlanır.
                    val srcVideoBps = result.sourceVideo?.videoBitsPerSecond ?: 0L
                    val requested = result.encoderSettings.bitrateBps.toLong()
                    if (srcVideoBps > 0 && requested > maxOf((srcVideoBps * BitratePolicy.STANDARD_SOURCE_FACTOR).toLong(), BitratePolicy.MIN_BITRATE) + 1) {
                        problems += "Standart hedef kaynağın ${BitratePolicy.STANDARD_SOURCE_FACTOR} katını aşıyor: $requested > $srcVideoBps×${BitratePolicy.STANDARD_SOURCE_FACTOR}"
                    }
                    val outVideoBytes = videoBytes(output)
                    val outVideoBps = if (video.times.size > 1) outVideoBytes * 8_000_000L * (video.times.size - 1) / video.times.size / maxOf(1L, video.times.last() - video.times.first()) else 0L
                    log(
                        "BITRATE name=$name srcVideoBps=$srcVideoBps declared=${result.sourceDeclaredBitrate} requested=$requested mode=${result.encoderSettings.bitrateMode} " +
                            "gop=${result.encoderSettings.iFrameIntervalSec} outVideoBps=$outVideoBps ratio=${if (srcVideoBps > 0) "%.2f".format(outVideoBps.toDouble() / srcVideoBps) else "-"} " +
                            "fps=${"%.3f".format(srcFps)}→${"%.3f".format(outFps)} reason=\"${result.encoderSettings.reason}\"",
                    )
                    log(
                        "VIDEO name=$name status=${if (problems.isEmpty()) "PASS" else "FAIL"} source=${source.mime} " +
                            "size=${result.width}x${result.height} decoded=${result.framesDecoded} encoded=${result.framesEncoded} " +
                            "outSamples=${video.times.size} maxPtsDriftUs=$worst ms=${result.elapsedMs} fps=${"%.1f".format(result.framesPerSecond)} " +
                            "encoder=${result.encoderName} windowsB=${result.stats.windowsB} cuts=${result.stats.sceneCuts} " +
                            "audioIn=${sourceAudio?.mime ?: "-"} audioOut=${audio?.mime ?: "-"} audioSamples=${audio?.times?.size ?: 0} " +
                            "audio=\"${result.audio}\" notes=\"${result.notes.joinToString(" | ")}\" bytes=${output.length()}",
                    )
                }
                if (name.startsWith("u")) publish(output, output.name)
            } catch (e: Exception) {
                problems += "${e.javaClass.simpleName}: ${e.message}"
                log("VIDEO name=$name status=FAIL error=\"${e.javaClass.simpleName}: ${e.message}\"")
            }
            if (problems.isEmpty()) passed++ else failures += "$name: ${problems.joinToString("; ")}"
            input.delete()
            output.delete()
        }
        log("SUMMARY videos=${names.size} passed=$passed failed=${failures.size} skipped=$skipped")
        assertTrue("başarısız videolar:\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    // ---- 4. Piksel düzeyi: onarılan bölge yazılıyor, geri kalanı bozulmuyor ----------

    private class Planes(val width: Int, val height: Int, val y: ByteArray, val u: ByteArray, val v: ByteArray)

    /** Dosyanın ilk [limit] karesini çözüp sıkı Y/U/V düzlemlerine kopyalar (çekirdekten bağımsız yol). */
    private fun decodeFrames(file: File, limit: Int): List<Planes> {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.path)
        val index = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith("video/") }
        val format = extractor.getTrackFormat(index)
        extractor.selectTrack(index)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        // Referans yol: tekrarlanabilir olsun diye varsa yazılım çözücü (donanım çözücüleri cihazdan cihaza değişir;
        // API 37 emülatöründe c2.goldfish çözücüsü stop() sırasında CodecException veriyor).
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val software = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull { info ->
            !info.isEncoder && !info.isAlias && info.isSoftwareOnly && info.supportedTypes.any { it.equals(mime, ignoreCase = true) }
        }
        val codec = if (software != null) MediaCodec.createByCodecName(software.name) else MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()
        val frames = ArrayList<Planes>()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        try {
            var guard = 0
            while (frames.size < limit && guard++ < 20_000) {
                if (!inputDone) {
                    val i = codec.dequeueInputBuffer(10_000)
                    if (i >= 0) {
                        val size = extractor.readSampleData(codec.getInputBuffer(i)!!, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(i, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val o = codec.dequeueOutputBuffer(info, 10_000)
                if (o >= 0) {
                    if (info.size > 0) {
                        val image = codec.getOutputImage(o)!!
                        assertEquals(ImageFormat.YUV_420_888, image.format)
                        val crop = image.cropRect
                        val (w, h) = crop.width() to crop.height()
                        val (py, pu, pv) = image.planes
                        val y = ByteArray(w * h)
                        val u = ByteArray((w / 2) * (h / 2))
                        val v = ByteArray((w / 2) * (h / 2))
                        for (row in 0 until h) {
                            for (x in 0 until w) y[row * w + x] = py.buffer.get((crop.top + row) * py.rowStride + (crop.left + x) * py.pixelStride)
                        }
                        for (row in 0 until h / 2) {
                            for (x in 0 until w / 2) {
                                val at = (crop.top / 2 + row) * pu.rowStride + (crop.left / 2 + x) * pu.pixelStride
                                u[row * (w / 2) + x] = pu.buffer.get(at)
                                v[row * (w / 2) + x] = pv.buffer.get(at)
                            }
                        }
                        frames += Planes(w, h, y, u, v)
                    }
                    val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(o, false)
                    if (end) break
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }
        return frames
    }

    @Test
    fun d_restoredRegionIsWrittenAndRestIsPreservedThroughEncoding() {
        val input = asset("s6_no_audio.mp4")           // 640×360, 30 kare/sn, 60 kare
        val output = File(app.cacheDir, "pixels_out.mp4")
        output.delete()
        // Sahte "onarıcı": her karede sabit bir bölgeyi bilinen renge boyar.
        val (rx, ry, rw, rh) = listOf(96, 64, 160, 96)
        val (cy, cu, cv) = Triple(200, 90, 170)
        val yPlane = direct(rw * rh).also { b -> repeat(rw * rh) { b.put(it, cy.toByte()) } }
        val wPlane = direct(rw * rh).also { b -> repeat(rw * rh) { b.put(it, 255.toByte()) } }
        val uvPlane = direct(rw * rh / 2).also { b -> for (i in 0 until rw * rh / 2 step 2) { b.put(i, cu.toByte()); b.put(i + 1, cv.toByte()) } }
        var estimates = 0
        val restorer = WindowRestorer { session, plan ->
            for (pos in 0 until 30) {
                if (plan.wantsEstimate(pos) && plan.isValid(pos)) {
                    session.submitEstimate(pos, rx, ry, rw, rh, yPlane, uvPlane, wPlane)
                    estimates++
                }
            }
        }
        val result = VideoJob(app, Uri.fromFile(input), Uri.fromFile(output), JobOptions(quality = 2.0f), restorer).run()
        assertTrue("bütünlük", result.intact)
        assertEquals("her kare için ROI kadar piksel değişmeli", result.framesDecoded * rw * rh, result.stats.modifiedPixels)

        val before = decodeFrames(input, 24)
        val after = decodeFrames(output, 24)
        assertEquals(before.size, after.size)
        assertTrue(before.size >= 20)
        var squared = 0.0
        var count = 0L
        var chromaDiff = 0.0
        var chromaCount = 0L
        var roiY = 0.0
        var roiU = 0.0
        var roiV = 0.0
        var roiCount = 0L
        for (f in before.indices) {
            val (a, b) = before[f] to after[f]
            assertEquals(a.width, b.width)
            assertEquals(a.height, b.height)
            val (w, h) = a.width to a.height
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val inside = x >= rx + 8 && x < rx + rw - 8 && y >= ry + 8 && y < ry + rh - 8
                    val near = x >= rx - 16 && x < rx + rw + 16 && y >= ry - 16 && y < ry + rh + 16
                    val i = y * w + x
                    if (inside) {
                        roiY += abs((b.y[i].toInt() and 255) - cy)
                        if (x % 2 == 0 && y % 2 == 0) {
                            val c = (y / 2) * (w / 2) + x / 2
                            roiU += abs((b.u[c].toInt() and 255) - cu)
                            roiV += abs((b.v[c].toInt() and 255) - cv)
                            roiCount++
                        }
                    } else if (!near) {
                        val d = (a.y[i].toInt() and 255) - (b.y[i].toInt() and 255)
                        squared += d * d
                        count++
                        if (x % 2 == 0 && y % 2 == 0) {
                            val c = (y / 2) * (w / 2) + x / 2
                            chromaDiff += abs((a.u[c].toInt() and 255) - (b.u[c].toInt() and 255)) + abs((a.v[c].toInt() and 255) - (b.v[c].toInt() and 255))
                            chromaCount += 2
                        }
                    }
                }
            }
        }
        val psnr = 10 * log10(255.0 * 255.0 / (squared / count).coerceAtLeast(1e-9))
        val chroma = chromaDiff / chromaCount
        val insideCount = (rw - 16).toLong() * (rh - 16) * before.size
        val (errY, errU, errV) = Triple(roiY / insideCount, roiU / roiCount, roiV / roiCount)
        log(
            "PIXELS frames=${before.size} estimates=$estimates outsidePsnrY=${"%.2f".format(psnr)}dB outsideChromaMeanAbs=${"%.2f".format(chroma)} " +
                "roiMeanAbsY=${"%.2f".format(errY)} roiMeanAbsU=${"%.2f".format(errU)} roiMeanAbsV=${"%.2f".format(errV)} encoder=${result.encoderName}",
        )
        // Kodlama kayıplıdır; eşikler "bozulma yok" (kanal karışması, satır kayması) ile "kodek gürültüsü"nü ayırır.
        assertTrue("ROI dışı parlaklık bozulmuş: PSNR $psnr dB", psnr > 24.0)
        assertTrue("ROI dışı renk bozulmuş: $chroma", chroma < 8.0)
        assertTrue("ROI parlaklığı yazılmamış: $errY", errY < 8.0)
        assertTrue("ROI rengi yazılmamış: U $errU V $errV", errU < 8.0 && errV < 8.0)
        input.delete()
        output.delete()
    }

    // ---- 5. Etkinlik + ön plan hizmeti ------------------------------------------------

    @Test
    fun e_activityAndForegroundServiceCompleteAJob() {
        val input = asset("s9_tiny_128x96.mp4")
        val output = File(app.cacheDir, "service_out.mp4")
        output.delete()
        val intent = Intent(app, App::class.java).putExtra("vc_input", input.path).putExtra("vc_output", output.path)
        ActivityScenario.launch<App>(intent).use {
            val deadline = SystemClock.elapsedRealtime() + 180_000
            while (JobState.result == null && JobState.error == null && SystemClock.elapsedRealtime() < deadline) {
                Thread.sleep(200)
            }
            assertEquals("hizmet hata verdi", null, JobState.error)
            val result = JobState.result
            assertNotNull("hizmet 180 sn içinde bitmedi", result)
            assertTrue("bütünlük", result!!.intact)
            assertTrue("çıktı dosyası boş", output.length() > 0)
            assertEquals(60L, result.framesEncoded)
            log("SERVICE status=PASS frames=${result.framesEncoded} ms=${result.elapsedMs} audio=\"${result.audio}\" bytes=${output.length()}")
        }
    }

    private companion object {
        const val TAG = "VCTEST"
    }
}
