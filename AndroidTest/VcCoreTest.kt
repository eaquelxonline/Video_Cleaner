// Video_Cleaner — VC-Core motor arayüzü (vc*) cihaz testleri.
//
// Planın JNI sözleşmesini (vcInitEngine / vcPushHardwareFrame / vcProcessWindow / vcRunTesterAudit)
// gerçek Android çalışma zamanında doğrular. Sonuçlar logcat'e "VCTEST" etiketiyle yazılır.

package com.video.cleaner

import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VcCoreTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val assets = instrumentation.context.assets

    private fun direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    /** Motoru tek adım boşaltır: pencereleri işler, çıkan kareleri `sink`'e verir. Bitti mi? */
    private fun drain(ctx: Long, info: ByteBuffer, out: ByteBuffer, meta: ByteBuffer, sink: (ByteArray, Long) -> Unit): Int {
        while (true) {
            when (val step = Core.vcPoll(ctx, info)) {
                Core.STEP_WINDOW -> assertTrue(Core.vcProcessWindow(ctx, false) > 0)
                Core.STEP_OUTPUT -> while (Core.vcPopOutputNv12(ctx, out, meta) == 1) {
                    val bytes = ByteArray(out.capacity())
                    out.position(0)
                    out.get(bytes)
                    out.clear()
                    sink(bytes, meta.getLong(8))
                }
                else -> return step
            }
        }
    }

    /** Görüntünün üç düzleminden sıkı NV12 (motorun saklama biçimi) üretir. */
    private fun nv12(image: Image, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h * 3 / 2)
        val (y, u, v) = image.planes
        for (row in 0 until h) {
            for (x in 0 until w) out[row * w + x] = y.buffer.get(row * y.rowStride + x * y.pixelStride)
        }
        for (row in 0 until h / 2) {
            for (x in 0 until w / 2) {
                out[w * h + row * w + 2 * x] = u.buffer.get(row * u.rowStride + x * u.pixelStride)
                out[w * h + row * w + 2 * x + 1] = v.buffer.get(row * v.rowStride + x * v.pixelStride)
            }
        }
        return out
    }

    @Test
    fun i_hardwareBufferPathIsBitExactAndAudited() {
        val file = File(app.cacheDir, "vc_hw.mp4")
        assets.open("videos/s6_no_audio.mp4").use { input -> file.outputStream().use { input.copyTo(it) } }
        val extractor = MediaExtractor().apply { setDataSource(file.path) }
        val track = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith("video/") }
        val format = extractor.getTrackFormat(track)
        extractor.selectTrack(track)
        val w = format.getInteger(MediaFormat.KEY_WIDTH)
        val h = format.getInteger(MediaFormat.KEY_HEIGHT)
        val reader = ImageReader.newInstance(w, h, ImageFormat.YUV_420_888, 4, HardwareBuffer.USAGE_CPU_READ_OFTEN)
        // Emülatörün donanım çözücüsü (c2.goldfish, API 37) bozuk olabildiğinden yazılım çözücü tercih edilir.
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val software = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull { info ->
            !info.isEncoder && !info.isAlias && info.isSoftwareOnly && info.supportedTypes.any { it.equals(mime, ignoreCase = true) }
        }
        val decoder = if (software != null) MediaCodec.createByCodecName(software.name) else MediaCodec.createDecoderByType(mime)
        decoder.configure(format, reader.surface, null, 0)
        decoder.start()

        val ctx = Core.vcInitEngine(Core.ENGINE_MODE_AUTO, false)
        assertTrue("vcInitEngine: $ctx", ctx > 0)
        val info = direct(Core.POLL_INFO_BYTES)
        val out = direct(w * h * 3 / 2)
        val meta = direct(Core.OUTPUT_META_BYTES)
        val expected = ArrayDeque<Pair<ByteArray, Long>>()
        var compared = 0
        val sink: (ByteArray, Long) -> Unit = { bytes, pts ->
            val (want, wantPts) = expected.removeFirst()
            assertEquals("PTS", wantPts, pts)
            assertArrayEquals("kare $compared bit-exact olmalı", want, bytes)
            compared++
        }
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var pushed = 0
        var cpuFallback = false
        try {
            while (!outputDone) {
                if (!inputDone) {
                    val i = decoder.dequeueInputBuffer(10_000)
                    if (i >= 0) {
                        val size = extractor.readSampleData(decoder.getInputBuffer(i)!!, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            decoder.queueInputBuffer(i, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val o = decoder.dequeueOutputBuffer(bufferInfo, 10_000)
                if (o < 0) continue
                val end = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                val render = bufferInfo.size > 0
                decoder.releaseOutputBuffer(o, render)
                if (render) {
                    var image: Image? = null
                    val deadline = SystemClock.uptimeMillis() + 2_000
                    while (image == null && SystemClock.uptimeMillis() < deadline) {
                        image = reader.acquireNextImage()
                        if (image == null) SystemClock.sleep(2)
                    }
                    requireNotNull(image) { "ImageReader kare vermedi" }
                    image.use { img ->
                        val pts = img.timestamp / 1000
                        expected.addLast(nv12(img, w and 1.inv(), h and 1.inv()) to pts)
                        val hb = img.hardwareBuffer!!
                        fun push(): Int {
                            if (!cpuFallback) {
                                val r = Core.vcPushHardwareFrame(ctx, hb, pts, 0)
                                if (r != UNSUPPORTED) return r
                                // Bu cihazın ImageReader tamponu CPU düzlemlerine ayrılamıyor: CPU yoluna düş.
                                cpuFallback = true
                                Log.i(TAG, "VCCORE hardware_buffer lockPlanes desteklenmiyor; CPU düzlem yoluna düşüldü")
                            }
                            val nv = expected.last().first
                            val buf = direct(nv.size).apply { put(nv); flip() }
                            return Core.vcPushNv12(ctx, w and 1.inv(), h and 1.inv(), buf, pts, 0)
                        }
                        var seq = push()
                        if (seq == -3) { // halka dolu: boşaltıp yeniden dene
                            drain(ctx, info, out, meta, sink)
                            seq = push()
                        }
                        hb.close()
                        assertEquals("vcPushHardwareFrame sıra numarası", pushed, seq)
                        pushed++
                    }
                    drain(ctx, info, out, meta, sink)
                }
                if (end) outputDone = true
            }
            assertEquals(0, Core.vcFinish(ctx))
            var guard = 0
            while (drain(ctx, info, out, meta, sink) != Core.STEP_FINISHED && guard++ < 1000) Unit
            assertEquals("tüm kareler çıkmalı", pushed, compared)
            assertTrue("kare itilmedi", pushed > 0)
            val audit = Core.vcRunTesterAudit(ctx)
            Log.i(TAG, "VCAUDIT hardware=$audit")
            assertTrue(audit, audit.contains(if (cpuFallback) "\"frames_cpu\":$pushed" else "\"frames_hw\":$pushed"))
            assertTrue(audit, audit.contains("\"bit_exact\":true"))
            assertTrue(audit, audit.contains("\"violations\":0"))
            Log.i(TAG, "VCCORE status=PASS path=${if (cpuFallback) "cpu_fallback" else "hardware_buffer"} frames=$pushed storage=${Regex("\"storage\":\"(\\w+)\"").find(audit)?.groupValues?.get(1)}")
        } finally {
            Core.vcDestroy(ctx)
            decoder.stop()
            decoder.release()
            reader.close()
            extractor.release()
            file.delete()
        }
    }

    @Test
    fun j_modesAndPlatformAudit() {
        val (w, h) = 320 to 180
        val frame = direct(w * h * 3 / 2)
        for (mode in listOf(Core.ENGINE_MODE_AUTO, Core.ENGINE_MODE_A)) {
            val ctx = Core.vcInitEngine(mode, true)
            val info = direct(Core.POLL_INFO_BYTES)
            val out = direct(w * h * 3 / 2)
            val meta = direct(Core.OUTPUT_META_BYTES)
            var popped = 0
            try {
                for (i in 0 until 40) {
                    for (b in 0 until frame.capacity()) frame.put(b, ((b * 7 + i * 13) and 0xff).toByte())
                    while (true) {
                        val seq = Core.vcPushNv12(ctx, w, h, frame, i * 33_333L, 0)
                        if (seq == -3) {
                            drain(ctx, info, out, meta) { _, _ -> popped++ }
                            continue
                        }
                        assertEquals(i, seq)
                        break
                    }
                }
                Core.vcFinish(ctx)
                var guard = 0
                while (drain(ctx, info, out, meta) { _, _ -> popped++ } != Core.STEP_FINISHED && guard++ < 1000) Unit
                assertEquals(40, popped)
                val audit = Core.vcRunTesterAudit(ctx)
                Log.i(TAG, "VCAUDIT mode=$mode $audit")
                assertTrue(audit, audit.contains("\"frames_equal\":true"))
                if (mode == Core.ENGINE_MODE_A) assertTrue(audit, audit.contains("\"windows_a\":40"))
                // Sayfa boyutu 16 KB olan cihazda .so LOAD bölütleri de en az 16 KB hizalı olmalı.
                val page = Regex("\"page_size\":(\\d+)").find(audit)!!.groupValues[1].toInt()
                val align = Regex("\"elf_load_align\":(\\d+)").find(audit)!!.groupValues[1].toLong()
                assertTrue("ELF LOAD hizası $align < sayfa $page", align >= page)
                assertTrue("ELF LOAD hizası 16 KB değil: $align", align >= 16384)
            } finally {
                Core.vcDestroy(ctx)
            }
        }
    }

    private companion object {
        const val TAG = "VCTEST"
        const val UNSUPPORTED = -14
    }
}
