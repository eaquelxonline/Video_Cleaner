// Video_Cleaner — 10-11. adım cihaz testleri: ayrı altyazı izi remux'u, HardwareBuffer çözme yolu,
// kodek anlaşması (H.264/HEVC/AV1/APV, HDR10), çıkarım köprüsü durumu ve iş kaydı (checkpoint).
// Sonuçlar logcat'e "VCTEST" etiketiyle yazılır.

package com.video.cleaner

import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.util.zip.CRC32
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CodecPathsTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val assets = instrumentation.context.assets

    private fun asset(path: String): File {
        val target = File(app.cacheDir, path.substringAfterLast('/'))
        assets.open(path).use { input -> target.outputStream().use { input.copyTo(it) } }
        return target
    }

    /** Bir izin her paketi: (ilk pakete göre zaman, bayrak, CRC32, boyut). */
    private data class Packet(val relUs: Long, val sync: Boolean, val crc: Long, val size: Int)

    private fun packets(file: File, prefix: String): Pair<String, List<Packet>>? {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(file.path)
            val track = (0 until ex.trackCount).firstOrNull { ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true } ?: return null
            ex.selectTrack(track)
            val buffer = ByteBuffer.allocateDirect(4 shl 20)
            val out = ArrayList<Packet>()
            var first = Long.MIN_VALUE
            while (ex.sampleTrackIndex >= 0) {
                val size = ex.readSampleData(buffer, 0)
                if (size < 0) break
                val bytes = ByteArray(size)
                buffer.position(0)
                buffer.get(bytes, 0, size)
                buffer.clear()
                val crc = CRC32().apply { update(bytes) }.value
                if (first == Long.MIN_VALUE) first = ex.sampleTime
                out += Packet(ex.sampleTime - first, ex.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0, crc, size)
                if (!ex.advance()) break
            }
            return ex.getTrackFormat(track).getString(MediaFormat.KEY_MIME)!! to out
        } finally {
            ex.release()
        }
    }

    private fun mimes(file: File): List<String> {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(file.path)
            return (0 until ex.trackCount).map { ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME) ?: "" }
        } finally {
            ex.release()
        }
    }

    @Test
    fun k_softSubtitleRemuxCopiesPacketsBitExact() {
        for (name in listOf("subs/srt_h264_aac.mkv", "subs/movtext_h264_aac.mp4")) {
            val input = asset(name)
            val output = File(app.cacheDir, input.nameWithoutExtension + "_remux.mp4").apply { delete() }
            try {
                val sourceMimes = mimes(input)
                val info = VideoInfo.probe(app, Uri.fromFile(input))
                assertNotNull("probe $name", info)
                Log.i(TAG, "REMUX name=$name tracks=$sourceMimes subtitles=${info!!.subtitleTracks}")
                val result = VideoJob(app, Uri.fromFile(input), Uri.fromFile(output), JobOptions(softSubtitleRemux = true)).run()
                assertTrue("remux bayrağı", result.remuxed)
                assertTrue("bütünlük", result.intact)
                val outMimes = mimes(output)
                assertTrue("çıktıda altyazı izi kalmamalı: $outMimes", outMimes.none(::isSubtitleMime))
                assertEquals("kaynak altyazı izleri raporlanmalı", sourceMimes.filter(::isSubtitleMime), result.removedSubtitleTracks)

                // Video: her paket bayt bayt, anahtar kare bayrağı ve göreli zaman damgasıyla aynı.
                val (srcMime, srcVideo) = packets(input, "video/")!!
                val (outMime, outVideo) = packets(output, "video/")!!
                assertEquals(srcMime, outMime)
                assertEquals("video paket sayısı", srcVideo.size, outVideo.size)
                assertEquals("video paketleri bit-exact", srcVideo.map { it.crc to it.size }, outVideo.map { it.crc to it.size })
                assertEquals("anahtar kareler", srcVideo.map { it.sync }, outVideo.map { it.sync })
                val drift = srcVideo.indices.maxOf { kotlin.math.abs(srcVideo[it].relUs - outVideo[it].relUs) }
                assertTrue("video zaman damgası sapması $drift µs", drift <= 1000)
                // Ses: AAC aynen kopyalanır.
                val (srcAudioMime, srcAudio) = packets(input, "audio/")!!
                val (outAudioMime, outAudio) = packets(output, "audio/")!!
                assertEquals(srcAudioMime, outAudioMime)
                assertEquals("ses paketleri bit-exact", srcAudio.map { it.crc }, outAudio.map { it.crc })
                Log.i(
                    TAG,
                    "REMUX name=$name status=PASS videoPackets=${outVideo.size} audioPackets=${outAudio.size} drift=$drift " +
                        "removed=${result.removedSubtitleTracks} out=$outMimes ms=${result.elapsedMs} notes=\"${result.notes.joinToString(" | ")}\"",
                )
            } finally {
                input.delete()
                output.delete()
            }
        }
    }

    @Test
    fun l_hardwareBufferDecodePathKeepsEveryFrame() {
        for (name in listOf("videos/s6_no_audio.mp4", "videos/s10_short_rot90.mp4", "videos/s1_odd_361x271.webm")) {
            val input = asset(name)
            val output = File(app.cacheDir, input.nameWithoutExtension + "_hb.mp4").apply { delete() }
            try {
                val (_, source) = packets(input, "video/")!!
                val result = VideoJob(app, Uri.fromFile(input), Uri.fromFile(output), JobOptions(hardwareBufferDecode = true)).run()
                val (_, out) = packets(output, "video/")!!
                val fellBack = result.notes.any { it.contains("bayt tamponu yolu kullanıldı") }
                val usedSurface = result.notes.any { it.contains("ImageReader") && !it.contains("çalışmadı") }
                Log.i(
                    TAG,
                    "HBDECODE name=$name status=${if (result.intact && out.size == source.size) "PASS" else "FAIL"} path=${if (fellBack) "byte_fallback" else if (usedSurface) "hardware_buffer" else "?"} " +
                        "frames=${source.size}->${out.size} size=${result.width}x${result.height} notes=\"${result.notes.joinToString(" | ")}\"",
                )
                assertTrue("bütünlük ($name)", result.intact)
                assertEquals("kare sayısı ($name)", source.size, out.size)
                assertTrue("yol belirsiz: ${result.notes}", fellBack || usedSurface)
                val drift = source.indices.maxOf { kotlin.math.abs(source.sortedBy { p -> p.relUs }[it].relUs - out.sortedBy { p -> p.relUs }[it].relUs) }
                assertTrue("zaman damgası sapması $drift µs ($name)", drift <= 1000)
            } finally {
                input.delete()
                output.delete()
            }
        }
    }

    @Test
    fun m_codecNegotiatorReportsAndAv1WhenAvailable() {
        val survey = CodecNegotiator.survey()
        for (s in survey) Log.i(TAG, "CODECS ${s.describe()} encoders=${s.encoders}")
        assertEquals(CodecNegotiator.ALL, survey.map { it.mime })
        assertTrue("H.264 kodlayıcı her cihazda olmalı", survey.first { it.mime == CodecNegotiator.AVC }.usable)
        assertEquals(listOf(CodecNegotiator.AVC, CodecNegotiator.HEVC), CodecNegotiator.order(JobOptions()))
        assertEquals(listOf(CodecNegotiator.HEVC, CodecNegotiator.AVC), CodecNegotiator.order(JobOptions(preferHevc = true)))
        assertFalse("APV MP4'e yazılmaz (doğrulanmadı)", CodecNegotiator.APV in CodecNegotiator.order(JobOptions(codec = CodecNegotiator.APV)))

        val av1 = survey.first { it.mime == CodecNegotiator.AV1 }
        if (!av1.usable) {
            Log.i(TAG, "AV1ENCODE status=SKIPPED reason=\"${av1.describe()}\"")
            return
        }
        val input = asset("videos/s9_tiny_128x96.mp4")
        val output = File(app.cacheDir, "s9_av1.mp4").apply { delete() }
        try {
            val result = VideoJob(app, Uri.fromFile(input), Uri.fromFile(output), JobOptions(codec = CodecNegotiator.AV1)).run()
            val (mime, out) = packets(output, "video/")!!
            Log.i(TAG, "AV1ENCODE status=${if (result.intact && mime == CodecNegotiator.AV1) "PASS" else "FAIL"} encoder=${result.encoderName} mime=$mime frames=${out.size} bytes=${output.length()}")
            assertTrue("bütünlük", result.intact)
            assertEquals(CodecNegotiator.AV1, result.outputMime)
            assertEquals(CodecNegotiator.AV1, mime)
            assertEquals(60, out.size)
        } finally {
            input.delete()
            output.delete()
        }
    }

    @Test
    fun n_inferenceRuntimeAndCheckpointAreHonest() {
        val report = InferenceRuntime.probe(app)
        Log.i(TAG, "RUNTIME ${InferenceRuntime.describe(report)}")
        assertNull("ağırlık/arka uç yokken onarıcı bağlanmamalı", InferenceRuntime.restorer(app))
        assertTrue(report.active.startsWith("kimlik"))

        val diag = Diagnostics.collect(app)
        val tester = diag.first { it.first == "VC-Tester" }.second
        assertTrue("VC-Tester bütünlüğü: $tester", tester.startsWith("bütünlük GEÇTİ"))

        JobCheckpoint.clear(app)
        val record = JobCheckpoint.Record(
            "content://in", "content://out", JobOptions(preferHevc = true, quality = 2f, softSubtitleRemux = true),
            JobCheckpoint.STATE_RUNNING, 120, 4_000_000, 10_000_000, 1L,
        )
        JobCheckpoint.save(app, record)
        assertEquals(record, JobCheckpoint.load(app))
        // Hizmet çalışmıyorken "çalışıyor" kaydı yarıda kalmış iştir.
        assertEquals(JobCheckpoint.STATE_INTERRUPTED, JobCheckpoint.pending(app)?.state)
        JobCheckpoint.mark(app, JobCheckpoint.STATE_TIMEOUT)
        assertEquals(JobCheckpoint.STATE_TIMEOUT, JobCheckpoint.pending(app)?.state)
        JobCheckpoint.clear(app)
        assertNull(JobCheckpoint.pending(app))
        Log.i(TAG, "CHECKPOINT status=PASS")
    }

    private companion object {
        const val TAG = "VCTEST"
    }
}
