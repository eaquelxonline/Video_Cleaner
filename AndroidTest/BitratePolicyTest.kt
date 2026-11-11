// Video_Cleaner — bit hızı politikası birim testleri (saf hesap; cihazda koşar ama MediaCodec kullanmaz).

package com.videocleaner.engine

import android.media.MediaFormat
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BitratePolicyTest {
    private val avc = MediaFormat.MIMETYPE_VIDEO_AVC
    private val hevc = MediaFormat.MIMETYPE_VIDEO_HEVC

    private fun input(
        w: Int = 1280, h: Int = 720, fps: Double = 30.0, src: Long = 1_148_000, gopUs: Long = 1_000_000,
        srcMime: String = avc, dstMime: String = avc, quality: Float = 1f,
    ) = BitratePolicy.Input(w, h, fps, src, gopUs, srcMime, dstMime, quality)

    /** Eski kural: max(beyan×1.25, 0.14 bpppf)×kalite — beyan yoksa (çoğu telefon dosyası) yalnızca taban. */
    private fun legacy(w: Int, h: Int, fps: Double, declared: Long, quality: Float): Long =
        (maxOf(declared * 5 / 4, (w.toLong() * h * fps * 0.14).toLong()) * quality).toLong()

    @Test
    fun standardStaysNearSourceInsteadOfInflating() {
        // Kullanıcının bildirdiği 720p örneği: kaynak 1,43 Mb/s, eski kural 3,87 Mb/s (2,6×) üretiyordu.
        val plan = BitratePolicy.plan(input(src = 1_432_000))
        val ratio = plan.bitrateBps.toDouble() / 1_432_000
        assertTrue("Standart oranı $ratio", ratio <= BitratePolicy.STANDARD_SOURCE_FACTOR + 1e-9)
        assertTrue("eski kuraldan küçük olmalı", plan.bitrateBps < legacy(1280, 720, 30.0, 0, 1f))
    }

    @Test
    fun standardIsBoundedForEverySizeAndRate() {
        for ((w, h) in listOf(640 to 360, 854 to 480, 1280 to 720, 1920 to 1080, 576 to 1024)) {
            for (fps in listOf(24.0, 30.0, 60.0)) {
                for (src in listOf(100_000L, 400_000L, 1_500_000L, 8_000_000L, 40_000_000L)) {
                    val plan = BitratePolicy.plan(input(w, h, fps, src))
                    assertTrue("${w}x$h@$fps src=$src → ${plan.bitrateBps}", plan.bitrateBps <= maxOf((src * BitratePolicy.STANDARD_SOURCE_FACTOR).toLong(), BitratePolicy.MIN_BITRATE))
                    assertTrue("kaynağın altına inilmez", plan.bitrateBps >= maxOf(src, BitratePolicy.MIN_BITRATE))
                }
            }
        }
    }

    @Test
    fun highTargetsAtLeastStandard() {
        for (src in listOf(300_000L, 2_000_000L, 12_000_000L, 40_000_000L)) {
            val std = BitratePolicy.plan(input(src = src, quality = 1f)).bitrateBps
            val high = BitratePolicy.plan(input(src = src, quality = 2f)).bitrateBps
            assertTrue("yüksek $high ≥ standart $std", high >= std)
        }
        // Düşük bit hızlı kaynakta Yüksek gerçekten daha fazla bit ister.
        assertEquals((300_000 * BitratePolicy.HIGH_SOURCE_FACTOR).toLong(), BitratePolicy.plan(input(src = 300_000, quality = 2f)).bitrateBps)
    }

    @Test
    fun highBitrateSourceIsNotInflatedNorReduced() {
        // Telefonda denenen 1080p örnek: 16 Mb/s. ×2 = 32 Mb/s israf olurdu; eşik (0,20 bpppf ≈ 12,4 Mb/s)
        // kaynağın altında kaldığı için hedef kaynağın kendisidir — ne şişirilir ne kaynağın altına inilir.
        val plan = BitratePolicy.plan(input(w = 1920, h = 1080, fps = 30.0, src = 16_000_000))
        assertEquals(16_000_000L, plan.bitrateBps)
    }

    @Test
    fun capAppliesBetweenSourceAndMultiple() {
        // 360p30 846 kb/s (qable örneği): ×2 = 1,69 Mb/s; eşik 0,20 bpppf = 1,382 Mb/s → hedef eşik.
        val plan = BitratePolicy.plan(input(w = 640, h = 360, fps = 30.0, src = 846_000))
        assertEquals((640.0 * 360 * 30 * BitratePolicy.STANDARD_CAP_BPPPF).toLong(), plan.bitrateBps)
        // 720p30 1,43 Mb/s: ×2 = 2,86 Mb/s eşiğin (5,53 Mb/s) altında → tam kat.
        assertEquals((1_432_000 * BitratePolicy.STANDARD_SOURCE_FACTOR).toLong(), BitratePolicy.plan(input(src = 1_432_000)).bitrateBps)
    }

    @Test
    fun unknownSourceBitrateKeepsLegacyFloor() {
        // Ölçüm yoksa kaliteyi düşürecek bir tahmin yapılmaz: eski taban aynen korunur.
        for (q in listOf(1f, 2f)) {
            val plan = BitratePolicy.plan(input(w = 1920, h = 1080, fps = 30.0, src = 0, quality = q))
            assertEquals(legacy(1920, 1080, 30.0, 0, q), plan.bitrateBps)
        }
    }

    @Test
    fun keyframeIntervalFollowsSourceWithinBounds() {
        assertEquals(4f, BitratePolicy.plan(input(gopUs = 4_000_000)).iFrameIntervalSec)
        assertEquals(1f, BitratePolicy.plan(input(gopUs = 500_000)).iFrameIntervalSec)
        assertEquals(10f, BitratePolicy.plan(input(gopUs = 30_000_000)).iFrameIntervalSec)
        assertEquals(1f, BitratePolicy.plan(input(gopUs = 0)).iFrameIntervalSec)
    }

    @Test
    fun tinySourcesGetSaneMinimum() {
        assertEquals(BitratePolicy.MIN_BITRATE, BitratePolicy.plan(input(src = 11_000)).bitrateBps)
    }

    @Test
    fun sameCodecHasNeutralFactor() {
        assertEquals(1.0, BitratePolicy.codecFactor(avc, avc), 0.0)
        assertEquals(1.0, BitratePolicy.codecFactor(hevc, hevc), 0.0)
    }
}
