// Video_Cleaner v2.0 — Android çekirdek katmanı
//
//   Core      Rust çekirdeğinin (libvideo_cleaner.so) JNI yüzeyi
//   CleanerSession  Tutamağı ve tamponları saran güvenli Kotlin arayüzü
//   VideoJob        MediaCodec ile çöz → Rust halkası/zamanlayıcı → MediaCodec ile kodla → MP4
//                   (ayrı altyazı izi varsa: yeniden kodlamadan remux, 0 yapay zekâ)
//   CodecNegotiator H.264 / HEVC / AV1 / APV ve HDR10 kodlayıcı yeteneklerine göre çıktı türü
//   InferenceRuntime Vulkan / LiteRT / ONNX köprüsünün ve model ağırlıklarının durumu
//   JobCheckpoint   Ön plan hizmeti süre sınırı (Android 15+: mediaProcessing 6 saat) için iş kaydı
//   CleanerService  Uzun işi taşıyan ön plan hizmeti
//
// DURUM: Onarım modeli (Vulkan çıkarımı) henüz bağlı değildir. `WindowRestorer`
// verilmediğinde her pencere "tahminsiz" tamamlanır ve kareler DEĞİŞTİRİLMEDEN
// yeniden kodlanır. Bu, 1. aşamanın (kare kaybı olmadan oku-yaz) altyapı yoludur.
//
// Kopya sayısı: çözücü görüntüsü → halka (1 kopya), halka → kodlayıcı görüntüsü
// (1 kopya). Sıfır kopya hattı (AHardwareBuffer/Vulkan) GPU aşamasının işidir.

package com.video.cleaner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.os.SystemClock
import android.provider.DocumentsContract
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.max

// =============================================================================
// 1. JNI yüzeyi
// =============================================================================

/**
 * Rust çekirdeğinin JNI işlevleri. İmzalar `Rust/Core.rs` içindeki `jni` modülüyle
 * birebir eşleşmelidir. Tüm tamponlar DOĞRUDAN (direct) `ByteBuffer` olmalıdır;
 * hatalar istisna yerine negatif kod olarak döner.
 */
object Core {
    const val ABI_VERSION = 2

    const val STEP_NEED_FRAMES = 0
    const val STEP_WINDOW = 1
    const val STEP_OUTPUT = 2
    const val STEP_FINISHED = 3

    const val POLL_INFO_BYTES = 64
    const val OUTPUT_META_BYTES = 32
    const val STATS_BYTES = 112

    const val HINT_NORMAL = 0
    const val HINT_PRECISION = 1
    const val HINT_BYPASS = 2

    const val OPTION_AUTO_SCENE_CUT = 1
    const val OPTION_INTEGRITY_CHECK = 2

    const val OUT_FLAG_RESTORED = 1 shl 0
    const val OUT_FLAG_MODE_A = 1 shl 1
    const val OUT_FLAG_FUSED = 1 shl 2
    const val OUT_FLAG_BYPASS = 1 shl 3
    const val OUT_FLAG_LOW_CONTEXT = 1 shl 4
    const val OUT_FLAG_SCENE_START = 1 shl 5

    init {
        System.loadLibrary("video_cleaner")
    }

    @JvmStatic external fun nativeAbiVersion(): Int
    @JvmStatic external fun nativeKernelBackend(): Int
    @JvmStatic external fun nativePageSize(): Int
    @JvmStatic external fun nativeCreate(width: Int, height: Int): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeSetOption(handle: Long, key: Int, value: Int): Int
    @JvmStatic external fun nativePoll(handle: Long, info: ByteBuffer): Int

    @JvmStatic
    external fun nativePushYuv420(
        handle: Long,
        y: ByteBuffer, yOffset: Int, yRowStride: Int, yPixelStride: Int,
        u: ByteBuffer, uOffset: Int,
        v: ByteBuffer, vOffset: Int,
        uvRowStride: Int, uvPixelStride: Int,
        ptsUs: Long, flags: Int,
    ): Long

    @JvmStatic external fun nativePushNv12(handle: Long, frame: ByteBuffer, ptsUs: Long, flags: Int): Long
    @JvmStatic external fun nativeFinish(handle: Long): Int
    @JvmStatic external fun nativeSetHint(handle: Long, seq: Long, hint: Int): Int
    @JvmStatic external fun nativeReadWindowFrame(handle: Long, pos: Int, dst: ByteBuffer): Int

    @JvmStatic
    external fun nativeSubmitEstimate(
        handle: Long, pos: Int,
        roiX: Int, roiY: Int, roiW: Int, roiH: Int,
        y: ByteBuffer, uv: ByteBuffer, weight: ByteBuffer,
    ): Int

    @JvmStatic external fun nativeCompleteWindow(handle: Long): Int

    @JvmStatic
    external fun nativePopOutputYuv420(
        handle: Long,
        y: ByteBuffer, yOffset: Int, yRowStride: Int,
        u: ByteBuffer, uOffset: Int,
        v: ByteBuffer, vOffset: Int,
        uvRowStride: Int, uvPixelStride: Int,
        meta: ByteBuffer,
    ): Int

    @JvmStatic external fun nativePopOutputNv12(handle: Long, dst: ByteBuffer, meta: ByteBuffer): Int
    @JvmStatic external fun nativeStats(handle: Long, out: ByteBuffer): Int

    // ---- VC-Core motor arayüzü (Java_com_video_cleaner_Core_vc*) -----------------------------
    // Halka belleği Android'de AHardwareBuffer'dır (ayrılamazsa yığın). Geometri ilk karede belirlenir.

    const val ENGINE_MODE_AUTO = 0
    const val ENGINE_MODE_B = 1
    const val ENGINE_MODE_A = 2

    /**
     * VC-Core'u başlatır; bağlam tutamağı veya hata kodu (-4096..0). Tutamak bir adrestir ve ARM64'te
     * etiketli olduğundan NEGATİF görünebilir: geçerlilik `!in CleanerSession.ERROR_CODES` ile denetlenir.
     */
    @JvmStatic external fun vcInitEngine(mode: Int, preferVulkan: Boolean): Long

    /** YUV 4:2:0 donanım tamponunu (ImageReader `Image.hardwareBuffer`) Java yığınına kopyalamadan iter. */
    @JvmStatic external fun vcPushHardwareFrame(ctx: Long, buffer: HardwareBuffer, ptsUs: Long, flags: Int): Int

    /** Sıkı NV12 kare (CPU yolu). */
    @JvmStatic external fun vcPushNv12(ctx: Long, width: Int, height: Int, frame: ByteBuffer, ptsUs: Long, flags: Int): Int

    /** Sıradaki adım (STEP_*); `info` düzeni `nativePoll` ile aynı. */
    @JvmStatic external fun vcPoll(ctx: Long, info: ByteBuffer): Int

    /** Hazır pencereyi işler: 0 pencere yok, 1 Mod B, 2 Mod A. */
    @JvmStatic external fun vcProcessWindow(ctx: Long, forceModeA: Boolean): Int

    @JvmStatic external fun vcPopOutputNv12(ctx: Long, dst: ByteBuffer, meta: ByteBuffer): Int
    @JvmStatic external fun vcFinish(ctx: Long): Int

    /** VC-Tester çalışma zamanı raporu (JSON): mediaSync, preservation, scheduler, platform, runtime. */
    @JvmStatic external fun vcRunTesterAudit(ctx: Long): String

    @JvmStatic external fun vcDestroy(ctx: Long)

    /** Etkin SIMD arka ucunun adı. */
    fun kernelBackendName(): String = when (nativeKernelBackend()) {
        1 -> "x86_64 AVX2"
        2 -> "ARM64 NEON"
        else -> "taşınabilir (SIMD yok)"
    }
}

/** Rust çekirdeğinden dönen hata. `code`, `CoreError` sayısal değeridir. */
class CoreException(val code: Int, where: String) : RuntimeException("$where: ${describe(code)} ($code)") {
    companion object {
        fun describe(code: Int): String = when (code) {
            -1 -> "geçersiz kare boyutu"
            -2 -> "bellek ayrılamadı"
            -3 -> "halka dolu"
            -4 -> "yuva bir okuyucu tarafından tutuluyor"
            -5 -> "süren bir yazma var"
            -6 -> "zaman damgası artan sırada değil"
            -7 -> "kare halkada bulunamadı"
            -8 -> "çağrı bu durumda geçerli değil"
            -9 -> "geçersiz argüman"
            -10 -> "tampon çok küçük"
            -11 -> "akış sonlandı"
            -12 -> "geçersiz tutamak"
            -13 -> "çekirdekte iç hata"
            else -> "bilinmeyen hata"
        }
    }
}

// =============================================================================
// 2. Oturum
// =============================================================================

/** Modelin çalıştırılacağı 30 konumluk pencere (bkz. Rust `WindowPlan`). */
data class WindowPlan(
    val id: Long,
    /** true: Mod A (14+1+15, yalnızca konum 14 çözülür); false: Mod B (15+15). */
    val modeA: Boolean,
    val slide: Int,
    val validCount: Int,
    val emitStart: Int,
    val emitEnd: Int,
    val cacheStart: Int,
    val cacheEnd: Int,
    val flags: Int,
    /** Konum 0'ın sıra numarası; sahne başında negatif olabilir (dolgu). */
    val baseSeq: Long,
    /** Bit p = konum p gerçek bir kare içeriyor. */
    val validMask: Int,
) {
    fun isValid(position: Int): Boolean = validMask and (1 shl position) != 0

    fun wantsEstimate(position: Int): Boolean =
        position in emitStart until emitEnd || position in cacheStart until cacheEnd
}

sealed interface Step {
    data class NeedFrames(val count: Int) : Step
    data class Window(val plan: WindowPlan) : Step
    data class Output(val count: Int) : Step
    data object Finished : Step
}

data class OutputFrame(val seq: Long, val ptsUs: Long, val flags: Int, val modifiedPixels: Long)

data class CoreStats(
    val framesIn: Long = 0,
    val framesOut: Long = 0,
    val sceneCuts: Long = 0,
    val modifiedPixels: Long = 0,
    val preservationViolations: Long = 0,
    val orderViolations: Long = 0,
    val maxOccupancy: Long = 0,
    val windowsA: Long = 0,
    val windowsB: Long = 0,
    val bridges: Long = 0,
    val bypassed: Long = 0,
    val fused: Long = 0,
    val segments: Long = 0,
    val poolBytes: Long = 0,
)

/**
 * Bir video için Rust hattı. Tek iş parçacığından sürülmelidir.
 *
 * ```
 * while (true) when (val step = session.poll()) {
 *     is Step.NeedFrames -> /* en çok step.count kare it; kaynak bittiyse finish() */
 *     is Step.Window     -> /* çıkarımı çalıştır, submitEstimate(...), completeWindow() */
 *     is Step.Output     -> /* popInto(...) ile kareleri kodlayıcıya ver */
 *     Step.Finished      -> break
 * }
 * ```
 */
class CleanerSession(val width: Int, val height: Int) : AutoCloseable {
    private var handle: Long
    private val info = direct(Core.POLL_INFO_BYTES)
    private val meta = direct(Core.OUTPUT_META_BYTES)
    private val statsBuffer = direct(Core.STATS_BYTES)

    val frameBytes: Int get() = width * height * 3 / 2

    init {
        check(Core.nativeAbiVersion() == Core.ABI_VERSION) {
            "libvideo_cleaner.so sürümü uyumsuz (beklenen ABI ${Core.ABI_VERSION})"
        }
        val created = Core.nativeCreate(width, height)
        // Tutamak bir bellek adresidir. Android 11+ ARM64'te adreslerin üst baytı etiketlidir
        // (ör. 0xB4…), dolayısıyla Long olarak NEGATİF görünebilir: "<= 0" denetimi geçerli
        // tutamağı hata sanar. Hata kodları yalnızca küçük negatif sayılardır.
        if (created in ERROR_CODES) throw CoreException(created.toInt(), "nativeCreate(${width}x$height)")
        handle = created
    }

    private fun check(code: Int, where: String): Int {
        if (code < 0) throw CoreException(code, where)
        return code
    }

    fun setAutoSceneCut(enabled: Boolean) {
        check(Core.nativeSetOption(handle, Core.OPTION_AUTO_SCENE_CUT, if (enabled) 1 else 0), "setOption")
    }

    /** Her çıkan karede koruma kapısını yeniden denetler (yavaş; tanılama için). */
    fun setIntegrityCheck(enabled: Boolean) {
        check(Core.nativeSetOption(handle, Core.OPTION_INTEGRITY_CHECK, if (enabled) 1 else 0), "setOption")
    }

    fun poll(): Step = when (check(Core.nativePoll(handle, info), "poll")) {
        Core.STEP_NEED_FRAMES -> Step.NeedFrames(info.getInt(4))
        Core.STEP_OUTPUT -> Step.Output(info.getInt(4))
        Core.STEP_FINISHED -> Step.Finished
        else -> Step.Window(
            WindowPlan(
                id = info.getLong(8),
                modeA = info.getInt(16) == 0,
                slide = info.getInt(20),
                validCount = info.getInt(24),
                emitStart = info.getInt(28),
                emitEnd = info.getInt(32),
                cacheStart = info.getInt(36),
                cacheEnd = info.getInt(40),
                flags = info.getInt(44),
                baseSeq = info.getLong(48),
                validMask = info.getInt(56),
            ),
        )
    }

    /** Çözücüden gelen YUV 4:2:0 görüntüyü (kırpma dikdörtgeni dahil) halkaya kopyalar; sıra numarasını döndürür. */
    fun push(image: Image, ptsUs: Long): Long {
        val crop = image.cropRect
        val (y, u, v) = image.planes
        // 10 bit (P010): örnekler 16 bit küçük-endian, değer üst bitlerde → üst bayt (+1) alınır.
        val high = if (image.format == ImageFormat.YCBCR_P010) 1 else 0
        val yOffset = crop.top * y.rowStride + crop.left * y.pixelStride + high
        val uvOffset = (crop.top / 2) * u.rowStride + (crop.left / 2) * u.pixelStride + high
        val seq = Core.nativePushYuv420(
            handle,
            y.buffer, yOffset, y.rowStride, y.pixelStride,
            u.buffer, uvOffset,
            v.buffer, uvOffset,
            u.rowStride, u.pixelStride,
            ptsUs, 0,
        )
        if (seq < 0) throw CoreException(seq.toInt(), "push")
        return seq
    }

    fun finish() {
        check(Core.nativeFinish(handle), "finish")
    }

    fun setHint(seq: Long, hint: Int) {
        check(Core.nativeSetHint(handle, seq, hint), "setHint")
    }

    /** Pencere konumundaki kareyi NV12 olarak `dst` içine kopyalar; dolgu konumunda false döner. */
    fun readWindowFrame(position: Int, dst: ByteBuffer): Boolean =
        check(Core.nativeReadWindowFrame(handle, position, dst), "readWindowFrame") == 1

    /**
     * Bir pencere konumunun model çıktısını verir. ROI çift sınırlı olmalıdır.
     * `y`/`weight`: w×h bayt, `uv`: w×h/2 bayt (U0 V0 U1 V1 …). `weight` 0 = pikseli koru.
     */
    fun submitEstimate(position: Int, x: Int, y0: Int, w: Int, h: Int, y: ByteBuffer, uv: ByteBuffer, weight: ByteBuffer) {
        check(Core.nativeSubmitEstimate(handle, position, x, y0, w, h, y, uv, weight), "submitEstimate")
    }

    /** Pencereyi tamamlar. Tahmin verilmeyen konumlar orijinal hâliyle çıkar. */
    fun completeWindow() {
        check(Core.nativeCompleteWindow(handle), "completeWindow")
    }

    /** Sıradaki kesinleşmiş kareyi kodlayıcının giriş görüntüsüne yazar; hazır kare yoksa null. */
    fun popInto(image: Image): OutputFrame? {
        val (y, u, v) = image.planes
        val got = check(
            Core.nativePopOutputYuv420(
                handle,
                y.buffer, 0, y.rowStride,
                u.buffer, 0,
                v.buffer, 0,
                u.rowStride, u.pixelStride,
                meta,
            ),
            "popOutput",
        )
        if (got == 0) return null
        return OutputFrame(meta.getLong(0), meta.getLong(8), meta.getInt(16), meta.getLong(24))
    }

    fun stats(): CoreStats {
        check(Core.nativeStats(handle, statsBuffer), "stats")
        val v = LongArray(14) { statsBuffer.getLong(it * 8) }
        return CoreStats(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13])
    }

    override fun close() {
        if (handle != 0L) {
            Core.nativeDestroy(handle)
            handle = 0L
        }
    }

    companion object {
        /** nativeCreate / vcInitEngine'in hata olarak döndürebileceği aralık (0 ve CoreError kodları). */
        val ERROR_CODES = -4096L..0L

        private fun direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
    }
}

/**
 * Model çıkarımının bağlanacağı arayüz (Detector → Flow → Mamba; ileride Vulkan).
 * Uygulama, `plan.wantsEstimate(p)` olan her konum için `session.submitEstimate`
 * çağırır; çağırmadığı konumlar orijinal hâliyle çıkar.
 */
fun interface WindowRestorer {
    fun restore(session: CleanerSession, plan: WindowPlan)
}

// =============================================================================
// 3. Video işi: çöz → halka/zamanlayıcı → kodla → MP4
// =============================================================================
//
// Girdi: Android'in çözebildiği her kapsayıcı ve kodek (MP4, MKV, WebM, 3GP, TS…;
// H.264, HEVC, VP8, VP9, AV1, MPEG-4… — cihazın çözücülerine bağlıdır), her çözünürlük,
// kare hızı (değişken dahil) ve bit hızı. 720p veya belirli bir biçim ZORUNLU DEĞİLDİR.
//
// Bilinen sınırlar (kodda sessizce geçiştirilmez, sonuçta bildirilir):
//   * Çıktı her zaman MP4'tür (H.264 veya HEVC; açıkça istenirse AV1 — CodecNegotiator; 8 bit 4:2:0).
//   * Ayrı altyazı izi çıkarma (softSubtitleRemux) yeniden kodlamaz; VP8/VP9 video MP4'e kopyalanamaz.
//   * Tek sayılı genişlik/yükseklik 4:2:0'da kodlanamaz; son sütun/satır kırpılır.
//   * 10 bit / HDR kaynak ton eşlemesi yapılmadan 8 bite indirilir.
//   * Çözünürlük üst sınırı cihazın kodlayıcısına bağlıdır.

/** Arayüzde gösterilen kaynak bilgisi. */
data class VideoInfo(
    val width: Int = 0,
    val height: Int = 0,
    val rotation: Int = 0,
    val durationUs: Long = 0,
    val frameRate: Float = 0f,
    val videoMime: String = "",
    val audioMime: String? = null,
    val bitrate: Int = 0,
    /** Ayrı (yumuşak) altyazı izlerinin türleri; bunlar görüntüye gömülü değildir, yeniden kodlamadan atılabilir. */
    val subtitleTracks: List<String> = emptyList(),
) {
    companion object {
        /** Dosyanın akış bilgilerini okur; okunamazsa null. */
        fun probe(context: Context, uri: Uri): VideoInfo? {
            val extractor = MediaExtractor()
            return try {
                extractor.setDataSource(context, uri, null)
                val video = VideoJob.findTrack(extractor, "video/") ?: return null
                val format = extractor.getTrackFormat(video)
                val audio = VideoJob.findTrack(extractor, "audio/")?.let { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME) }
                VideoInfo(
                    width = format.intOr(MediaFormat.KEY_WIDTH, 0),
                    height = format.intOr(MediaFormat.KEY_HEIGHT, 0),
                    rotation = format.intOr(MediaFormat.KEY_ROTATION, 0),
                    durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L,
                    frameRate = format.getNumber(MediaFormat.KEY_FRAME_RATE)?.toFloat() ?: 0f,
                    videoMime = format.getString(MediaFormat.KEY_MIME) ?: "",
                    audioMime = audio,
                    bitrate = format.intOr(MediaFormat.KEY_BIT_RATE, 0),
                    subtitleTracks = (0 until extractor.trackCount)
                        .mapNotNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME) }
                        .filter(::isSubtitleMime),
                )
            } catch (e: Exception) {
                null
            } finally {
                extractor.release()
            }
        }
    }
}

private fun MediaFormat.intOr(key: String, fallback: Int): Int = if (containsKey(key)) getInteger(key) else fallback

/** Ayrı (yumuşak) altyazı veya kapalı başlık izi mi? Görüntüye gömülü değildir: kareler değiştirilmeden atılabilir. */
internal fun isSubtitleMime(mime: String?): Boolean {
    val m = mime?.lowercase(Locale.ROOT) ?: return false
    return m.startsWith("text/") || m in SUBTITLE_MIMES
}

private val SUBTITLE_MIMES = setOf(
    "application/x-subrip", "application/x-ssa", "application/x-ass", "application/ttml+xml", "application/x-quicktime-tx3g",
    "application/cea-608", "application/cea-708", "application/dvbsubs", "application/pgs", "application/vobsub", "application/x-media3-cues",
)

// =============================================================================
// 3a. Kodek anlaşması (CodecNegotiator)
// =============================================================================

/**
 * Çıktı kodeğini cihazın GERÇEK kodlayıcı yeteneklerine göre seçer: H.264, HEVC, AV1, APV ve HDR10.
 *
 * Varsayılan sıra değişmez (H.264 veya kullanıcı HEVC seçtiyse HEVC, sonra diğeri); AV1/APV yalnızca
 * açıkça istenirse ve MP4 çoklayıcı o türü yazabiliyorsa denenir. HDR10 yeteneği raporlanır; VC-Core
 * halkası 8 bit 4:2:0 olduğundan HDR kaynak bu sürümde 8 bit SDR etiketli kodlanır (ton eşlemesi yok) —
 * bu, sonuç notlarında açıkça yazılır.
 */
object CodecNegotiator {
    const val AVC = MediaFormat.MIMETYPE_VIDEO_AVC
    const val HEVC = MediaFormat.MIMETYPE_VIDEO_HEVC
    const val AV1 = MediaFormat.MIMETYPE_VIDEO_AV1

    /** MediaFormat.MIMETYPE_VIDEO_APV (API 36). */
    const val APV = "video/apv"
    val ALL = listOf(AVC, HEVC, AV1, APV)

    data class Support(
        val mime: String,
        val label: String,
        val encoders: List<String>,
        val hardware: Boolean,
        val tenBit: Boolean,
        val hdr10: Boolean,
        val muxable: Boolean,
        val maxWidth: Int,
        val maxHeight: Int,
    ) {
        /** Bu cihazda bu türde MP4 üretilebilir mi? */
        val usable: Boolean get() = encoders.isNotEmpty() && muxable

        fun describe(): String = buildString {
            append(label).append(": ")
            if (encoders.isEmpty()) {
                append("kodlayıcı yok")
                return@buildString
            }
            append(if (hardware) "donanım" else "yalnız yazılım")
            append(", en çok ${maxWidth}×$maxHeight")
            if (tenBit) append(", 10 bit")
            if (hdr10) append(", HDR10")
            if (!muxable) append(", MP4'e yazılamaz")
        }
    }

    fun label(mime: String): String = when (mime) {
        AVC -> "H.264"
        HEVC -> "HEVC"
        AV1 -> "AV1"
        APV -> "APV"
        else -> mime.substringAfter('/')
    }

    /**
     * MP4 çoklayıcının (MediaMuxer MPEG_4) bu türü yazabildiği sürüm. AV1: Android 14 (API 34).
     * APV: bu sürümde MP4 çoklayıcı desteği cihazda doğrulanmadığından seçilmez (yalnızca raporlanır).
     */
    fun muxable(mime: String): Boolean = when (mime) {
        AVC, HEVC -> true
        AV1 -> Build.VERSION.SDK_INT >= 34
        else -> false
    }

    /** Cihazdaki kodlayıcıların taranması (MediaCodecList, takma adlar hariç). */
    fun survey(): List<Support> {
        val encoders = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder && !it.isAlias }
        return ALL.map { mime ->
            var hardware = false
            var tenBit = false
            var hdr10 = false
            var maxW = 0
            var maxH = 0
            val names = ArrayList<String>()
            for (info in encoders) {
                val type = info.supportedTypes.firstOrNull { it.equals(mime, ignoreCase = true) } ?: continue
                val caps = runCatching { info.getCapabilitiesForType(type) }.getOrNull() ?: continue
                names += info.name
                if (info.isHardwareAccelerated) hardware = true
                caps.videoCapabilities?.let {
                    maxW = max(maxW, it.supportedWidths.upper)
                    maxH = max(maxH, it.supportedHeights.upper)
                }
                for (pl in caps.profileLevels) {
                    when (mime) {
                        HEVC -> {
                            if (pl.profile == MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10) tenBit = true
                            if (pl.profile == MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10HDR10 ||
                                pl.profile == MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10HDR10Plus
                            ) {
                                tenBit = true
                                hdr10 = true
                            }
                        }
                        AV1 -> {
                            if (pl.profile == MediaCodecInfo.CodecProfileLevel.AV1ProfileMain10) tenBit = true
                            if (pl.profile == MediaCodecInfo.CodecProfileLevel.AV1ProfileMain10HDR10 ||
                                pl.profile == MediaCodecInfo.CodecProfileLevel.AV1ProfileMain10HDR10Plus
                            ) {
                                tenBit = true
                                hdr10 = true
                            }
                        }
                        APV -> tenBit = true      // APV profilleri 10 bit ve üzeridir
                    }
                }
            }
            Support(mime, label(mime), names, hardware, tenBit, hdr10, muxable(mime), maxW, maxH)
        }
    }

    /** Kodlayıcı deneme sırası: açıkça istenen (yazılabiliyorsa) önce, ardından H.264/HEVC (tercihe göre). */
    fun order(options: JobOptions): List<String> {
        val base = if (options.preferHevc) listOf(HEVC, AVC) else listOf(AVC, HEVC)
        val explicit = options.codec?.takeIf { muxable(it) }
        return (listOfNotNull(explicit) + base).distinct()
    }

    /** Kaynak HDR aktarım eğrisi (PQ / HLG) taşıyor mu? */
    fun isHdr(format: MediaFormat?): Boolean {
        val transfer = format?.intOr(MediaFormat.KEY_COLOR_TRANSFER, 0) ?: 0
        return transfer == MediaFormat.COLOR_TRANSFER_ST2084 || transfer == MediaFormat.COLOR_TRANSFER_HLG
    }

    fun summary(): String = survey().joinToString(" · ") { it.describe() }
}

// =============================================================================
// 3b. Çıkarım köprüsü (Vulkan / LiteRT / ONNX) — durum raporu
// =============================================================================

/**
 * VC-Detector / VC-Flow / VC-Mamba'yı çalıştıracak arka ucun durumu.
 *
 * DURUM: Bu sürümde çıkarım arka ucu YOKTUR. Vulkan hesap gölgelendiricileri, LiteRT ve ONNX
 * kitaplıkları uygulamaya bağlı değildir ve eğitilmiş ağırlık dağıtılmaz. [restorer] bu yüzden
 * her zaman null döner; kareler değiştirilmeden işlenir. Bu nesne yalnızca cihazın neyi
 * desteklediğini ve neyin eksik olduğunu dürüstçe raporlar.
 */
object InferenceRuntime {
    /** Trainer.py --export çıktısı (models/ klasörüne konur). */
    val MODEL_FILES = listOf("video_detector.safetensors", "video_flow.safetensors", "video_mamba.safetensors")

    data class Report(
        /** Desteklenen en yüksek Vulkan sürümü ("1.3" gibi) veya "yok". */
        val vulkan: String,
        val vulkanCompute: Boolean,
        val liteRt: Boolean,
        val onnx: Boolean,
        val weightsFound: List<String>,
        val weightsMissing: List<String>,
        /** Gerçekte kullanılan onarıcı. */
        val active: String,
        val reason: String,
    )

    private fun hasClass(name: String): Boolean = runCatching { Class.forName(name, false, InferenceRuntime::class.java.classLoader) }.isSuccess

    fun probe(context: Context): Report {
        val pm = context.packageManager
        // FEATURE_VULKAN_HARDWARE_VERSION sürümü VK_MAKE_API_VERSION(0, büyük, küçük, 0) = büyük<<22 | küçük<<12.
        val vulkan = listOf(4, 3, 2, 1, 0).firstOrNull { minor ->
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, (1 shl 22) or (minor shl 12))
        }?.let { "1.$it" } ?: "yok"
        val compute = pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_COMPUTE, 0)
        val liteRt = hasClass("com.google.ai.edge.litert.Interpreter") || hasClass("org.tensorflow.lite.Interpreter")
        val onnx = hasClass("ai.onnxruntime.OrtEnvironment")
        val dir = java.io.File(context.filesDir, "models")
        val found = MODEL_FILES.filter { java.io.File(dir, it).isFile }
        return Report(
            vulkan, compute, liteRt, onnx, found, MODEL_FILES - found.toSet(),
            active = "kimlik (kareler değiştirilmez)",
            reason = buildString {
                append("çıkarım arka ucu bu sürümde yok")
                if (found.size < MODEL_FILES.size) append("; eğitilmiş ağırlık yok (${MODEL_FILES.size - found.size}/${MODEL_FILES.size} eksik)")
            },
        )
    }

    /** Pencere onarıcısı. Arka uç ve ağırlıklar olmadığından null: kareler değiştirilmeden işlenir. */
    @Suppress("UNUSED_PARAMETER")
    fun restorer(context: Context): WindowRestorer? = null

    fun describe(report: Report): String =
        "Vulkan ${report.vulkan}${if (report.vulkanCompute) " (hesap)" else ""} · LiteRT ${if (report.liteRt) "var" else "yok"} · " +
            "ONNX ${if (report.onnx) "var" else "yok"} · ağırlık ${report.weightsFound.size}/${MODEL_FILES.size} · " +
            "etkin: ${report.active} — ${report.reason}"
}

// =============================================================================
// 3c. Cihaz tanılaması (VC-Tester çalışma zamanı denetimi)
// =============================================================================

object Diagnostics {
    /**
     * Küçük bir VC-Core motorunu sentetik karelerle çalıştırır ve VC-Tester raporunu (JSON) döndürür:
     * halka belleği (AHardwareBuffer / yığın), sayfa boyutu, .so LOAD hizası, kare/PTS bütünlüğü.
     */
    fun runtimeAudit(): String {
        val ctx = Core.vcInitEngine(Core.ENGINE_MODE_AUTO, false)
        if (ctx in CleanerSession.ERROR_CODES) return "{\"error\":$ctx}"
        try {
            val (w, h) = 64 to 32
            val frame = ByteBuffer.allocateDirect(w * h * 3 / 2).order(ByteOrder.nativeOrder())
            val info = ByteBuffer.allocateDirect(Core.POLL_INFO_BYTES).order(ByteOrder.nativeOrder())
            val out = ByteBuffer.allocateDirect(w * h * 3 / 2).order(ByteOrder.nativeOrder())
            val meta = ByteBuffer.allocateDirect(Core.OUTPUT_META_BYTES).order(ByteOrder.nativeOrder())
            fun drain(): Int {
                while (true) {
                    when (val step = Core.vcPoll(ctx, info)) {
                        Core.STEP_WINDOW -> Core.vcProcessWindow(ctx, false)
                        Core.STEP_OUTPUT -> while (Core.vcPopOutputNv12(ctx, out, meta) == 1) out.clear()
                        else -> return step
                    }
                }
            }
            for (i in 0 until 8) {
                for (b in 0 until frame.capacity()) frame.put(b, ((b + i * 7) and 0xff).toByte())
                if (Core.vcPushNv12(ctx, w, h, frame, i * 33_333L, 0) == -3) {
                    drain()
                    Core.vcPushNv12(ctx, w, h, frame, i * 33_333L, 0)
                }
            }
            Core.vcFinish(ctx)
            var guard = 0
            while (drain() != Core.STEP_FINISHED && guard++ < 100) Unit
            return Core.vcRunTesterAudit(ctx)
        } finally {
            Core.vcDestroy(ctx)
        }
    }

    /** Arayüz ve logcat (VCDIAG) için satırlar. */
    fun collect(context: Context): List<Pair<String, String>> {
        val rows = ArrayList<Pair<String, String>>()
        rows += "Cihaz" to "${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"
        rows += "Çekirdek" to runCatching { "ABI ${Core.nativeAbiVersion()} · ${Core.kernelBackendName()} · sayfa ${Core.nativePageSize() / 1024} KB" }
            .getOrElse { "yüklenemedi: ${it.message}" }
        rows += "VC-Tester" to runCatching { summarizeAudit(runtimeAudit()) }.getOrElse { "çalışmadı: ${it.message}" }
        rows += "Kodlayıcılar" to runCatching { CodecNegotiator.summary() }.getOrElse { "taranamadı: ${it.message}" }
        rows += "Çıkarım" to runCatching { InferenceRuntime.describe(InferenceRuntime.probe(context)) }.getOrElse { "taranamadı: ${it.message}" }
        for ((k, v) in rows) Log.i("VCDIAG", "$k: $v")
        return rows
    }

    private fun field(json: String, key: String): String? =
        Regex("\"$key\":(\"[^\"]*\"|[^,}]+)").find(json)?.groupValues?.get(1)?.trim('"')

    /** Tester JSON'unun okunur özeti. */
    fun summarizeAudit(json: String): String {
        val ok = field(json, "frames_equal") == "true" && field(json, "bit_exact") == "true" && field(json, "violations") == "0"
        return "bütünlük ${if (ok) "GEÇTİ" else "KALDI"} · halka ${field(json, "storage") ?: "?"} · " +
            ".so hizası ${field(json, "elf_load_align") ?: "?"} B (16 KB ${if (field(json, "elf_16k_ok") == "true") "uygun" else "UYGUN DEĞİL"}) · " +
            "model: ${field(json, "models") ?: "?"}"
    }
}

/**
 * Video bit hızı ve anahtar kare politikası (saf hesap; Android'e bağımlı değildir, birim testlenir).
 *
 * Sabitler emülatör kalibrasyonuyla seçildi (.github/workflows/bitrate-calibration.yml; 14 klip: 360p–1080p ve
 * dikey, 24/30/60 FPS, düşük/yüksek bit hızı, gerçek/gürültülü/animasyon/durağan, H.264/HEVC/VP9 kaynak).
 * Hedefe uyan kodlayıcıda (OMX.google.h264), kaynağın anahtar kare aralığıyla, VMAF kaynağa göre:
 *   kaynak×1.5 → ort. 93,8, en düşük 86,1 (2 klip < 90), medyan boyut 1,50×
 *   kaynak×2.0 → ort. 95,8, en düşük 91,6 (0 klip < 90), medyan boyut 1,96×, en kötü 3,2×
 *   kaynak×3.0 → ort. 97,4, en düşük 93,5
 *   eski kural (max(beyan×1.25, 0.14 bpppf)) → ort. 96,6, en düşük 92,9; medyan 2,5×, ort. 5,8×, en kötü 33,7×
 * Piksel başına bite göre uyarlanan kurallar zorluğu öngörmedi (düşük bit hızlı kolay klipler); sabit kat seçildi.
 * Üst sınır: ölçülen 11 noktada çıktı ≥ 0,20 bpppf iken VMAF ≥ 96,5; bu yüzden hedef, kaynağın katı bu eşiği
 * aşıyorsa eşiğe (ama asla kaynağın kendi bit hızının altına değil) indirilir. Böylece 16 Mb/s'lik bir 1080p
 * kaynak 32 Mb/s'ye şişmez; kalibrasyon setinde ort. VMAF 95,8 → 95,4, en düşük 91,6 aynı, ort. boyut 2,11× → 1,95×.
 * Kaynağın kendi anahtar kare aralığını korumak aynı boyutta VMAF'ı 7 puana kadar artırdı.
 * Donanım kodlayıcıları farklı davranır; hedef kodlayıcının bit hızı aralığına kırpılır, kalite modu
 * (CQ/QP sınırı) kullanılmaz: AVC'de CQ yok, QP sınırı ölçülebilir fark yaratmadı.
 */
internal object BitratePolicy {
    /** Standart: kaynağın gerçek video bit hızının katı (hiçbir test klibinde VMAF < 90 değil). */
    const val STANDARD_SOURCE_FACTOR = 2.0
    /** Yüksek: kaynağın gerçek video bit hızının katı (ortalama VMAF eski kuralın üstünde). */
    const val HIGH_SOURCE_FACTOR = 3.0
    /** Kaynak katının yetiştiği "yeterli" çıktı yoğunluğu (piksel başına, kare başına bit); üstüne çıkılmaz. */
    const val STANDARD_CAP_BPPPF = 0.20
    const val HIGH_CAP_BPPPF = 0.30
    /** Kaynak bit hızı ölçülemezse eski taban (piksel başına, kare başına bit) korunur: belirsizlikte kaliteyi düşürme. */
    const val UNKNOWN_SOURCE_BPPPF = 0.14
    /** Akıl dışı küçük hedeflere karşı alt sınır (bit/s). */
    const val MIN_BITRATE = 64_000L
    /** Anahtar kare aralığı: kaynağınki korunur, bu sınırlar içinde (saniye). */
    const val MIN_GOP_SEC = 1f
    const val MAX_GOP_SEC = 10f

    data class Input(
        val width: Int,
        val height: Int,
        val frameRate: Double,
        /** Örnek tablosundan ölçülen kaynak video bit hızı; ölçülemediyse 0. */
        val sourceVideoBps: Long,
        /** Kaynağın ortalama anahtar kare aralığı (µs); bilinmiyorsa 0. */
        val sourceGopUs: Long,
        val sourceMime: String,
        val targetMime: String,
        /** JobOptions.quality: 1.0 = Standart, ≥ 2.0 = Yüksek. */
        val quality: Float,
    )

    data class Plan(val bitrateBps: Long, val iFrameIntervalSec: Float, val reason: String)

    /**
     * Kaynak ile hedef kodek farklıysa (ör. HEVC → H.264) aynı kalite için ek bit hızı oranı.
     * Ölçüm (c12/c13 HEVC, c14 VP9 → H.264): aynı katsayı H.264 kaynaklarla benzer veya daha iyi VMAF verdi
     * (×1.5: 93,5 / 93,3 / 95,5; ×2.0: 96,2 / 93,3 / 97,5); ek katsayı gerekmedi. HEVC hedef kodlayıcı
     * emülatörde ≤512 px olduğundan H.264 → HEVC tasarrufu ölçülemedi; ölçülmeden tasarruf varsayılmaz.
     */
    fun codecFactor(sourceMime: String, targetMime: String): Double = 1.0

    fun plan(input: Input): Plan {
        val gop = if (input.sourceGopUs > 0) (input.sourceGopUs / 1e6).toFloat().coerceIn(MIN_GOP_SEC, MAX_GOP_SEC) else MIN_GOP_SEC
        val high = input.quality >= 2f
        if (input.sourceVideoBps <= 0) {
            val fallback = (input.width.toDouble() * input.height * input.frameRate * UNKNOWN_SOURCE_BPPPF * input.quality).toLong()
            return Plan(max(fallback, MIN_BITRATE), gop, "kaynak bit hızı ölçülemedi: ${UNKNOWN_SOURCE_BPPPF} bpppf×${input.quality}")
        }
        val factor = if (high) HIGH_SOURCE_FACTOR else STANDARD_SOURCE_FACTOR
        val capBpppf = if (high) HIGH_CAP_BPPPF else STANDARD_CAP_BPPPF
        val codec = codecFactor(input.sourceMime, input.targetMime)
        val source = input.sourceVideoBps * codec
        val cap = input.width.toDouble() * input.height * input.frameRate * capBpppf
        // min(kaynak×kat, max(kaynak×1, eşik)): düşük bit hızlı kaynakta kat uygulanır, yüksek bit hızlıda eşikte
        // durulur; kaynağın kendi bit hızının altına hiçbir zaman inilmez.
        val target = minOf(source * factor, maxOf(source, cap)).toLong()
        val limited = source * factor > maxOf(source, cap)
        val reason = "kaynak ${input.sourceVideoBps / 1000} kb/s × ${if (limited) "%.2f".format(target / input.sourceVideoBps.toDouble()) + " (eşik $capBpppf bpppf)" else "$factor"} × kodek $codec"
        return Plan(max(target, MIN_BITRATE), gop, reason)
    }
}

/** Kullanıcının seçtiği çıktı ayarları. */
data class JobOptions(
    /** true: HEVC (daha küçük dosya), false: H.264 (en geniş uyumluluk). Cihaz desteklemiyorsa diğeri kullanılır. */
    val preferHevc: Boolean = false,
    /** Bit hızı çarpanı: 1.0 = kaynağa yakın, 2.0 = yüksek kalite. */
    val quality: Float = 1.0f,
    /** Yalnızca ölçüm/kalibrasyon testleri için: bit hızı politikasını atlayıp kodlayıcıyı doğrudan ayarlar. */
    val tuning: EncoderTuning? = null,
    /** Açıkça istenen çıktı türü (CodecNegotiator.AV1 gibi); null = [preferHevc]'ye göre H.264/HEVC. */
    val codec: String? = null,
    /**
     * true: yalnızca AYRI altyazı izleri atılır; video paketleri yeniden kodlanmadan, bayt bayt kopyalanır
     * (0 yapay zekâ, kalite kaybı yok). Görüntüye gömülü yazıya dokunmaz.
     */
    val softSubtitleRemux: Boolean = false,
    /**
     * true: çözücü çıktısı Surface → ImageReader (HardwareBuffer) yolundan alınır. Bu yol ilk karede
     * başarısız olursa iş, bayt tamponu yoluyla baştan yapılır ve bu bir notla bildirilir.
     */
    val hardwareBufferDecode: Boolean = false,
)

/**
 * Kodlayıcı ayarlarının elle verilmesi (kalibrasyon testleri). Boş alanlar politikaya bırakılır.
 * [bitrateMode] MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_* değeridir; [cqQuality] yalnızca CQ'da kullanılır.
 */
data class EncoderTuning(
    val bitrateBps: Int? = null,
    val bitrateMode: Int? = null,
    val cqQuality: Int? = null,
    val iFrameIntervalSec: Float? = null,
    /** QP üst sınırı; yalnızca kodlayıcı "qp-bounds" özelliğini destekliyorsa uygulanır (API 31+). */
    val qpMax: Int? = null,
)

/** Kaynak video akışının örnek tablosundan ölçülen gerçek değerleri (kapsayıcının beyan ettiği değil). */
data class SourceVideoStats(
    /** Yalnızca video örneklerinin toplam baytı. */
    val videoBytes: Long,
    val samples: Int,
    val syncSamples: Int,
    /** İlk örnekten son örneğin bitişine kadar süre (µs). */
    val durationUs: Long,
) {
    /** Video akışının ortalama bit hızı; ölçülemediyse 0. */
    val videoBitsPerSecond: Long get() = if (durationUs > 0) videoBytes * 8_000_000L / durationUs else 0L

    /** Ortalama anahtar kare aralığı (µs); tek anahtar kare varsa süre. */
    val averageGopUs: Long get() = if (syncSamples > 0) durationUs / syncSamples else 0L
}

data class JobProgress(
    val framesDecoded: Long = 0,
    val framesEncoded: Long = 0,
    val positionUs: Long = 0,
    val durationUs: Long = 0,
    val framesPerSecond: Double = 0.0,
    val elapsedMs: Long = 0,
    val width: Int = 0,
    val height: Int = 0,
    val stats: CoreStats = CoreStats(),
) {
    val fraction: Float
        get() = if (durationUs > 0) (positionUs.toDouble() / durationUs).coerceIn(0.0, 1.0).toFloat() else 0f

    /** Kalan süre tahmini (ms); hesaplanamıyorsa -1. */
    val remainingMs: Long
        get() = if (fraction > 0.02f) (elapsedMs * (1f - fraction) / fraction).toLong() else -1L
}

data class JobResult(
    val framesDecoded: Long,
    val framesEncoded: Long,
    val elapsedMs: Long,
    val width: Int,
    val height: Int,
    /** Çekirdekten çıkıp kodlayıcıya verilen kare sayısı. */
    val framesQueued: Long,
    /** Sesin akıbeti, kullanıcıya gösterilecek biçimde. */
    val audio: String,
    val outputMime: String,
    val encoderName: String,
    val restorerAttached: Boolean,
    /** Kullanıcının bilmesi gereken dönüşümler (kırpma, 10 bit → 8 bit…). */
    val notes: List<String>,
    val stats: CoreStats,
    /** Kodlayıcıya verilen ayarlar (tanılama ve kalibrasyon için). */
    val encoderSettings: EncoderSettings = EncoderSettings(),
    /** Kaynak video akışının ölçülen değerleri; ölçülemediyse null. */
    val sourceVideo: SourceVideoStats? = null,
    /** Kapsayıcının video izi için beyan ettiği bit hızı (MediaFormat.KEY_BIT_RATE); yoksa 0. */
    val sourceDeclaredBitrate: Int = 0,
    val sourceFrameRate: Float = 0f,
    /** true: video yeniden kodlanmadı, paketler aynen kopyalandı (ayrı altyazı izi çıkarma). */
    val remuxed: Boolean = false,
    /** Çıktıdan atılan ayrı altyazı izlerinin türleri. */
    val removedSubtitleTracks: List<String> = emptyList(),
) {
    /** Kare kaybı ve sıra hatası yok mu? (çözülen = çekirdekten çıkan = kodlayıcıya verilen = kodlanan) */
    val intact: Boolean
        get() = if (remuxed) {
            framesDecoded == framesEncoded && framesQueued == framesDecoded
        } else {
            framesDecoded == framesEncoded && framesQueued == framesDecoded &&
                stats.framesIn == framesDecoded && stats.framesOut == framesDecoded &&
                stats.orderViolations == 0L && stats.preservationViolations == 0L
        }

    val framesPerSecond: Double get() = if (elapsedMs > 0) framesEncoded * 1000.0 / elapsedMs else 0.0
}

/** Kodlayıcıya gerçekten verilen ayarlar. */
data class EncoderSettings(
    val bitrateBps: Int = 0,
    /** "VBR", "CBR", "CQ" veya "varsayılan" (mod verilmedi; kodlayıcı kendisi seçer). */
    val bitrateMode: String = "",
    val cqQuality: Int = -1,
    val qpMin: Int = -1,
    val qpMax: Int = -1,
    val iFrameIntervalSec: Float = 0f,
    val frameRate: Float = 0f,
    /** Politikanın gerekçesi (ör. "kaynak×1.00", "taban 0.06 bpppf"). */
    val reason: String = "",
)

class JobException(message: String, cause: Throwable? = null) : Exception(message, cause)
class JobCancelledException : Exception("İptal edildi")

// ---- ses ---------------------------------------------------------------------

/** Ses akışını çoklayıcıya taşıyan yol: aynen kopyalama veya AAC'ye dönüştürme. */
private interface AudioPath {
    val trackFormat: MediaFormat
    val label: String
    val finished: Boolean
    fun pump(muxer: MediaMuxer, track: Int, untilUs: Long)
    fun release()
}

/**
 * Çoklayıcıya gidecek ses zaman damgası: negatif değerler (ön dolgu) 0'a çekilir ve damgalar
 * kesin artan tutulur. MPEG4Writer negatif veya geri giden ses damgasında izi sessizce boş bırakır.
 */
internal fun monotonicAudioPts(timeUs: Long, lastPtsUs: Long): Long {
    val clamped = max(timeUs, 0L)
    return if (lastPtsUs == Long.MIN_VALUE) clamped else max(clamped, lastPtsUs + 1)
}

/** MP4'ün doğrudan taşıyabildiği ses (AAC, AMR): örnekler yeniden kodlanmadan kopyalanır. */
/**
 * [offsetUs]: zaman çizelgesi kaydırması. MP4'te AAC/Opus ön dolgu paketleri negatif damgalıdır (ör. −21 333 µs) ve
 * MP4 çoklayıcı negatif damga yazamaz. Paketleri değiştirmeden veya atmadan senkronu korumak için bütün çıktı
 * (ses ve görüntü) aynı miktarda ileri kaydırılır; böylece ses paketleri birebir korunur ve A/V ofseti kaynakla aynı kalır.
 */
private class AudioCopy(private val extractor: MediaExtractor, override val trackFormat: MediaFormat, private val offsetUs: Long) : AudioPath {
    override val label = "yeniden kodlanmadan kopyalandı"
    override var finished = false
        private set
    private val info = MediaCodec.BufferInfo()
    private val buffer: ByteBuffer = ByteBuffer.allocateDirect(max(trackFormat.intOr(MediaFormat.KEY_MAX_INPUT_SIZE, 0), 1 shl 20))
    private var lastPtsUs = Long.MIN_VALUE

    override fun pump(muxer: MediaMuxer, track: Int, untilUs: Long) {
        while (!finished) {
            // Akış sonu sampleTrackIndex ile anlaşılır: MP4'te AAC/MP3/Opus ön dolgu (priming) paketleri
            // edit list yüzünden NEGATİF zaman damgası taşır (ör. −21 333 µs); sampleTime < 0 akış sonu değildir.
            if (extractor.sampleTrackIndex < 0) {
                finished = true
                break
            }
            val time = extractor.sampleTime + offsetUs
            if (time > untilUs) break
            val size = extractor.readSampleData(buffer, 0)
            if (size < 0) {
                finished = true
                break
            }
            val key = if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
            val pts = monotonicAudioPts(time, lastPtsUs)
            lastPtsUs = pts
            buffer.position(0)
            buffer.limit(size)
            info.set(0, size, pts, key)
            muxer.writeSampleData(track, buffer, info)
            extractor.advance()
        }
    }

    override fun release() = Unit
}

/**
 * MP4'e doğrudan yazılamayan ses (Opus, Vorbis, MP3, FLAC, AC-3…): çözülür ve AAC-LC'ye kodlanır.
 * Çoklayıcı, AAC kodlayıcının çıkış biçimi bilinmeden başlatılamadığı için [prepare] ilk
 * paketleri üretip biçimi öğrenir; bu arada çıkan paketler kuyrukta bekler.
 */
private class AudioTranscode(private val extractor: MediaExtractor, source: MediaFormat, private val offsetUs: Long) : AudioPath {
    override val label = "AAC'ye dönüştürüldü (kaynak: ${source.getString(MediaFormat.KEY_MIME)})"
    override var finished = false
        private set
    override lateinit var trackFormat: MediaFormat
        private set

    private val decoder: MediaCodec
    private var encoder: MediaCodec? = null
    private val info = MediaCodec.BufferInfo()
    private val queued = ArrayDeque<Pair<ByteBuffer, MediaCodec.BufferInfo>>()
    private var inputDone = false
    private var decoderDone = false
    private var eosQueued = false
    private var pcm: ByteBuffer? = null
    private var pcmIndex = -1
    private var pcmPtsUs = 0L
    private var bytesPerSecond = 1L
    private var lastPtsUs = Long.MIN_VALUE

    init {
        source.setInteger(MediaFormat.KEY_PCM_ENCODING, android.media.AudioFormat.ENCODING_PCM_16BIT)
        decoder = MediaCodec.createDecoderByType(source.getString(MediaFormat.KEY_MIME) ?: throw JobException("Ses türü okunamadı"))
        decoder.configure(source, null, null, 0)
        decoder.start()
    }

    /** Kodlayıcının çıkış biçimi öğrenilene kadar çalıştırır. */
    fun prepare() {
        var guard = 0
        while (!::trackFormat.isInitialized && !finished) {
            step(null, -1)
            if (++guard > 20_000) throw JobException("Ses kodlayıcı çıkış biçimi alınamadı")
        }
        if (!::trackFormat.isInitialized) throw JobException("Ses akışı boş")
    }

    override fun pump(muxer: MediaMuxer, track: Int, untilUs: Long) {
        while (queued.isNotEmpty()) {
            val (data, meta) = queued.removeFirst()
            muxer.writeSampleData(track, data, meta)
        }
        var guard = 0
        while (!finished && lastPtsUs <= untilUs && ++guard < 100_000) step(muxer, track)
    }

    private fun startEncoder(pcmFormat: MediaFormat) {
        val rate = pcmFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = pcmFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        if (pcmFormat.intOr(MediaFormat.KEY_PCM_ENCODING, android.media.AudioFormat.ENCODING_PCM_16BIT) != android.media.AudioFormat.ENCODING_PCM_16BIT) {
            throw JobException("16 bit olmayan PCM çıkışı desteklenmiyor")
        }
        bytesPerSecond = rate.toLong() * channels * 2
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, rate, channels)
        format.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
        format.setInteger(MediaFormat.KEY_BIT_RATE, (64_000 * channels).coerceIn(64_000, 320_000))
        val enc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        enc.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        enc.start()
        encoder = enc
    }

    private fun step(muxer: MediaMuxer?, track: Int) {
        // 1) kaynak → çözücü
        if (!inputDone) {
            val index = decoder.dequeueInputBuffer(0)
            if (index >= 0) {
                val buffer = decoder.getInputBuffer(index) ?: throw JobException("Ses çözücü girişi alınamadı")
                val size = extractor.readSampleData(buffer, 0)
                if (size < 0) {
                    decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    inputDone = true
                } else {
                    // Zaman çizelgesi kaydırması (bkz. AudioCopy) sonrası damga ≥ 0 olur; yine de negatif kalırsa 0'a çekilir.
                    decoder.queueInputBuffer(index, 0, size, max(extractor.sampleTime + offsetUs, 0L), 0)
                    extractor.advance()
                }
            }
        }
        // 2) çözücü → PCM
        if (pcmIndex < 0 && !decoderDone) {
            val index = decoder.dequeueOutputBuffer(info, 2_000)
            if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (encoder == null) startEncoder(decoder.outputFormat)
            } else if (index >= 0) {
                if (encoder == null) startEncoder(decoder.outputFormat)
                val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                if (info.size > 0) {
                    val data = decoder.getOutputBuffer(index) ?: throw JobException("Ses çözücü çıkışı alınamadı")
                    data.position(info.offset)
                    data.limit(info.offset + info.size)
                    pcm = data
                    pcmIndex = index
                    pcmPtsUs = info.presentationTimeUs
                } else {
                    decoder.releaseOutputBuffer(index, false)
                }
                if (end) decoderDone = true
            }
        }
        // 3) PCM → AAC kodlayıcı (kodlayıcının giriş tamponu kadar parça parça)
        val enc = encoder
        if (enc != null) {
            val data = pcm
            if (data != null && pcmIndex >= 0) {
                val index = enc.dequeueInputBuffer(0)
                if (index >= 0) {
                    val target = enc.getInputBuffer(index) ?: throw JobException("Ses kodlayıcı girişi alınamadı")
                    val count = minOf(target.remaining(), data.remaining()) and 1.inv()
                    val slice = data.duplicate()
                    slice.limit(slice.position() + count)
                    target.put(slice)
                    data.position(data.position() + count)
                    enc.queueInputBuffer(index, 0, count, pcmPtsUs, 0)
                    pcmPtsUs += count * 1_000_000L / bytesPerSecond
                    if (data.remaining() < 2) {
                        decoder.releaseOutputBuffer(pcmIndex, false)
                        pcmIndex = -1
                        pcm = null
                    }
                }
            } else if (decoderDone && !eosQueued) {
                val index = enc.dequeueInputBuffer(0)
                if (index >= 0) {
                    enc.queueInputBuffer(index, 0, 0, pcmPtsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    eosQueued = true
                }
            }
            // 4) AAC → çoklayıcı (veya çoklayıcı başlayana kadar kuyruk)
            val index = enc.dequeueOutputBuffer(info, if (eosQueued) 2_000 else 0)
            if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                trackFormat = enc.outputFormat
            } else if (index >= 0) {
                val data = enc.getOutputBuffer(index) ?: throw JobException("Ses kodlayıcı çıkışı alınamadı")
                if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0) {
                    // Çoklayıcı kesin artan olmayan ses damgalarını reddeder.
                    val pts = monotonicAudioPts(info.presentationTimeUs, lastPtsUs)
                    val meta = MediaCodec.BufferInfo().apply { set(0, info.size, pts, info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) }
                    lastPtsUs = pts
                    data.position(info.offset)
                    data.limit(info.offset + info.size)
                    if (muxer != null) {
                        meta.offset = info.offset
                        muxer.writeSampleData(track, data, meta)
                    } else {
                        val copy = ByteBuffer.allocateDirect(info.size)
                        copy.put(data)
                        copy.flip()
                        queued.addLast(copy to meta)
                    }
                }
                val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                enc.releaseOutputBuffer(index, false)
                if (end) finished = true
            }
        } else if (decoderDone) {
            finished = true      // çözücü hiç ses üretmedi
        }
    }

    override fun release() {
        runCatching { decoder.stop() }
        runCatching { decoder.release() }
        runCatching { encoder?.stop() }
        runCatching { encoder?.release() }
    }
}

// ---- video -------------------------------------------------------------------

class VideoJob(
    private val context: Context,
    private val input: Uri,
    private val output: Uri,
    private val options: JobOptions = JobOptions(),
    private val restorer: WindowRestorer? = null,
    private val onProgress: (JobProgress) -> Unit = {},
) {
    @Volatile
    private var cancelled = false

    fun cancel() {
        cancelled = true
    }

    /**
     * Daha tek kare vermeden çöken video çözücü. [run] bu durumda işi aynı türdeki bir sonraki
     * çözücüyle baştan dener (ör. bazı cihaz/emülatör donanım çözücüleri CSD aşamasında hata verir).
     */
    private class DecoderFailedException(val decoderName: String, cause: Exception) : Exception(cause.message, cause)

    /** Surface → ImageReader yolu daha tek kare vermeden başarısız oldu. */
    private class HardwarePathFailedException(message: String, cause: Throwable? = null) : Exception(message, cause)

    /** İşi çağıran iş parçacığında baştan sona çalıştırır. Hata veya iptalde yarım çıktı silinir. */
    fun run(): JobResult {
        val skipped = LinkedHashSet<String>()
        var hardwarePath = options.hardwareBufferDecode
        val carried = ArrayList<String>()
        try {
            if (options.softSubtitleRemux) return remux()
            while (true) {
                try {
                    val result = transcode(skipped, hardwarePath)
                    return if (carried.isEmpty()) result else result.copy(notes = carried + result.notes)
                } catch (e: DecoderFailedException) {
                    deleteOutput()
                    Log.w(CleanerService.TAG, "Çözücü ${e.decoderName} kare vermeden hata verdi; sıradaki çözücü deneniyor", e.cause)
                    if (!skipped.add(e.decoderName) || skipped.size > MAX_DECODER_ATTEMPTS) throw e.cause ?: e
                } catch (e: HardwarePathFailedException) {
                    deleteOutput()
                    Log.w(CleanerService.TAG, "HardwareBuffer çözme yolu ilk karede başarısız; bayt tamponu yoluna geçildi", e.cause)
                    hardwarePath = false
                    carried += "HardwareBuffer çözme yolu bu cihazda çalışmadı (${e.message}); bayt tamponu yolu kullanıldı."
                }
            }
        } catch (e: Exception) {
            deleteOutput()
            throw when (e) {
                is JobCancelledException, is JobException -> e
                is CoreException -> JobException("Çekirdek hatası: ${e.message}", e)
                is MediaCodec.CodecException -> JobException("Kodek hatası: ${e.diagnosticInfo}", e)
                else -> JobException(e.message ?: e.javaClass.simpleName, e)
            }
        }
    }

    private fun deleteOutput() {
        runCatching {
            if (output.scheme == "file") output.path?.let { java.io.File(it).delete() } else DocumentsContract.deleteDocument(context.contentResolver, output)
        }
    }

    private class Encoder(val codec: MediaCodec, val mime: String, val name: String, val settings: EncoderSettings)

    /** Verilen boyutu destekleyen bir kodlayıcı seçip yapılandırır; tercih edilen tür olmazsa diğerini dener. */
    private fun startEncoder(
        width: Int, height: Int, frameRate: Float, source: SourceVideoStats?, sourceMime: String, color: MediaFormat?, sar: Pair<Int, Int>?,
    ): Encoder {
        val order = CodecNegotiator.order(options)
        val codecs = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder }
        var lastError: Exception? = null
        for (mime in order) {
            for (candidate in codecs) {
                val type = candidate.supportedTypes.firstOrNull { it.equals(mime, ignoreCase = true) } ?: continue
                val caps = runCatching { candidate.getCapabilitiesForType(type) }.getOrNull() ?: continue
                val video = caps.videoCapabilities ?: continue
                if (!video.isSizeSupported(width, height)) continue
                if (MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible !in caps.colorFormats) continue
                val format = MediaFormat.createVideoFormat(mime, width, height)
                format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                val tuning = options.tuning
                val encoderCaps = caps.encoderCapabilities
                val plan = BitratePolicy.plan(
                    BitratePolicy.Input(
                        width, height, frameRate.toDouble(), source?.videoBitsPerSecond ?: 0L, source?.averageGopUs ?: 0L,
                        sourceMime, mime, options.quality,
                    ),
                )
                val wanted = tuning?.bitrateBps?.toLong() ?: plan.bitrateBps
                val bitrate = wanted.coerceIn(video.bitrateRange.lower.toLong(), video.bitrateRange.upper.toLong()).toInt()
                format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                var modeLabel = "varsayılan"
                var cq = -1
                // Politika VBR ister (hedef ortalama bit hızı; kare başına dalgalanmaya izin verir); desteklenmiyorsa
                // kodlayıcının varsayılanı kalır. Kalibrasyon kendi modunu verebilir.
                val vbr = MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR
                val mode = if (tuning != null) tuning.bitrateMode else vbr.takeIf { encoderCaps?.isBitrateModeSupported(it) == true }
                if (mode != null) {
                    if (encoderCaps == null || !encoderCaps.isBitrateModeSupported(mode)) continue     // bu kodlayıcı istenen modu desteklemiyor
                    format.setInteger(MediaFormat.KEY_BITRATE_MODE, mode)
                    modeLabel = bitrateModeName(mode)
                    if (mode == MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CQ) {
                        cq = (tuning?.cqQuality ?: encoderCaps.qualityRange.upper).coerceIn(encoderCaps.qualityRange.lower, encoderCaps.qualityRange.upper)
                        format.setInteger(MediaFormat.KEY_QUALITY, cq)
                    }
                }
                val rates = runCatching { video.getSupportedFrameRatesFor(width, height) }.getOrNull()
                val rate = if (rates != null) frameRate.toDouble().coerceIn(rates.lower, rates.upper).toFloat() else frameRate
                format.setFloat(MediaFormat.KEY_FRAME_RATE, rate)
                val gop = tuning?.iFrameIntervalSec ?: plan.iFrameIntervalSec
                format.setFloat(MediaFormat.KEY_I_FRAME_INTERVAL, gop)
                var qpMax = -1
                val wantedQpMax = tuning?.qpMax
                if (wantedQpMax != null) {
                    if (Build.VERSION.SDK_INT < 31 || !caps.isFeatureSupported(FEATURE_QP_BOUNDS)) continue
                    format.setInteger(KEY_VIDEO_QP_MAX, wantedQpMax)
                    qpMax = wantedQpMax
                }
                val settings = EncoderSettings(
                    bitrateBps = bitrate, bitrateMode = modeLabel, cqQuality = cq, qpMax = qpMax, iFrameIntervalSec = gop, frameRate = rate,
                    reason = if (tuning?.bitrateBps != null) "elle (kalibrasyon)" else plan.reason,
                )
                // Renk bilgisi kaynaktan korunur. Kaynak belirtmiyorsa (0/yok) oynatıcıların etiketsiz videoya
                // uyguladığı varsayılan açıkça yazılır (mpv kuralı: genişlik ≥ 1280 veya yükseklik > 576 → BT.709,
                // aksi BT.601); sınırlı aralık; SDR.
                // Aksi halde bazı kodlayıcılar (OMX.google.h264, API 30) matris katsayısını 0 = RGB yazıyor ve
                // etikete uyan oynatıcılarda renkler bozuluyordu (kalibrasyonda "gbr" etiketi).
                val standard = color?.intOr(MediaFormat.KEY_COLOR_STANDARD, 0) ?: 0
                val range = color?.intOr(MediaFormat.KEY_COLOR_RANGE, 0) ?: 0
                var transfer = color?.intOr(MediaFormat.KEY_COLOR_TRANSFER, 0) ?: 0
                // 8 bit çıktı HDR aktarım eğrisi taşıyamaz.
                if (transfer == MediaFormat.COLOR_TRANSFER_ST2084 || transfer == MediaFormat.COLOR_TRANSFER_HLG) transfer = MediaFormat.COLOR_TRANSFER_SDR_VIDEO
                format.setInteger(
                    MediaFormat.KEY_COLOR_STANDARD,
                    if (standard != 0) standard else if (width >= 1280 || height > 576) MediaFormat.COLOR_STANDARD_BT709 else MediaFormat.COLOR_STANDARD_BT601_NTSC,
                )
                format.setInteger(MediaFormat.KEY_COLOR_RANGE, if (range != 0) range else MediaFormat.COLOR_RANGE_LIMITED)
                format.setInteger(MediaFormat.KEY_COLOR_TRANSFER, if (transfer != 0) transfer else MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
                // Kare olmayan piksel (anamorfik) kaynakta piksel en-boy oranı korunur; kodlayıcı VUI'ye yazar.
                if (sar != null) {
                    format.setInteger(MediaFormat.KEY_PIXEL_ASPECT_RATIO_WIDTH, sar.first)
                    format.setInteger(MediaFormat.KEY_PIXEL_ASPECT_RATIO_HEIGHT, sar.second)
                }
                try {
                    val codec = MediaCodec.createByCodecName(candidate.name)
                    try {
                        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                        codec.start()
                        return Encoder(codec, mime, candidate.name, settings)
                    } catch (e: Exception) {
                        codec.release()
                        lastError = e
                    }
                } catch (e: Exception) {
                    lastError = e
                }
            }
        }
        throw JobException("Bu cihazda ${width}×$height çözünürlüğünü kodlayabilen bir ${order.joinToString("/") { CodecNegotiator.label(it) }} kodlayıcı yok", lastError)
    }

    /**
     * Video izinin örnek tablosunu (veriyi okumadan, yalnızca boyutları) tarayıp gerçek video bit hızını ölçer.
     * Kapsayıcının KEY_BIT_RATE beyanı çoğu dosyada yoktur ya da ses dahil toplamdır; ona güvenilmez.
     */
    private fun measureVideoTrack(track: Int): SourceVideoStats? {
        val probe = MediaExtractor()
        try {
            probe.setDataSource(context, input, null)
            probe.selectTrack(track)
            var bytes = 0L
            var samples = 0
            var sync = 0
            var first = Long.MAX_VALUE
            var last = Long.MIN_VALUE
            while (probe.sampleTrackIndex >= 0) {
                val size = probe.sampleSize
                if (size > 0) bytes += size
                val time = probe.sampleTime
                first = minOf(first, time)
                last = maxOf(last, time)
                if (probe.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) sync++
                samples++
                if (!probe.advance()) break
            }
            if (samples < 2 || last <= first) return null
            // Son karenin süresi: ortalama kare aralığı.
            val durationUs = (last - first) + (last - first) / (samples - 1)
            return SourceVideoStats(bytes, samples, sync, durationUs)
        } finally {
            probe.release()
        }
    }

    /** [mime] türünü çözebilen çözücü adları; yazılım çözücüler donanımdakilerden sonra gelir. */
    private fun decoderNames(mime: String, skipped: Set<String>): List<String> =
        MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
            .filter { info ->
                !info.isEncoder && !info.isAlias && info.name !in skipped && info.supportedTypes.any { it.equals(mime, ignoreCase = true) }
            }
            .sortedBy { if (it.isSoftwareOnly) 1 else 0 }
            .map { it.name }

    /**
     * Ayrı altyazı izi çıkarma (0 yapay zekâ): video paketleri ve uyumlu ses paketleri YENİDEN KODLANMADAN,
     * bayt bayt ve zaman damgalarıyla birlikte yeni bir MP4'e kopyalanır; altyazı / kapalı başlık izleri yazılmaz.
     * Görüntüye gömülü (yakılmış) yazıya dokunmaz. Video türü MP4'e kopyalanamıyorsa (VP8/VP9) iş reddedilir.
     */
    private fun remux(): JobResult {
        val started = SystemClock.elapsedRealtime()
        val extractor = MediaExtractor()
        val audioPaths = ArrayList<Pair<AudioPath, MediaExtractor>>()
        val opened = ArrayList<MediaExtractor>()
        var muxer: MediaMuxer? = null
        var descriptor: ParcelFileDescriptor? = null
        var muxerStarted = false
        val notes = ArrayList<String>()
        try {
            try {
                extractor.setDataSource(context, input, null)
            } catch (e: Exception) {
                throw JobException("Dosya açılamadı veya kapsayıcı biçimi tanınmadı", e)
            }
            val formats = (0 until extractor.trackCount).map { extractor.getTrackFormat(it) }
            val mimes = formats.map { it.getString(MediaFormat.KEY_MIME) ?: "" }
            val videoTrack = mimes.indexOfFirst { it.startsWith("video/") }.takeIf { it >= 0 } ?: throw JobException("Dosyada video akışı bulunamadı")
            val videoFormat = formats[videoTrack]
            val videoMime = mimes[videoTrack]
            if (videoMime !in REMUX_VIDEO_TYPES && !(videoMime == CodecNegotiator.AV1 && Build.VERSION.SDK_INT >= 34)) {
                throw JobException("${CodecNegotiator.label(videoMime)} videosu MP4'e yeniden kodlanmadan taşınamıyor; normal temizleme kullanın")
            }
            val subtitles = mimes.filter(::isSubtitleMime)
            val skippedOther = mimes.filterIndexed { i, m -> i != videoTrack && !m.startsWith("audio/") && !isSubtitleMime(m) && m.startsWith("video/") }
            if (skippedOther.isNotEmpty()) notes += "Ek video izleri (${skippedOther.size}) çıktıya alınmadı."
            val width = videoFormat.intOr(MediaFormat.KEY_WIDTH, 0)
            val height = videoFormat.intOr(MediaFormat.KEY_HEIGHT, 0)

            val fd = context.contentResolver.openFileDescriptor(output, "rw") ?: throw JobException("Çıktı dosyası açılamadı")
            descriptor = fd
            val mux = MediaMuxer(fd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = mux
            mux.setOrientationHint(videoFormat.intOr(MediaFormat.KEY_ROTATION, 0))

            // Ön dolgu nedeniyle negatif damgalı ilk paket varsa bütün çıktı aynı miktarda kaydırılır (bkz. AudioCopy).
            extractor.selectTrack(videoTrack)
            var offsetUs = max(0L, -extractor.sampleTime)
            val audioTracks = mimes.indices.filter { mimes[it].startsWith("audio/") }
            val audioExtractors = audioTracks.map { track ->
                MediaExtractor().also {
                    opened += it
                    it.setDataSource(context, input, null)
                    it.selectTrack(track)
                    offsetUs = max(offsetUs, -it.sampleTime)
                }
            }
            offsetUs = offsetUs.coerceAtMost(MAX_TIMELINE_OFFSET_US)
            val audioLabels = ArrayList<String>()
            for ((i, track) in audioTracks.withIndex()) {
                val source = audioExtractors[i]
                val format = formats[track]
                try {
                    val path: AudioPath = if (mimes[track] in MP4_AUDIO_TYPES) AudioCopy(source, format, offsetUs)
                    else AudioTranscode(source, format, offsetUs).also { it.prepare() }
                    audioPaths += path to source
                    audioLabels += path.label
                } catch (e: Exception) {
                    audioLabels += "ses izi atlandı (${mimes[track]}: ${e.message ?: e.javaClass.simpleName})"
                    notes += "Bir ses izi çıktıya alınamadı."
                }
            }
            val videoOut = mux.addTrack(videoFormat)
            val audioOut = audioPaths.map { (path, _) -> mux.addTrack(path.trackFormat) }
            mux.start()
            muxerStarted = true

            val capacity = maxOf(videoFormat.intOr(MediaFormat.KEY_MAX_INPUT_SIZE, 0), width * height * 3 / 2, 1 shl 20)
            val buffer = ByteBuffer.allocateDirect(capacity)
            val sampleInfo = MediaCodec.BufferInfo()
            var copied = 0L
            var lastPts = 0L
            var lastReport = 0L
            val durationUs = if (videoFormat.containsKey(MediaFormat.KEY_DURATION)) videoFormat.getLong(MediaFormat.KEY_DURATION) else 0L
            while (extractor.sampleTrackIndex >= 0) {
                if (cancelled) throw JobCancelledException()
                val size = extractor.readSampleData(buffer, 0)
                if (size < 0) break
                val pts = extractor.sampleTime + offsetUs
                val flags = if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                buffer.position(0)
                buffer.limit(size)
                sampleInfo.set(0, size, pts, flags)
                mux.writeSampleData(videoOut, buffer, sampleInfo)
                buffer.clear()
                copied++
                lastPts = pts
                for ((k, entry) in audioPaths.withIndex()) entry.first.pump(mux, audioOut[k], pts)
                extractor.advance()
                val now = SystemClock.elapsedRealtime()
                if (now - lastReport >= PROGRESS_INTERVAL_MS) {
                    lastReport = now
                    val elapsed = (now - started).coerceAtLeast(1)
                    onProgress(JobProgress(copied, copied, lastPts - offsetUs, durationUs, copied * 1000.0 / elapsed, elapsed, width, height))
                }
            }
            for ((k, entry) in audioPaths.withIndex()) entry.first.pump(mux, audioOut[k], Long.MAX_VALUE)
            mux.stop()
            muxerStarted = false
            notes += if (subtitles.isEmpty()) "Dosyada ayrı altyazı izi yoktu; video ve ses yeniden kodlanmadan kopyalandı."
            else "${subtitles.size} ayrı altyazı izi çıkarıldı (${subtitles.joinToString()}); video yeniden kodlanmadı."
            return JobResult(
                framesDecoded = copied, framesEncoded = copied, elapsedMs = SystemClock.elapsedRealtime() - started,
                width = width, height = height, framesQueued = copied,
                audio = if (audioLabels.isEmpty()) "ses akışı yok" else audioLabels.joinToString("; "),
                outputMime = videoMime, encoderName = "yok (remux, yeniden kodlama yok)", restorerAttached = false,
                notes = notes, stats = CoreStats(), remuxed = true, removedSubtitleTracks = subtitles,
                sourceFrameRate = videoFormat.getNumber(MediaFormat.KEY_FRAME_RATE)?.toFloat() ?: 0f,
            )
        } finally {
            for ((path, _) in audioPaths) runCatching { path.release() }
            for (source in opened) runCatching { source.release() }
            if (muxerStarted) runCatching { muxer?.stop() }
            runCatching { muxer?.release() }
            runCatching { descriptor?.close() }
            runCatching { extractor.release() }
        }
    }

    private fun transcode(skippedDecoders: Set<String>, hardwarePath: Boolean): JobResult {
        val started = SystemClock.elapsedRealtime()
        val extractor = MediaExtractor()
        var audioExtractor: MediaExtractor? = null
        var audio: AudioPath? = null
        var decoder: MediaCodec? = null
        var encoder: Encoder? = null
        var muxer: MediaMuxer? = null
        var descriptor: ParcelFileDescriptor? = null
        var session: CleanerSession? = null
        var reader: ImageReader? = null
        var muxerStarted = false
        val notes = ArrayList<String>()
        try {
            // ---- kaynak ------------------------------------------------------
            try {
                extractor.setDataSource(context, input, null)
            } catch (e: Exception) {
                throw JobException("Dosya açılamadı veya kapsayıcı biçimi tanınmadı", e)
            }
            val videoTrack = findTrack(extractor, "video/") ?: throw JobException("Dosyada video akışı bulunamadı")
            val sourceFormat = extractor.getTrackFormat(videoTrack)
            extractor.selectTrack(videoTrack)
            val sourceMime = sourceFormat.getString(MediaFormat.KEY_MIME) ?: throw JobException("Video türü okunamadı")
            val durationUs = if (sourceFormat.containsKey(MediaFormat.KEY_DURATION)) sourceFormat.getLong(MediaFormat.KEY_DURATION) else 0L
            val rotation = sourceFormat.intOr(MediaFormat.KEY_ROTATION, 0)
            val sourceBitrate = sourceFormat.intOr(MediaFormat.KEY_BIT_RATE, 0)
            // Piksel en-boy oranı (SAR): 1:1 değilse çıktıya taşınır (VUI + MP4 'pasp').
            val sourceSar = run {
                val w = sourceFormat.intOr(MediaFormat.KEY_PIXEL_ASPECT_RATIO_WIDTH, 1)
                val h = sourceFormat.intOr(MediaFormat.KEY_PIXEL_ASPECT_RATIO_HEIGHT, 1)
                if (w > 0 && h > 0 && w != h) w to h else null
            }
            val sourceVideo = runCatching { measureVideoTrack(videoTrack) }.getOrNull()
            // Kare hızı yalnızca kodlayıcının hız denetimi içindir; kare zaman damgaları kaynaktan aynen taşınır.
            // Beyan yoksa örnek tablosundan ölçülen ortalama kullanılır.
            val measuredFps = sourceVideo?.let { if (it.durationUs > 0) it.samples * 1_000_000f / it.durationUs else null }
            val frameRate = (sourceFormat.getNumber(MediaFormat.KEY_FRAME_RATE)?.toFloat() ?: measuredFps ?: 30f).coerceIn(1f, 960f)

            if (CodecNegotiator.isHdr(sourceFormat)) {
                val hdr10 = runCatching { CodecNegotiator.survey().any { it.hdr10 && it.usable } }.getOrDefault(false)
                notes += "HDR kaynak (PQ/HLG): VC-Core 8 bit çalıştığından çıktı 8 bit SDR etiketli (ton eşlemesi yok)." +
                    if (hdr10) " Cihazda HDR10 kodlayıcı var; 10 bit hat sonraki sürümün işi." else ""
            }
            // HardwareBuffer yolu: çözücü Surface'e (ImageReader) yazar; kareler HardwareBuffer olarak alınır.
            val surfaceReader = if (hardwarePath) {
                val w = sourceFormat.intOr(MediaFormat.KEY_WIDTH, 0)
                val h = sourceFormat.intOr(MediaFormat.KEY_HEIGHT, 0)
                if (w <= 0 || h <= 0) throw HardwarePathFailedException("kaynak boyutu bilinmiyor")
                try {
                    ImageReader.newInstance(w, h, ImageFormat.YUV_420_888, 4, HardwareBuffer.USAGE_CPU_READ_OFTEN).also { reader = it }
                } catch (e: Exception) {
                    throw HardwarePathFailedException("ImageReader açılamadı: ${e.message}", e)
                }
            } else {
                sourceFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                null
            }
            // Yapılandırılamayan çözücü atlanır; sıradaki denenir (ör. tek sayılı boyutu reddeden VP9 çözücü).
            var decoderError: Exception? = null
            var decoderName = ""
            for (name in decoderNames(sourceMime, skippedDecoders)) {
                val codec = runCatching { MediaCodec.createByCodecName(name) }.getOrElse { decoderError = it as? Exception; null } ?: continue
                try {
                    codec.configure(sourceFormat, surfaceReader?.surface, null, 0)
                    codec.start()
                    decoder = codec
                    decoderName = name
                    break
                } catch (e: Exception) {
                    codec.release()
                    decoderError = e
                }
            }
            val dec = decoder ?: throw JobException("Bu cihaz '$sourceMime' videosunu çözemiyor", decoderError)

            // ---- hedef -------------------------------------------------------
            val fd = context.contentResolver.openFileDescriptor(output, "rw") ?: throw JobException("Çıktı dosyası açılamadı")
            descriptor = fd
            val mux = MediaMuxer(fd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = mux
            mux.setOrientationHint(rotation)

            // ---- ses -----------------------------------------------------------
            var timelineOffsetUs = 0L
            var audioTrackOut = -1
            var audioLabel = "ses akışı yok"
            val audioSource = MediaExtractor().also { it.setDataSource(context, input, null) }
            audioExtractor = audioSource
            findTrack(audioSource, "audio/")?.let { track ->
                val format = audioSource.getTrackFormat(track)
                audioSource.selectTrack(track)
                // Ön dolgu: ilk ses paketi negatif damgalıysa bütün çıktı o kadar ileri kaydırılır (bkz. AudioCopy).
                timelineOffsetUs = max(0L, -audioSource.sampleTime).coerceAtMost(MAX_TIMELINE_OFFSET_US)
                try {
                    val path: AudioPath = if (format.getString(MediaFormat.KEY_MIME) in MP4_AUDIO_TYPES) {
                        AudioCopy(audioSource, format, timelineOffsetUs)
                    } else {
                        AudioTranscode(audioSource, format, timelineOffsetUs).also {
                            audio = it
                            it.prepare()
                        }
                    }
                    audio = path
                    audioTrackOut = mux.addTrack(path.trackFormat)
                    audioLabel = path.label
                } catch (e: Exception) {
                    audio?.release()
                    audio = null
                    audioTrackOut = -1
                    audioLabel = "ses atlandı (${format.getString(MediaFormat.KEY_MIME)}: ${e.message ?: e.javaClass.simpleName})"
                    notes += "Ses akışı çıktıya alınamadı."
                }
            }

            // ---- döngü durumu --------------------------------------------------
            val info = MediaCodec.BufferInfo()
            var colorFormat: MediaFormat? = null
            var inputDone = false
            var decoderDone = false
            var eosQueued = false
            var encoderDone = false
            var videoTrackOut = -1
            var framesDecoded = 0L
            var framesQueued = 0L
            var framesEncoded = 0L
            var lastPtsIn = Long.MIN_VALUE
            var lastPtsOut = 0L
            var lastReport = 0L

            fun drainEncoder(timeoutUs: Long) {
                val enc = encoder?.codec ?: return
                while (true) {
                    val index = enc.dequeueOutputBuffer(info, timeoutUs)
                    when {
                        index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            check(!muxerStarted) { "kodlayıcı biçimi iki kez değişti" }
                            // Kodlayıcı SAR'ı çıkış biçimine koymayabilir; çoklayıcı 'pasp' kutusunu bu biçimden yazar.
                            val trackFormat = enc.outputFormat
                            sourceSar?.let { (w, h) ->
                                trackFormat.setInteger(MediaFormat.KEY_PIXEL_ASPECT_RATIO_WIDTH, w)
                                trackFormat.setInteger(MediaFormat.KEY_PIXEL_ASPECT_RATIO_HEIGHT, h)
                            }
                            videoTrackOut = mux.addTrack(trackFormat)
                            mux.start()
                            muxerStarted = true
                        }
                        index >= 0 -> {
                            val data = enc.getOutputBuffer(index) ?: throw JobException("Kodlayıcı çıktısı okunamadı")
                            val config = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                            if (!config && info.size > 0) {
                                check(muxerStarted) { "çoklayıcı başlamadan veri geldi" }
                                if (timelineOffsetUs != 0L) info.presentationTimeUs += timelineOffsetUs
                                mux.writeSampleData(videoTrackOut, data, info)
                                info.presentationTimeUs -= timelineOffsetUs
                                framesEncoded++
                                lastPtsOut = info.presentationTimeUs
                            }
                            val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            enc.releaseOutputBuffer(index, false)
                            if (end) {
                                encoderDone = true
                                return
                            }
                        }
                        else -> return      // INFO_TRY_AGAIN_LATER
                    }
                }
            }

            fun pushDecoded(image: Image, ptsUs: Long) {
                if (image.format != ImageFormat.YUV_420_888 && image.format != ImageFormat.YCBCR_P010) {
                    if (surfaceReader != null && framesDecoded == 0L) throw HardwarePathFailedException("piksel biçimi ${image.format}")
                    throw JobException("Desteklenmeyen piksel biçimi (${image.format}); 4:2:0 olmayan kaynak")
                }
                val crop = image.cropRect
                // 4:2:0 kodlama çift boyut ister: tek sayılı boyutta son sütun/satır kırpılır.
                val width = crop.width() and 1.inv()
                val height = crop.height() and 1.inv()
                val active = session ?: run {
                    if (width < 2 || height < 2) throw JobException("Kare boyutu çok küçük: ${crop.width()}×${crop.height()}")
                    if (width != crop.width() || height != crop.height()) {
                        notes += "Tek sayılı boyut (${crop.width()}×${crop.height()}) ${width}×$height olarak kırpıldı."
                    }
                    if (image.format == ImageFormat.YCBCR_P010) notes += "10 bit kaynak 8 bite indirildi (ton eşlemesi yok)."
                    if (surfaceReader != null) notes += "Çözme yolu: Surface → ImageReader (HardwareBuffer)."
                    encoder = startEncoder(width, height, frameRate, sourceVideo, sourceMime, colorFormat, sourceSar)
                    CleanerSession(width, height).also { session = it }
                }
                if (width != active.width || height != active.height) {
                    throw JobException("Çözünürlük akış ortasında değişti (${active.width}×${active.height} → ${width}×$height); desteklenmiyor")
                }
                // Çekirdek kesin artan zaman damgası ister; bozuk akışlarda en az 1 µs ilerlet.
                val pts = if (lastPtsIn == Long.MIN_VALUE) ptsUs else max(ptsUs, lastPtsIn + 1)
                active.push(image, pts)
                lastPtsIn = pts
                framesDecoded++
            }

            fun decodeStepUnchecked() {
                if (!inputDone) {
                    val index = dec.dequeueInputBuffer(CODEC_TIMEOUT_US)
                    if (index >= 0) {
                        val buffer = dec.getInputBuffer(index) ?: throw JobException("Çözücü girişi alınamadı")
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            dec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            dec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val index = dec.dequeueOutputBuffer(info, CODEC_TIMEOUT_US)
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    colorFormat = dec.outputFormat
                } else if (index >= 0 && surfaceReader != null) {
                    val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    // Surface kipinde bazı çözücüler boyutu 0 bildirir: akış sonu dışındaki her tampon çizilir.
                    val render = info.size > 0 || !end
                    dec.releaseOutputBuffer(index, render)
                    if (render) {
                        var image: Image? = null
                        val deadline = SystemClock.uptimeMillis() + SURFACE_FRAME_TIMEOUT_MS
                        while (image == null && SystemClock.uptimeMillis() < deadline) {
                            image = surfaceReader.acquireNextImage()
                            if (image == null) SystemClock.sleep(1)
                        }
                        if (image == null) {
                            if (framesDecoded == 0L) throw HardwarePathFailedException("ImageReader kare vermedi")
                            if (info.size > 0) throw JobException("ImageReader kare vermedi (kare ${framesDecoded + 1})")
                        } else {
                            image.use { img -> pushDecoded(img, info.presentationTimeUs) }
                        }
                    }
                    if (end) decoderDone = true
                } else if (index >= 0) {
                    val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    if (info.size > 0) {
                        val image = dec.getOutputImage(index) ?: throw JobException("Çözücü görüntüsü alınamadı")
                        pushDecoded(image, info.presentationTimeUs)
                    }
                    dec.releaseOutputBuffer(index, false)
                    if (end) decoderDone = true
                }
            }

            fun decodeStep() {
                try {
                    decodeStepUnchecked()
                } catch (e: MediaCodec.CodecException) {
                    if (framesDecoded == 0L && surfaceReader != null) throw HardwarePathFailedException("çözücü hatası: ${e.diagnosticInfo}", e)
                    if (framesDecoded == 0L) throw DecoderFailedException(decoderName, e)
                    throw e
                }
            }

            fun encodeStep(active: CleanerSession) {
                val enc = encoder?.codec ?: throw JobException("Kodlayıcı başlatılmadı")
                val index = enc.dequeueInputBuffer(CODEC_TIMEOUT_US)
                if (index < 0) return       // boş giriş tamponu yok: önce çıktı boşaltılacak
                val image = enc.getInputImage(index) ?: throw JobException("Kodlayıcı giriş görüntüsü alınamadı")
                val frame = active.popInto(image)
                if (frame == null) {
                    enc.queueInputBuffer(index, 0, 0, lastPtsOut, 0)
                    return
                }
                enc.queueInputBuffer(index, 0, active.frameBytes, frame.ptsUs, 0)
                framesQueued++
            }

            // ---- ana döngü -----------------------------------------------------
            while (!encoderDone) {
                if (cancelled) throw JobCancelledException()
                val active = session
                when (val step = active?.poll() ?: Step.NeedFrames(1)) {
                    is Step.NeedFrames -> if (!decoderDone) {
                        decodeStep()
                    } else {
                        (active ?: throw JobException("Videoda çözülebilir kare yok")).finish()
                    }
                    is Step.Window -> {
                        val current = active ?: throw JobException("Oturum yok")
                        restorer?.restore(current, step.plan)
                        current.completeWindow()
                    }
                    is Step.Output -> encodeStep(active ?: throw JobException("Oturum yok"))
                    Step.Finished -> if (!eosQueued) {
                        val enc = encoder?.codec ?: throw JobException("Kodlayıcı başlatılmadı")
                        val index = enc.dequeueInputBuffer(CODEC_TIMEOUT_US)
                        if (index >= 0) {
                            enc.queueInputBuffer(index, 0, 0, lastPtsIn + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            eosQueued = true
                        }
                    }
                }
                drainEncoder(if (eosQueued) CODEC_TIMEOUT_US else 0)
                if (muxerStarted && audioTrackOut >= 0) audio?.pump(mux, audioTrackOut, lastPtsOut + timelineOffsetUs)

                val now = SystemClock.elapsedRealtime()
                if (now - lastReport >= PROGRESS_INTERVAL_MS) {
                    lastReport = now
                    val elapsed = (now - started).coerceAtLeast(1)
                    onProgress(
                        JobProgress(
                            framesDecoded, framesEncoded, lastPtsOut, durationUs, framesEncoded * 1000.0 / elapsed, elapsed,
                            session?.width ?: 0, session?.height ?: 0, session?.stats() ?: CoreStats(),
                        ),
                    )
                }
            }
            if (muxerStarted && audioTrackOut >= 0) audio?.pump(mux, audioTrackOut, Long.MAX_VALUE)

            val finalSession = session ?: throw JobException("Videoda çözülebilir kare yok")
            val finalEncoder = encoder ?: throw JobException("Kodlayıcı başlatılmadı")
            val stats = finalSession.stats()
            // Kare sayıları burada denetlenip hata fırlatılmaz: bazı donanım kodlayıcıları bit hızı
            // baskısında kare düşürebilir. Dosya korunur; uyumsuzluk JobResult.intact = false olarak
            // arayüze "bütünlük uyarısı" şeklinde yansır.
            mux.stop()
            muxerStarted = false
            return JobResult(
                framesDecoded, framesEncoded, SystemClock.elapsedRealtime() - started,
                finalSession.width, finalSession.height, framesQueued, audioLabel, finalEncoder.mime, finalEncoder.name,
                restorer != null, notes, stats,
                encoderSettings = finalEncoder.settings, sourceVideo = sourceVideo,
                sourceDeclaredBitrate = sourceBitrate, sourceFrameRate = frameRate,
            )
        } finally {
            session?.close()
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            runCatching { reader?.close() }
            runCatching { encoder?.codec?.stop() }
            runCatching { encoder?.codec?.release() }
            runCatching { audio?.release() }
            if (muxerStarted) runCatching { muxer?.stop() }
            runCatching { muxer?.release() }
            runCatching { descriptor?.close() }
            runCatching { audioExtractor?.release() }
            runCatching { extractor.release() }
        }
    }

    companion object {
        private const val CODEC_TIMEOUT_US = 10_000L
        private const val SURFACE_FRAME_TIMEOUT_MS = 2_000L
        private const val MAX_DECODER_ATTEMPTS = 4

        /** Ön dolgu kaydırması için üst sınır: AAC/Opus/MP3 ön dolgusu bir iki paketi geçmez (≤ 100 ms). */
        private const val MAX_TIMELINE_OFFSET_US = 100_000L

        /** MediaCodecInfo.CodecCapabilities.FEATURE_QpBounds ve MediaFormat.KEY_VIDEO_QP_MAX (API 31). */
        const val FEATURE_QP_BOUNDS = "qp-bounds"
        const val KEY_VIDEO_QP_MAX = "video-qp-max"

        fun bitrateModeName(mode: Int): String = when (mode) {
            MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CQ -> "CQ"
            MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR -> "VBR"
            MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR -> "CBR"
            3 -> "CBR_FD"      // BITRATE_MODE_CBR_FD (API 31)
            else -> "mod($mode)"
        }
        private const val PROGRESS_INTERVAL_MS = 250L

        /** Yeniden kodlamadan MP4'e taşınabilen video türleri (AV1 ayrıca API 34+). */
        private val REMUX_VIDEO_TYPES = setOf(
            MediaFormat.MIMETYPE_VIDEO_AVC, MediaFormat.MIMETYPE_VIDEO_HEVC, MediaFormat.MIMETYPE_VIDEO_MPEG4, MediaFormat.MIMETYPE_VIDEO_H263,
        )

        /** MP4 çoklayıcının doğrudan yazabildiği ses türleri; diğerleri AAC'ye dönüştürülür. */
        private val MP4_AUDIO_TYPES = setOf(
            MediaFormat.MIMETYPE_AUDIO_AAC, MediaFormat.MIMETYPE_AUDIO_AMR_NB, MediaFormat.MIMETYPE_AUDIO_AMR_WB,
        )

        internal fun findTrack(extractor: MediaExtractor, prefix: String): Int? =
            (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true
            }
    }
}

// =============================================================================
// 4. Arayüzle paylaşılan durum ve ön plan hizmeti
// =============================================================================

/** İşin arayüze yansıyan durumu. Yalnızca ana iş parçacığından değiştirilir. */
object JobState {
    var running by mutableStateOf(false)
        internal set
    var progress by mutableStateOf(JobProgress())
        internal set
    var result by mutableStateOf<JobResult?>(null)
        internal set
    var error by mutableStateOf<String?>(null)
        internal set

    /** Son işin çıktı dosyası (aç / paylaş düğmeleri için). */
    var output by mutableStateOf<Uri?>(null)
        internal set
}

/**
 * Uzun işin kalıcı kaydı (checkpoint).
 *
 * Android 15+ `mediaProcessing` ön plan hizmetine 24 saatte toplam 6 saat tanır; süre dolunca sistem
 * `onTimeout` çağırır ve hizmet birkaç saniye içinde durmalıdır. Süreç öldürülebilir de. Bu kayıt işin
 * girdisini, çıktısını, ayarlarını ve ilerlemesini saklar; uygulama yeniden açıldığında yarım kalan iş
 * nedeniyle birlikte gösterilir ve tek dokunuşla yeniden başlatılabilir.
 *
 * Sınır (dürüstçe): MP4 çoklayıcı yarım dosyaya ekleme yapamadığı için iş kaldığı kareden DEVAM ETMEZ;
 * aynı ayarlarla baştan çalışır. Yarım çıktı dosyası silinir.
 */
object JobCheckpoint {
    const val STATE_RUNNING = "running"
    const val STATE_TIMEOUT = "timeout"
    const val STATE_INTERRUPTED = "interrupted"

    data class Record(
        val input: String,
        val output: String,
        val options: JobOptions,
        val state: String,
        val framesDone: Long,
        val positionUs: Long,
        val durationUs: Long,
        val updatedMs: Long,
    ) {
        val fraction: Float get() = if (durationUs > 0) (positionUs.toDouble() / durationUs).coerceIn(0.0, 1.0).toFloat() else 0f
    }

    private const val PREFS = "vc_checkpoint"

    fun save(context: Context, record: Record) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString("input", record.input)
            .putString("output", record.output)
            .putBoolean("hevc", record.options.preferHevc)
            .putFloat("quality", record.options.quality)
            .putString("codec", record.options.codec)
            .putBoolean("remux", record.options.softSubtitleRemux)
            .putBoolean("hwDecode", record.options.hardwareBufferDecode)
            .putString("state", record.state)
            .putLong("frames", record.framesDone)
            .putLong("position", record.positionUs)
            .putLong("duration", record.durationUs)
            .putLong("updated", record.updatedMs)
            .commit()
    }

    fun load(context: Context): Record? {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val input = p.getString("input", null) ?: return null
        val output = p.getString("output", null) ?: return null
        return Record(
            input, output,
            JobOptions(
                preferHevc = p.getBoolean("hevc", false), quality = p.getFloat("quality", 1f), codec = p.getString("codec", null),
                softSubtitleRemux = p.getBoolean("remux", false), hardwareBufferDecode = p.getBoolean("hwDecode", false),
            ),
            p.getString("state", STATE_RUNNING) ?: STATE_RUNNING,
            p.getLong("frames", 0), p.getLong("position", 0), p.getLong("duration", 0), p.getLong("updated", 0),
        )
    }

    /** Hizmet çalışmıyorken "çalışıyor" görünen kayıt: süreç öldürülmüş demektir. */
    fun pending(context: Context): Record? {
        val record = load(context) ?: return null
        return if (record.state == STATE_RUNNING && !JobState.running) record.copy(state = STATE_INTERRUPTED) else record.takeIf { it.state != STATE_RUNNING }
    }

    fun mark(context: Context, state: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString("state", state).commit()
    }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().clear().commit()
    }
}

/**
 * Video işini taşıyan ön plan hizmeti. Android 15+ üzerinde `mediaProcessing`,
 * Android 11-14 üzerinde `dataSync` türüyle çalışır.
 */
class CleanerService : Service() {
    private val main = Handler(Looper.getMainLooper())
    private var job: VideoJob? = null
    private var worker: Thread? = null

    @Volatile
    private var timedOut = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL -> job?.cancel()
            ACTION_START -> {
                val input = intent.getStringExtra(EXTRA_INPUT)?.let(Uri::parse)
                val output = intent.getStringExtra(EXTRA_OUTPUT)?.let(Uri::parse)
                val options = JobOptions(
                    preferHevc = intent.getBooleanExtra(EXTRA_HEVC, false),
                    quality = intent.getFloatExtra(EXTRA_QUALITY, 1.0f),
                    codec = intent.getStringExtra(EXTRA_CODEC),
                    softSubtitleRemux = intent.getBooleanExtra(EXTRA_REMUX, false),
                    hardwareBufferDecode = intent.getBooleanExtra(EXTRA_HW_DECODE, false),
                )
                if (input == null || output == null || worker != null) {
                    if (worker == null) stopSelf(startId)
                } else {
                    startJob(input, output, options)
                }
            }
        }
        return START_NOT_STICKY
    }

    private fun startJob(input: Uri, output: Uri, options: JobOptions) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Video işleme", NotificationManager.IMPORTANCE_LOW))
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        } else {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        startForeground(NOTIFICATION_ID, notification("Hazırlanıyor…", 0f), type)

        JobState.running = true
        JobState.progress = JobProgress()
        JobState.result = null
        JobState.error = null
        JobState.output = null

        var lastNotified = 0L
        var lastSaved = 0L
        timedOut = false
        val checkpoint = JobCheckpoint.Record(input.toString(), output.toString(), options, JobCheckpoint.STATE_RUNNING, 0, 0, 0, System.currentTimeMillis())
        JobCheckpoint.save(applicationContext, checkpoint)
        val videoJob = VideoJob(applicationContext, input, output, options, InferenceRuntime.restorer(applicationContext)) { progress ->
            main.post { JobState.progress = progress }
            val now = SystemClock.elapsedRealtime()
            if (now - lastSaved >= CHECKPOINT_INTERVAL_MS) {
                lastSaved = now
                JobCheckpoint.save(
                    applicationContext,
                    checkpoint.copy(framesDone = progress.framesEncoded, positionUs = progress.positionUs, durationUs = progress.durationUs, updatedMs = System.currentTimeMillis()),
                )
            }
            if (now - lastNotified >= 1000) {
                lastNotified = now
                manager.notify(NOTIFICATION_ID, notification("${progress.framesEncoded} kare · ${"%.0f".format(progress.framesPerSecond)} kare/sn", progress.fraction))
            }
        }
        job = videoJob
        worker = Thread({
            val wakeLock = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "VideoCleaner:job")
            wakeLock.acquire(MAX_JOB_MS)
            var result: JobResult? = null
            var error: String? = null
            try {
                result = videoJob.run()
            } catch (e: JobCancelledException) {
                error = if (timedOut) {
                    "Android ön plan işleme süresi (24 saatte 6 saat) doldu. İş kaydedildi; uygulamadan yeniden başlatabilirsiniz (baştan işlenir)."
                } else {
                    "İptal edildi"
                }
            } catch (e: Throwable) {
                error = e.message ?: e.javaClass.simpleName
            } finally {
                if (wakeLock.isHeld) wakeLock.release()
            }
            // Tek satırlık makine-okur özet: `adb logcat -s VideoCleaner` ile izlenir.
            val r = result
            if (r != null) {
                Log.i(
                    TAG,
                    "RESULT ok=true intact=${r.intact} decoded=${r.framesDecoded} queued=${r.framesQueued} encoded=${r.framesEncoded} " +
                        "size=${r.width}x${r.height} ms=${r.elapsedMs} encoder=${r.encoderName} mime=${r.outputMime} " +
                        "windowsB=${r.stats.windowsB} windowsA=${r.stats.windowsA} cuts=${r.stats.sceneCuts} " +
                        "order=${r.stats.orderViolations} audio=\"${r.audio}\" notes=\"${r.notes.joinToString(" | ")}\"",
                )
            } else {
                Log.e(TAG, "RESULT ok=false error=\"$error\"")
            }
            // Başarı veya kullanıcı iptali: kayıt silinir. Süre dolması: kayıt "timeout" olarak kalır.
            if (timedOut) JobCheckpoint.mark(applicationContext, JobCheckpoint.STATE_TIMEOUT) else JobCheckpoint.clear(applicationContext)
            main.post {
                JobState.output = if (result != null) output else null
                JobState.result = result
                JobState.error = error
                JobState.running = false
                job = null
                worker = null
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }, "video-cleaner-job").also { it.start() }
    }

    /**
     * Android 15+: ön plan hizmeti süre sınırı doldu. Sistem birkaç saniye içinde durmamızı bekler:
     * iş iptal edilir, kayıt "timeout" olarak işaretlenir ve hizmet hemen ön plandan çıkıp durur.
     */
    override fun onTimeout(startId: Int, fgsType: Int) {
        timedOut = true
        JobCheckpoint.mark(applicationContext, JobCheckpoint.STATE_TIMEOUT)
        job?.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        job?.cancel()
        super.onDestroy()
    }

    private fun notification(text: String, fraction: Float): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, App::class.java), PendingIntent.FLAG_IMMUTABLE)
        val cancel = PendingIntent.getService(
            this, 1, Intent(this, CleanerService::class.java).setAction(ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("Video işleniyor")
            .setContentText(text)
            .setProgress(1000, (fraction * 1000).toInt(), fraction <= 0f)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "İptal", cancel).build())
            .build()
    }

    companion object {
        private const val ACTION_START = "com.video.cleaner.START"
        private const val ACTION_CANCEL = "com.video.cleaner.CANCEL"
        private const val EXTRA_INPUT = "input"
        private const val EXTRA_OUTPUT = "output"
        private const val EXTRA_HEVC = "hevc"
        private const val EXTRA_QUALITY = "quality"
        private const val EXTRA_CODEC = "codec"
        private const val EXTRA_REMUX = "remux"
        private const val EXTRA_HW_DECODE = "hwDecode"
        private const val CHECKPOINT_INTERVAL_MS = 5_000L
        const val TAG = "VideoCleaner"
        private const val CHANNEL_ID = "video_cleaner_job"
        private const val NOTIFICATION_ID = 1
        private const val MAX_JOB_MS = 6L * 60 * 60 * 1000

        fun start(context: Context, input: Uri, output: Uri, options: JobOptions = JobOptions()) {
            val intent = Intent(context, CleanerService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_INPUT, input.toString())
                .putExtra(EXTRA_OUTPUT, output.toString())
                .putExtra(EXTRA_HEVC, options.preferHevc)
                .putExtra(EXTRA_QUALITY, options.quality)
                .putExtra(EXTRA_CODEC, options.codec)
                .putExtra(EXTRA_REMUX, options.softSubtitleRemux)
                .putExtra(EXTRA_HW_DECODE, options.hardwareBufferDecode)
            context.startForegroundService(intent)
        }

        fun cancel(context: Context) {
            context.startService(Intent(context, CleanerService::class.java).setAction(ACTION_CANCEL))
        }
    }
}
