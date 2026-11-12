// Video_Cleaner — çıkarım iskelet testi: EĞİTİLMEMİŞ (sabit tohumlu rastgele) üç model telefonda koşturulur.
//
// Amaç eğitimden ÖNCE mobil çalışma zamanı darboğazlarını görmek: operatör uyumluluğu (ONNX Runtime
// CPU / XNNPACK / NNAPI), pencere başına süre, en yüksek bellek ve ısıl durum. Çıktı görüntüsü anlamsızdır,
// ölçülmez. Modeller CI'da Models/Export.py ile üretilip AndroidTest/assets/onnx/ altına konur (depoda
// tutulmaz); yoksa test "ATLANDI" yazar. Yalnızca `bench=true` enstrümantasyon argümanıyla çalışır.
// Sonuçlar logcat'e "VCBENCH" etiketiyle yazılır; NNAPI'nin hangi düğümleri aldığı ORT'nin kendi
// günlüğündeki "number of nodes supported by NNAPI" satırındadır.

package com.video.cleaner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicLong
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class InferenceSkeletonTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val testAssets = instrumentation.context.assets

    /** /proc/self/status → VmRSS (MB). */
    private fun rssMb(): Long =
        File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmRSS:") }
            ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.div(1024) ?: -1

    private fun thermal(): Int = (app.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus

    private fun copyAsset(name: String): File? {
        val target = File(app.cacheDir, name)
        return runCatching {
            testAssets.open("onnx/$name").use { input -> target.outputStream().use { input.copyTo(it, 1 shl 20) } }
            target
        }.getOrNull()
    }

    @Test
    fun z_untrainedModelsRunOnDevice() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("bench=true değil", args.getString("bench") == "true")
        val available = runCatching { testAssets.list("onnx")?.toList() }.getOrNull().orEmpty()
        if (available.none { it.endsWith(".onnx") }) {
            Log.i(TAG, "VCBENCH status=SKIPPED reason=\"assets/onnx yok (Models/Export.py çalıştırılmadı)\"")
            return
        }
        val env = OrtEnvironment.getEnvironment()
        val budgetMs = (args.getString("benchBudgetMin")?.toLongOrNull() ?: 18L) * 60_000L
        val started = SystemClock.elapsedRealtime()
        Log.i(
            TAG,
            "VCBENCH device=\"${Build.MANUFACTURER} ${Build.MODEL}\" soc=\"${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "-"}\" api=${Build.VERSION.SDK_INT} " +
                "cores=${Runtime.getRuntime().availableProcessors()} files=${available.sorted()}",
        )
        // Sıra: önce küçük/ucuz olanlar, böylece süre bütçesi biterse en azından onlar ölçülmüş olur.
        val plan = listOf(
            "detector" to "fp32", "detector" to "int8", "mamba" to "fp32", "mamba" to "int8", "flow" to "fp32", "flow" to "int8",
        )
        for ((model, variant) in plan) {
            val file = "vc_${model}_$variant.onnx"
            if (file !in available) continue
            val eps = if (variant == "fp32") listOf("cpu", "xnnpack", "nnapi") else listOf("cpu")
            val local = copyAsset(file) ?: continue
            try {
                for (ep in eps) {
                    if (SystemClock.elapsedRealtime() - started > budgetMs) {
                        Log.i(TAG, "VCBENCH model=$model variant=$variant ep=$ep status=SKIPPED reason=\"süre bütçesi doldu\"")
                        continue
                    }
                    runOne(env, local, model, variant, ep)
                    System.gc()
                }
            } finally {
                local.delete()
            }
        }
        Log.i(TAG, "VCBENCH status=DONE total_s=${(SystemClock.elapsedRealtime() - started) / 1000} thermal=${thermal()}")
    }

    private fun runOne(env: OrtEnvironment, file: File, model: String, variant: String, ep: String) {
        val peak = AtomicLong(rssMb())
        val sampling = java.util.concurrent.atomic.AtomicBoolean(true)
        val sampler = Thread {
            while (sampling.get()) {
                val now = rssMb()
                peak.accumulateAndGet(now) { a, b -> maxOf(a, b) }
                Thread.sleep(100)
            }
        }.apply { isDaemon = true; start() }
        val rssBefore = rssMb()
        val thermalBefore = thermal()
        var session: OrtSession? = null
        val tensors = ArrayList<OnnxTensor>()
        try {
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(4)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            when (ep) {
                "xnnpack" -> options.addXnnpack(mapOf("intra_op_num_threads" to "4"))
                "nnapi" -> {
                    options.addNnapi()
                    // NNAPI'nin düğüm kapsamı ORT günlüğüne INFO düzeyinde yazılır.
                    options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                }
            }
            val t0 = SystemClock.elapsedRealtime()
            session = env.createSession(file.path, options)
            val loadMs = SystemClock.elapsedRealtime() - t0
            val feeds = LinkedHashMap<String, OnnxTensor>()
            val rnd = java.util.Random(7)
            for ((name, info) in session.inputInfo) {
                val shape = (info.info as ai.onnxruntime.TensorInfo).shape
                val count = shape.fold(1L) { a, b -> a * b }.toInt()
                val data = FloatBuffer.allocate(count)
                repeat(count) { data.put(rnd.nextFloat()) }
                data.flip()
                feeds[name] = OnnxTensor.createTensor(env, data, shape).also { tensors += it }
            }
            val window = feeds.values.first().info.shape[1].toInt()
            val t1 = SystemClock.elapsedRealtime()
            session.run(feeds).close()                          // ilk çalıştırma (ısınma, derleme)
            val firstMs = SystemClock.elapsedRealtime() - t1
            val t2 = SystemClock.elapsedRealtime()
            session.run(feeds).close()
            val runMs = SystemClock.elapsedRealtime() - t2
            Log.i(
                TAG,
                "VCBENCH model=$model variant=$variant ep=$ep status=OK load_ms=$loadMs first_ms=$firstMs window_ms=$runMs " +
                    "frame_ms=${"%.1f".format(runMs.toDouble() / window)} frames=$window rss_before_mb=$rssBefore peak_rss_mb=${peak.get()} " +
                    "thermal=$thermalBefore->${thermal()} file_mb=${file.length() shr 20}",
            )
        } catch (t: Throwable) {
            Log.i(
                TAG,
                "VCBENCH model=$model variant=$variant ep=$ep status=FAIL error=\"${t.javaClass.simpleName}: ${t.message?.take(300)?.replace('"', '\'')}\" " +
                    "peak_rss_mb=${peak.get()}",
            )
        } finally {
            sampling.set(false)
            sampler.join(500)
            tensors.forEach { it.close() }
            runCatching { session?.close() }
        }
    }

    private companion object {
        const val TAG = "VCTEST"
    }
}
