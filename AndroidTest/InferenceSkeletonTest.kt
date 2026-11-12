// Video_Cleaner — hibrit çıkarım iskelet testi: EĞİTİLMEMİŞ (sabit tohumlu rastgele) üç model telefonda.
//
// Mimari: ONNX yalnızca evrişim/MatMul hızlandırıcısıdır. Her model üç grafiğe bölünmüştür (Models/Export.py):
// kare başına kodlayıcı, 30 karelik zamansal çekirdek, kare başına çözücü (akışta: kare başına kodlayıcı +
// çift başına eşleştirme). 30 karelik pencere ve ara özellikler grafiğin DIŞINDA (burada yerel bellekte
// doğrudan tamponlar; üretimde Rust halka tamponu) tutulur; her kare BİR kez kodlanır.
//
// Ölçülen: grafik başına süre, Mod B'de kare başına toplam süre, oturum yükleme, en yüksek RSS, ısıl durum,
// NNAPI düğüm kapsamı (ORT günlüğünde "number of nodes supported by NNAPI"). Çıktı görüntüsü anlamsızdır.
// Modeller CI'da üretilip AndroidTest/assets/onnx/ altına konur (depoda tutulmaz). Yalnızca `bench=true` ile.
// Sonuçlar logcat'e "VCBENCH" satırları olarak yazılır.

package com.video.cleaner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class InferenceSkeletonTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val testAssets = instrumentation.context.assets

    /** Yerel bellekte (Java yığını dışında) tutulan tensör: halka tamponunun yerini tutar. */
    private class Feature(val shape: LongArray, val data: FloatBuffer) {
        val perItem: Int get() = (data.capacity() / shape[0]).toInt()
    }

    private fun direct(count: Int): FloatBuffer = ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    private fun random(shape: LongArray, seed: Long): Feature {
        val count = shape.fold(1L) { a, b -> a * b }.toInt()
        val buf = direct(count)
        val rnd = java.util.Random(seed)
        repeat(count) { buf.put(rnd.nextFloat()) }
        buf.rewind()
        return Feature(shape, buf)
    }

    private fun concat(parts: List<Feature>): Feature {
        val shape = parts[0].shape.copyOf().also { it[0] = parts.sumOf { p -> p.shape[0] } }
        val out = direct(parts.sumOf { it.data.capacity() })
        for (p in parts) out.put(p.data.duplicate().also { it.rewind() })
        out.rewind()
        return Feature(shape, out)
    }

    private fun item(f: Feature, i: Int): Feature {
        val per = f.perItem
        val src = f.data.duplicate().also { it.position(i * per); it.limit((i + 1) * per) }
        val out = direct(per)
        out.put(src)
        out.rewind()
        return Feature(f.shape.copyOf().also { it[0] = 1 }, out)
    }

    /** /proc/self/status → VmRSS (MB). */
    private fun rssMb(): Long =
        File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmRSS:") }
            ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.div(1024) ?: -1

    private fun thermal(): Int = (app.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus

    private inner class Pipeline(private val env: OrtEnvironment, private val sessions: Map<String, OrtSession>) {
        val ms = LinkedHashMap<String, Double>()
        private val watchdog = java.util.concurrent.Executors.newSingleThreadScheduledExecutor()
        private val started = SystemClock.elapsedRealtime()

        fun close() = watchdog.shutdownNow()

        fun shapeOf(graph: String, input: String): LongArray = (sessions.getValue(graph).inputInfo.getValue(input).info as TensorInfo).shape

        fun run(graph: String, vararg feeds: Pair<String, Feature>): List<Feature> {
            // Hat başına süre sınırı: bir yürütme sağlayıcısı yavaş yedek çekirdeğe düşerse (x86'da FP16 MatMul
            // gibi) tek çağrı dakikalar sürebilir; RunOptions.setTerminate ile kesilir, sonuç FAIL yazılır.
            if (SystemClock.elapsedRealtime() - started > PIPELINE_LIMIT_MS) throw IllegalStateException("hat süre sınırı aşıldı ($graph öncesi)")
            val tensors = feeds.associate { (name, f) -> name to OnnxTensor.createTensor(env, f.data.duplicate().also { it.rewind() }, f.shape) }
            val options = OrtSession.RunOptions()
            val guard = watchdog.schedule(Runnable { runCatching { options.setTerminate(true) } }, CALL_LIMIT_MS, java.util.concurrent.TimeUnit.MILLISECONDS)
            try {
                val t0 = SystemClock.elapsedRealtimeNanos()
                sessions.getValue(graph).run(tensors, options).use { result ->
                    ms[graph] = (ms[graph] ?: 0.0) + (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                    return result.map { entry ->
                        val tensor = entry.value as OnnxTensor
                        val values = tensor.floatBuffer                  // ORT kopyası (kısa ömürlü)
                        val copy = direct(values.capacity())
                        copy.put(values)
                        copy.rewind()
                        Feature(tensor.info.shape, copy)
                    }
                }
            } finally {
                guard.cancel(false)
                options.close()
                tensors.values.forEach { it.close() }
            }
        }
    }

    @Test
    fun z_untrainedHybridPipelineRunsOnDevice() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("bench=true değil", args.getString("bench") == "true")
        val available = runCatching { testAssets.list("onnx")?.toList() }.getOrNull().orEmpty()
        if (available.none { it.endsWith(".onnx") }) {
            Log.i(TAG, "VCBENCH status=SKIPPED reason=\"assets/onnx yok (Models/Export.py çalıştırılmadı)\"")
            return
        }
        val env = OrtEnvironment.getEnvironment()
        val budgetMs = (args.getString("benchBudgetMin")?.toLongOrNull() ?: 16L) * 60_000L
        val started = SystemClock.elapsedRealtime()
        Log.i(
            TAG,
            "VCBENCH device=\"${Build.MANUFACTURER} ${Build.MODEL}\" soc=\"${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "-"}\" api=${Build.VERSION.SDK_INT} " +
                "cores=${Runtime.getRuntime().availableProcessors()} files=${available.count { it.endsWith(".onnx") }}",
        )
        for (variant in listOf("fp32", "fp16")) {
            for (ep in listOf("xnnpack", "cpu", "nnapi")) {
                if (SystemClock.elapsedRealtime() - started > budgetMs) {
                    Log.i(TAG, "VCBENCH variant=$variant ep=$ep status=SKIPPED reason=\"süre bütçesi doldu\"")
                    continue
                }
                runPipeline(env, variant, ep)
                System.gc()
            }
        }
        Log.i(TAG, "VCBENCH status=DONE total_s=${(SystemClock.elapsedRealtime() - started) / 1000} thermal=${thermal()}")
    }

    private fun runPipeline(env: OrtEnvironment, variant: String, ep: String) {
        val graphs = listOf("det_enc", "det_core", "det_dec", "flow_enc", "flow_pair", "mamba_enc", "mamba_core", "mamba_dec")
        val peak = AtomicLong(rssMb())
        val sampling = AtomicBoolean(true)
        val sampler = Thread {
            while (sampling.get()) {
                peak.accumulateAndGet(rssMb()) { a, b -> maxOf(a, b) }
                Thread.sleep(100)
            }
        }.apply { isDaemon = true; start() }
        val rssBefore = rssMb()
        val thermalBefore = thermal()
        val sessions = LinkedHashMap<String, OrtSession>()
        val files = ArrayList<File>()
        var stage = "yükleme"
        var pipeline: Pipeline? = null
        try {
            val t0 = SystemClock.elapsedRealtime()
            for (g in graphs) {
                stage = "yükleme $g"
                val name = "vc_${g}_$variant.onnx"
                val file = File(app.cacheDir, name)
                testAssets.open("onnx/$name").use { input -> file.outputStream().use { input.copyTo(it, 1 shl 20) } }
                files += file
                val options = OrtSession.SessionOptions()
                options.setIntraOpNumThreads(4)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                when (ep) {
                    "xnnpack" -> options.addXnnpack(mapOf("intra_op_num_threads" to "4"))
                    "nnapi" -> {
                        options.addNnapi()
                        options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                    }
                }
                sessions[g] = env.createSession(file.path, options)
            }
            val loadMs = SystemClock.elapsedRealtime() - t0
            val p = Pipeline(env, sessions)
            pipeline = p
            val t = 30
            val valid = Feature(longArrayOf(1, t.toLong()), direct(t).also { b -> repeat(t) { b.put(1f) }; b.rewind() })

            stage = "algılayıcı"
            val frame = random(p.shapeOf("det_enc", "frame"), 1)
            val det = (0 until t).map { p.run("det_enc", "frame" to frame) }
            val dCore = p.run("det_core", "x8" to concat(det.map { it[2] }), "valid" to valid)
            for (i in 0 until t) p.run("det_dec", "x16" to item(dCore[1], i), "s0" to det[i][0], "s1" to det[i][1], "s2" to item(dCore[0], i))

            stage = "akış"
            val fm = random(p.shapeOf("flow_enc", "frame_mask"), 2)
            val fe = (0 until t).map { p.run("flow_enc", "frame_mask" to fm) }
            val mask = random(p.shapeOf("flow_pair", "mask_src"), 3)
            for (i in 0 until t - 1) {
                for ((a, b) in listOf(i to i + 1, i + 1 to i)) {
                    p.run("flow_pair", "feat_src" to fe[a][0], "feat_dst" to fe[b][0], "ctx_src" to fe[a][1], "mask_src" to mask, "mask_dst" to mask)
                }
            }

            stage = "onarım"
            val packed = random(p.shapeOf("mamba_enc", "packed"), 4)
            val me = (0 until t).map { p.run("mamba_enc", "packed" to packed) }
            val context = random(p.shapeOf("mamba_core", "context"), 5)
            val mCore = p.run("mamba_core", "x8" to concat(me.map { it[2] }), "valid" to valid, "context" to context)
            for (i in 0 until t) {
                p.run("mamba_dec", "packed" to packed, "x16" to item(mCore[1], i), "s3" to item(mCore[0], i), "s2" to me[i][1], "s1" to me[i][0])
            }

            // Mod B (pencere 15 kare kayar): kare başına = kodlayıcı + 2 çift + 2 çözücü + çekirdek/15
            val per = LinkedHashMap<String, Double>()
            for ((g, total) in p.ms) {
                per[g] = when {
                    g.endsWith("_core") -> total / 15
                    g.endsWith("_dec") -> total / t * 2
                    else -> total / t
                }
            }
            val frameMs = per.values.sum()
            Log.i(
                TAG,
                "VCBENCH variant=$variant ep=$ep status=OK load_ms=$loadMs frame_ms=${"%.1f".format(frameMs)} " +
                    per.entries.joinToString(" ") { (g, v) -> "$g=${"%.1f".format(v)}" } +
                    " window_ms=${"%.0f".format(p.ms.values.sum())} rss_before_mb=$rssBefore peak_rss_mb=${peak.get()} thermal=$thermalBefore->${thermal()}",
            )
        } catch (e: Throwable) {
            Log.i(
                TAG,
                "VCBENCH variant=$variant ep=$ep status=FAIL stage=\"$stage\" error=\"${e.javaClass.simpleName}: ${e.message?.take(300)?.replace('"', '\'')}\" " +
                    "peak_rss_mb=${peak.get()}",
            )
        } finally {
            sampling.set(false)
            sampler.join(500)
            pipeline?.close()
            sessions.values.forEach { runCatching { it.close() } }
            files.forEach { it.delete() }
        }
    }

    private companion object {
        const val TAG = "VCTEST"
        const val CALL_LIMIT_MS = 60_000L          // tek grafik çağrısı
        const val PIPELINE_LIMIT_MS = 4 * 60_000L  // bir değişken × sağlayıcı hattının tamamı
    }
}
