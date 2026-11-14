// Video_Cleaner v2.0 — Compose arayüzü
//
// Metinler bu dosyadadır; res/ yalnızca uygulama simgesini ve logoyu içerir. Video
// seçimi ve çıktı konumu Depolama Erişim Çerçevesi (SAF) ile yapılır; depolama izni
// istenmez. Uygulama başka uygulamalardan "Birlikte aç" / "Paylaş" ile de video alır.

package com.video.cleaner

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class App : ComponentActivity() {
    /** Başka bir uygulamadan ("Birlikte aç" / "Paylaş") gelen video. */
    private var incoming by mutableStateOf<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (savedInstanceState == null) handleIntent(intent)
        setContent {
            VideoCleanerTheme {
                MainScreen(incoming = incoming, onIncomingConsumed = { incoming = null })
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        intent ?: return
        // Yalnızca hata ayıklama derlemesinde: adb ile arayüzsüz otomatik test.
        //   adb shell am start -n com.video.cleaner/.App --es vc_input <dosya> --es vc_output <dosya>
        val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
        val autoInput = intent.getStringExtra("vc_input")
        val autoOutput = intent.getStringExtra("vc_output")
        if (debuggable && autoInput != null && autoOutput != null) {
            val options = JobOptions(
                preferHevc = intent.getBooleanExtra("vc_hevc", false),
                codec = intent.getStringExtra("vc_codec"),
                softSubtitleRemux = intent.getBooleanExtra("vc_remux", false),
                hardwareBufferDecode = intent.getBooleanExtra("vc_hw_decode", false),
            )
            CleanerService.start(this, Uri.fromFile(java.io.File(autoInput)), Uri.fromFile(java.io.File(autoOutput)), options)
            return
        }
        val uri: Uri? = when (intent.action) {
            Intent.ACTION_VIEW -> intent.data
            Intent.ACTION_SEND -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(Intent.EXTRA_STREAM)
            }
            else -> null
        }
        if (uri != null) incoming = uri
    }
}

private val BrandNavy = Color(0xFF0B1630)
private val BrandCyan = Color(0xFF35E0E0)

@Composable
fun VideoCleanerTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme(primary = BrandCyan, onPrimary = BrandNavy)
        else -> lightColorScheme(primary = Color(0xFF00696B))
    }
    MaterialTheme(colorScheme = colors, content = content)
}

/** Yerel çekirdeğin durumu; kitaplık yüklenemezse hata metni. */
private fun describeCore(): String = try {
    val page = Core.nativePageSize() / 1024
    "Çekirdek v${Core.nativeAbiVersion()} · ${Core.kernelBackendName()} · $page KB sayfa"
} catch (e: Throwable) {
    "Yerel çekirdek yüklenemedi: ${e.message ?: e.javaClass.simpleName}"
}

private fun displayName(context: Context, uri: Uri): String {
    runCatching {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) return cursor.getString(0) ?: ""
        }
    }
    return uri.lastPathSegment ?: "video"
}

private fun suggestedOutputName(inputName: String): String {
    val base = inputName.substringBeforeLast('.', inputName).ifBlank { "video" }
    return "${base}_temiz.mp4"
}

private fun codecLabel(mime: String?): String = when (mime) {
    null, "" -> "yok"
    "video/avc" -> "H.264"
    "video/hevc" -> "HEVC"
    "video/x-vnd.on2.vp9" -> "VP9"
    "video/x-vnd.on2.vp8" -> "VP8"
    "video/av01" -> "AV1"
    "video/mp4v-es" -> "MPEG-4"
    "audio/mp4a-latm" -> "AAC"
    "audio/opus" -> "Opus"
    "audio/vorbis" -> "Vorbis"
    "audio/mpeg" -> "MP3"
    else -> mime.substringAfter('/')
}

private fun clock(ms: Long): String {
    val seconds = (ms / 1000).coerceAtLeast(0)
    return if (seconds >= 3600) "%d:%02d:%02d".format(Locale.ROOT, seconds / 3600, seconds / 60 % 60, seconds % 60)
    else "%d:%02d".format(Locale.ROOT, seconds / 60, seconds % 60)
}

@Composable
fun MainScreen(incoming: Uri? = null, onIncomingConsumed: () -> Unit = {}) {
    val context = LocalContext.current
    var input by rememberSaveable { mutableStateOf<Uri?>(null) }
    var inputName by rememberSaveable { mutableStateOf("") }
    var preferHevc by rememberSaveable { mutableStateOf(false) }
    var highQuality by rememberSaveable { mutableStateOf(false) }
    var softRemux by rememberSaveable { mutableStateOf(false) }
    var diagnostics by remember { mutableStateOf<List<Pair<String, String>>?>(null) }
    var showDiagnostics by rememberSaveable { mutableStateOf(false) }
    var checkpointVersion by remember { mutableStateOf(0) }
    val coreInfo = remember { describeCore() }
    val pending = remember(JobState.running, checkpointVersion) { if (JobState.running) null else JobCheckpoint.pending(context) }
    LaunchedEffect(showDiagnostics) {
        if (showDiagnostics && diagnostics == null) diagnostics = withContext(Dispatchers.IO) { Diagnostics.collect(context) }
    }

    LaunchedEffect(incoming) {
        if (incoming != null) {
            input = incoming
            inputName = displayName(context, incoming)
            onIncomingConsumed()
        }
    }
    val info by produceState<VideoInfo?>(initialValue = null, input) {
        val uri = input
        value = if (uri == null) null else withContext(Dispatchers.IO) { VideoInfo.probe(context, uri) }
    }

    val pickVideo = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            // Kalıcı okuma izni: iş yarım kalırsa (6 saat sınırı, süreç sonlanması) yeniden başlatılabilsin.
            runCatching { context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            input = uri
            inputName = displayName(context, uri)
            softRemux = false
        }
    }
    val createOutput = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("video/mp4")) { uri ->
        val source = input
        if (uri != null && source != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            }
            val remux = softRemux && info?.subtitleTracks?.isNotEmpty() == true
            CleanerService.start(context, source, uri, JobOptions(preferHevc, if (highQuality) 2.0f else 1.0f, softSubtitleRemux = remux))
        }
    }
    // Bildirim izni reddedilse de iş çalışır; yalnızca ilerleme bildirimi görünmez.
    val askNotifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        createOutput.launch(suggestedOutputName(inputName))
    }

    val onStart = {
        val needsPermission = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        if (needsPermission) {
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            createOutput.launch(suggestedOutputName(inputName))
        }
    }
    val send = { action: String ->
        JobState.output?.let { uri ->
            val intent = Intent(action).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            if (action == Intent.ACTION_VIEW) intent.setDataAndType(uri, "video/mp4") else intent.setType("video/mp4").putExtra(Intent.EXTRA_STREAM, uri)
            runCatching { context.startActivity(Intent.createChooser(intent, null)) }
        }
        Unit
    }

    MainContent(
        coreInfo = coreInfo,
        inputName = inputName.takeIf { input != null },
        info = info,
        preferHevc = preferHevc,
        highQuality = highQuality,
        running = JobState.running,
        progress = JobState.progress,
        result = JobState.result,
        error = JobState.error,
        softRemux = softRemux,
        pending = pending,
        diagnostics = diagnostics.takeIf { showDiagnostics },
        diagnosticsOpen = showDiagnostics,
        onPick = { pickVideo.launch(arrayOf("video/*")) },
        onSoftRemux = { softRemux = it },
        onResume = { record ->
            CleanerService.start(context, Uri.parse(record.input), Uri.parse(record.output), record.options)
            checkpointVersion++
        },
        onDiscard = {
            JobCheckpoint.clear(context)
            checkpointVersion++
        },
        onDiagnostics = { showDiagnostics = !showDiagnostics },
        onCodec = { preferHevc = it },
        onQuality = { highQuality = it },
        onStart = onStart,
        onCancel = { CleanerService.cancel(context) },
        onOpen = { send(Intent.ACTION_VIEW) },
        onShare = { send(Intent.ACTION_SEND) },
    )
}

@Composable
fun MainContent(
    coreInfo: String,
    inputName: String?,
    info: VideoInfo?,
    preferHevc: Boolean,
    highQuality: Boolean,
    running: Boolean,
    progress: JobProgress,
    result: JobResult?,
    error: String?,
    softRemux: Boolean = false,
    pending: JobCheckpoint.Record? = null,
    diagnostics: List<Pair<String, String>>? = null,
    diagnosticsOpen: Boolean = false,
    onPick: () -> Unit,
    onSoftRemux: (Boolean) -> Unit = {},
    onResume: (JobCheckpoint.Record) -> Unit = {},
    onDiscard: () -> Unit = {},
    onDiagnostics: () -> Unit = {},
    onCodec: (Boolean) -> Unit,
    onQuality: (Boolean) -> Unit,
    onStart: () -> Unit,
    onCancel: () -> Unit,
    onOpen: () -> Unit,
    onShare: () -> Unit,
) {
    Scaffold { inner ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Column(modifier = Modifier.widthIn(max = 560.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Header(coreInfo)

                Surface(color = MaterialTheme.colorScheme.tertiaryContainer, shape = RoundedCornerShape(14.dp)) {
                    Text(
                        "Onarım modeli henüz bağlı değil: video çözülür, 15 karelik pencere hattından geçer ve " +
                            "kareler değiştirilmeden yeniden kodlanır. Altyazı silme, eğitilmiş modellerle etkinleşecek.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                    )
                }

                if (pending != null && !running) {
                    SectionCard(title = "Yarım kalan iş", isError = true) {
                        Text(
                            (if (pending.state == JobCheckpoint.STATE_TIMEOUT) "Android ön plan işleme süresi (24 saatte 6 saat) doldu."
                            else "Uygulama işlem sürerken kapatıldı.") +
                                " %d kare işlenmişti (%%%d). Yeniden başlatınca iş aynı ayarlarla BAŞTAN işlenir."
                                    .format(Locale.ROOT, pending.framesDone, (pending.fraction * 100).toInt()),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(top = 8.dp)) {
                            FilledTonalButton(onClick = { onResume(pending) }) { Text("Yeniden başlat") }
                            OutlinedButton(onClick = onDiscard) { Text("Kaydı sil") }
                        }
                    }
                }

                SectionCard(title = "Kaynak") {
                    if (inputName == null) {
                        Text(
                            "Her kapsayıcı, kodek, çözünürlük ve kare hızı kabul edilir; cihazınızın çözebildiği her video işlenir.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Button(onClick = onPick, enabled = !running, modifier = Modifier.fillMaxWidth().padding(top = 12.dp)) {
                            Text("Video seç")
                        }
                    } else {
                        Text(inputName, style = MaterialTheme.typography.titleSmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
                        if (info != null) {
                            Text(
                                buildString {
                                    append("${info.width}×${info.height}")
                                    if (info.frameRate > 0f) append(" · %.4g kare/sn".format(Locale.ROOT, info.frameRate))
                                    if (info.durationUs > 0) append(" · ${clock(info.durationUs / 1000)}")
                                    append(" · ${codecLabel(info.videoMime)}")
                                    append(" · ses: ${codecLabel(info.audioMime)}")
                                    if (info.rotation != 0) append(" · ${info.rotation}° döndürülmüş")
                                    if (info.subtitleTracks.isNotEmpty()) append(" · ${info.subtitleTracks.size} ayrı altyazı izi")
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(top = 4.dp),
                            )
                        }
                        TextButton(onClick = onPick, enabled = !running, modifier = Modifier.padding(top = 4.dp)) { Text("Başka video seç") }
                    }
                }

                SectionCard(title = "Çıktı (MP4)") {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(selected = !preferHevc, onClick = { onCodec(false) }, enabled = !running, label = { Text("H.264") })
                        FilterChip(selected = preferHevc, onClick = { onCodec(true) }, enabled = !running, label = { Text("HEVC") })
                        Spacer(Modifier.width(8.dp))
                        FilterChip(selected = !highQuality, onClick = { onQuality(false) }, enabled = !running, label = { Text("Standart") })
                        FilterChip(selected = highQuality, onClick = { onQuality(true) }, enabled = !running, label = { Text("Yüksek") })
                    }
                    if (info != null && info.subtitleTracks.isNotEmpty()) {
                        FilterChip(
                            selected = softRemux,
                            onClick = { onSoftRemux(!softRemux) },
                            enabled = !running,
                            label = { Text("Yalnızca ayrı altyazı izini çıkar") },
                        )
                        if (softRemux) {
                            Text(
                                "Bu dosyadaki altyazı görüntüye gömülü değil, ayrı bir izde. Video yeniden kodlanmadan, " +
                                    "bayt bayt kopyalanır; kalite kaybı olmaz. Görüntüye gömülü yazılara dokunulmaz.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    Text(
                        (if (preferHevc) "HEVC: daha küçük dosya. " else "H.264: en geniş uyumluluk. ") +
                            (if (highQuality) "Yüksek: bit hızı iki katı." else "Standart: kaynağa yakın bit hızı.") +
                            " Ses mümkünse aynen kopyalanır.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                    Button(
                        onClick = onStart,
                        enabled = !running && inputName != null,
                        modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    ) { Text("İşle ve kaydet") }
                }

                if (running) {
                    SectionCard(title = "İşleniyor") {
                        Text(
                            if (progress.durationUs > 0) "%%%d".format(Locale.ROOT, (progress.fraction * 100).toInt()) else "…",
                            style = MaterialTheme.typography.displaySmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        if (progress.durationUs > 0) {
                            LinearProgressIndicator(progress = { progress.fraction }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                        }
                        Text(
                            buildString {
                                append("%d kare · %.1f kare/sn".format(Locale.ROOT, progress.framesEncoded, progress.framesPerSecond))
                                if (progress.width > 0) append(" · ${progress.width}×${progress.height}")
                                append("\nGeçen ${clock(progress.elapsedMs)}")
                                if (progress.remainingMs >= 0) append(" · kalan yaklaşık ${clock(progress.remainingMs)}")
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(top = 12.dp),
                        )
                        StatsText(progress.stats)
                        OutlinedButton(onClick = onCancel, modifier = Modifier.padding(top = 12.dp)) { Text("İptal") }
                    }
                }

                error?.let { message ->
                    SectionCard(title = "İşlem tamamlanamadı", isError = true) {
                        Text(message, style = MaterialTheme.typography.bodyMedium)
                    }
                }

                result?.let { done ->
                    SectionCard(title = if (done.intact) "Tamamlandı" else "Tamamlandı — bütünlük uyarısı", isError = !done.intact) {
                        Text(
                            "%d kare · %s · %.1f kare/sn · %d×%d · %s".format(
                                Locale.ROOT, done.framesEncoded, clock(done.elapsedMs), done.framesPerSecond,
                                done.width, done.height, codecLabel(done.outputMime),
                            ),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            if (done.intact) {
                                "Kare bütünlüğü: çözülen ${done.framesDecoded} = kodlanan ${done.framesEncoded}, sıra hatası yok."
                            } else {
                                "Çözülen ${done.framesDecoded}, kodlayıcıya verilen ${done.framesQueued}, kodlanan ${done.framesEncoded}, " +
                                    "sıra hatası ${done.stats.orderViolations}. Çıktı dosyası korundu; kare sayısını denetleyin."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text("Ses: ${done.audio}.", style = MaterialTheme.typography.bodyMedium)
                        for (note in done.notes) {
                            Text(note, style = MaterialTheme.typography.bodyMedium)
                        }
                        StatsText(done.stats)
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(top = 12.dp)) {
                            FilledTonalButton(onClick = onOpen) { Text("Aç") }
                            OutlinedButton(onClick = onShare) { Text("Paylaş") }
                        }
                    }
                }

                SectionCard(title = "Cihaz tanılaması") {
                    TextButton(onClick = onDiagnostics) { Text(if (diagnosticsOpen) "Gizle" else "VC-Tester denetimini çalıştır") }
                    if (diagnosticsOpen) {
                        if (diagnostics == null) {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(top = 4.dp))
                        } else {
                            for ((key, value) in diagnostics) {
                                Text(
                                    "$key: $value",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(top = 4.dp),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Header(coreInfo: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Image(
            painter = painterResource(R.drawable.logo),
            contentDescription = null,
            modifier = Modifier.size(56.dp).clip(RoundedCornerShape(14.dp)),
        )
        Column(modifier = Modifier.padding(start = 14.dp)) {
            Text("Video Cleaner", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            Text(coreInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun StatsText(stats: CoreStats) {
    Text(
        "Pencere: ${stats.windowsB} blok (11+4), ${stats.windowsA} hassas (7+1+7), ${stats.bridges} köprü · " +
            "sahne geçişi ${stats.sceneCuts} · halka ${stats.poolBytes / (1024 * 1024)} MiB",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(top = 8.dp),
    )
}

@Composable
private fun SectionCard(title: String, isError: Boolean = false, content: @Composable () -> Unit) {
    val colors = if (isError) {
        CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
    } else {
        CardDefaults.cardColors()
    }
    Card(colors = colors, shape = RoundedCornerShape(20.dp), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(4.dp))
            content()
        }
    }
}

@Preview(showBackground = true, widthDp = 380)
@Composable
private fun MainContentPreview() {
    MaterialTheme {
        MainContent(
            coreInfo = "Çekirdek v2 · ARM64 NEON · 16 KB sayfa",
            inputName = "ornek_720p60.mp4",
            info = VideoInfo(1280, 720, 0, 60_000_000, 60f, "video/avc", "audio/mp4a-latm", 4_000_000),
            preferHevc = false,
            highQuality = false,
            running = true,
            progress = JobProgress(
                framesDecoded = 930, framesEncoded = 900, positionUs = 15_000_000, durationUs = 60_000_000,
                framesPerSecond = 212.0, elapsedMs = 4_300, width = 1280, height = 720,
                stats = CoreStats(framesIn = 930, framesOut = 900, sceneCuts = 3, windowsB = 60, poolBytes = 30_412_800),
            ),
            result = null,
            error = null,
            onPick = {}, onCodec = {}, onQuality = {}, onStart = {}, onCancel = {}, onOpen = {}, onShare = {},
        )
    }
}
