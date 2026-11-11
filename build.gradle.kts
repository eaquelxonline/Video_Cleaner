// Video_Cleaner v2.0 — Android uygulaması (kök proje = uygulama modülü)
//
//   ./gradlew assembleDebug        Rust çekirdeğini (cargo-ndk) derler, 16 KB hizasını denetler, APK üretir
//   ./gradlew verifyApkAlignment   Üretilen APK'ların 16 KB ZIP hizasını zipalign ile denetler
//
// Gereksinimler: JDK 21, Android SDK (platform 37, NDK 30.0.16248370), Rust 1.99+,
//   rustup target add aarch64-linux-android x86_64-linux-android
//   cargo install cargo-ndk
//
// Kotlin: AGP 9'un yerleşik Kotlin desteği kullanılır; `org.jetbrains.kotlin.android`
// eklentisi BİLEREK uygulanmaz (AGP 9'da uygulanırsa derleme hata verir). Kotlin
// derleyici sürümünü aşağıdaki Compose derleyici eklentisinin sürümü belirler
// (AGP'nin getirdiği 2.2.10, bağımlılık çözümlemesinde 2.4.21'e yükselir).

import com.android.build.api.artifact.SingleArtifact
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import org.gradle.process.ExecOperations

plugins {
    id("com.android.application") version "9.5.1"
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.21"
}

val minAndroidApi = 30          // Android 11

android {
    namespace = "com.videocleaner.engine"
    compileSdk = 37              // Android 17
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "com.videocleaner.engine"
        minSdk = minAndroidApi
        targetSdk = 37
        versionCode = 1
        versionName = "2.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // Yalnızca 64 bit. Rust çekirdeği de tam olarak bu ABI'ler için derlenir.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildFeatures {
        compose = true
    }

    // Kilitli klasör yapısı: kaynaklar varsayılan src/main/... yerine kök dizindedir.
    // res/ yalnızca uygulama simgesini ve logoyu içerir; arayüz Compose ile, metinler Kotlin içindedir.
    sourceSets {
        named("main") {
            manifest.srcFile("AndroidManifest.xml")
            kotlin.directories += "Kotlin"
            res.directories += "res"
        }
        // Cihaz üstü testler ve test videoları: ./gradlew connectedDebugAndroidTest
        named("androidTest") {
            kotlin.directories += "AndroidTest"
            assets.directories += "AndroidTest/assets"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    packaging {
        jniLibs {
            // .so dosyaları APK içinde sıkıştırılmadan, sayfa hizalı durur ve doğrudan
            // APK'dan eşlenir (16 KB sayfalı cihazlar için zorunlu).
            useLegacyPackaging = false
        }
    }

    lint {
        // Uygulama kalıcı kullanıcı verisi tutmaz; allowBackup="false" yeterlidir.
        disable += "DataExtractionRules"
        // minSdk 30'da başlatıcı her zaman uyarlanabilir simgeyi (mipmap-anydpi) kullanır;
        // yuvarlak/kare PNG'ler simgeyi PNG olarak isteyen araçlar için bilerek tutulur.
        disable += "IconXmlAndPng"
    }
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.09.00"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}

// -----------------------------------------------------------------------------
// Rust çekirdeği: cargo-ndk ile derle, her .so için 16 KB ELF hizasını doğrula.
// -----------------------------------------------------------------------------

abstract class CargoNdkBuild @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sources: ConfigurableFileCollection

    @get:Input
    abstract val abis: ListProperty<String>

    @get:Input
    abstract val minApi: Property<Int>

    @get:Input
    abstract val ndkVersion: Property<String>

    @get:Internal
    abstract val ndkDirectory: DirectoryProperty

    @get:Internal
    abstract val crateDirectory: DirectoryProperty

    @get:Internal
    abstract val cargoTargetDirectory: DirectoryProperty

    @get:OutputDirectory
    abstract val jniLibs: DirectoryProperty

    @TaskAction
    fun build() {
        val ndk = ndkDirectory.get().asFile
        val out = jniLibs.get().asFile
        out.deleteRecursively()
        out.mkdirs()

        val command = listOf("cargo", "ndk", "--platform", minApi.get().toString()) +
            abis.get().flatMap { listOf("-t", it) } +
            listOf("-o", out.absolutePath, "build", "--release")
        try {
            exec.exec {
                workingDir = crateDirectory.get().asFile
                environment("ANDROID_NDK_HOME", ndk.absolutePath)
                environment("CARGO_TARGET_DIR", cargoTargetDirectory.get().asFile.absolutePath)
                commandLine(command)
            }
        } catch (e: Exception) {
            throw GradleException(
                "Rust çekirdeği derlenemedi (${command.joinToString(" ")}).\n" +
                    "Gerekenler: `cargo install cargo-ndk` ve " +
                    "`rustup target add aarch64-linux-android x86_64-linux-android`.",
                e,
            )
        }

        // 16 KB sayfa uyumluluğu: her LOAD segmentinin hizası en az 0x4000 olmalı.
        val prebuilt = ndk.resolve("toolchains/llvm/prebuilt").listFiles()?.firstOrNull { it.isDirectory }
            ?: throw GradleException("NDK araç zinciri bulunamadı: $ndk")
        val readelf = prebuilt.resolve("bin").listFiles()?.firstOrNull { it.nameWithoutExtension == "llvm-readelf" }
            ?: throw GradleException("llvm-readelf bulunamadı: $prebuilt")
        for (abi in abis.get()) {
            val library = out.resolve("$abi/libvideo_cleaner.so")
            if (!library.isFile) throw GradleException("Beklenen kitaplık üretilmedi: $library")
            val stdout = ByteArrayOutputStream()
            exec.exec {
                commandLine(readelf.absolutePath, "-lW", library.absolutePath)
                standardOutput = stdout
            }
            val alignments = stdout.toString().lineSequence()
                .map { it.trim() }
                .filter { it.startsWith("LOAD") }
                .map { it.split(Regex("\\s+")).last().removePrefix("0x").toLong(16) }
                .toList()
            if (alignments.isEmpty() || alignments.any { it < 16384 }) {
                throw GradleException("$abi/libvideo_cleaner.so 16 KB hizalı değil (LOAD hizaları: $alignments)")
            }
            logger.lifecycle("libvideo_cleaner.so [$abi]: ${library.length()} bayt, LOAD hizası ${alignments.min()} bayt (16 KB uyumlu)")
        }
    }
}

// -----------------------------------------------------------------------------
// APK denetimi: yerel kitaplıklar ZIP içinde 16 KB sınırına hizalı mı?
// -----------------------------------------------------------------------------

abstract class VerifyApkAlignment @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val apkDirectory: DirectoryProperty

    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val zipalign: RegularFileProperty

    @TaskAction
    fun verify() {
        val apks = apkDirectory.get().asFile.walkTopDown().filter { it.extension == "apk" }.toList()
        if (apks.isEmpty()) throw GradleException("Denetlenecek APK bulunamadı: ${apkDirectory.get().asFile}")
        for (apk in apks) {
            exec.exec {
                commandLine(zipalign.get().asFile.absolutePath, "-c", "-P", "16", "4", apk.absolutePath)
            }
            logger.lifecycle("${apk.name}: 16 KB ZIP hizası doğrulandı")
        }
    }
}

androidComponents {
    onVariants { variant ->
        val suffix = variant.name.replaceFirstChar { it.uppercase() }

        val rust = tasks.register<CargoNdkBuild>("buildRust$suffix") {
            group = "build"
            description = "Rust çekirdeğini ${variant.name} için derler ve 16 KB hizasını doğrular"
            sources.from(
                layout.projectDirectory.file("Cargo.toml"),
                layout.projectDirectory.dir("Rust"),
                layout.projectDirectory.dir("Assembly"),
            )
            abis.set(android.defaultConfig.ndk.abiFilters.sorted())
            minApi.set(minAndroidApi)
            ndkVersion.set(android.ndkVersion)
            ndkDirectory.set(sdkComponents.ndkDirectory)
            crateDirectory.set(layout.projectDirectory)
            cargoTargetDirectory.set(layout.buildDirectory.dir("rust-target"))
        }
        variant.sources.jniLibs?.addGeneratedSourceDirectory(rust, CargoNdkBuild::jniLibs)

        val verify = tasks.register<VerifyApkAlignment>("verify${suffix}ApkAlignment") {
            group = "verification"
            description = "${variant.name} APK'sının 16 KB ZIP hizasını zipalign ile denetler"
            apkDirectory.set(variant.artifacts.get(SingleArtifact.APK))
            zipalign.set(
                sdkComponents.sdkDirectory.map { sdk ->
                    val tools = sdk.asFile.resolve("build-tools").listFiles()?.filter { it.isDirectory }?.maxByOrNull { it.name }
                        ?: throw GradleException("SDK build-tools bulunamadı: ${sdk.asFile}")
                    val binary = tools.listFiles()?.firstOrNull { it.nameWithoutExtension == "zipalign" }
                        ?: throw GradleException("zipalign bulunamadı: $tools")
                    sdk.file(binary.relativeTo(sdk.asFile).path)
                },
            )
        }
        tasks.matching { it.name == "verifyApkAlignment" }.configureEach { dependsOn(verify) }
    }
}

tasks.register("verifyApkAlignment") {
    group = "verification"
    description = "Tüm varyantların APK'larında 16 KB ZIP hizasını denetler"
}
