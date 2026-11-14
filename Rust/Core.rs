//! # Video_Cleaner v2.0 — Rust çekirdeği
//!
//! Bu dosya çalışma zamanı motorunun model-dışı kısmının tamamıdır:
//!
//! | Bölüm            | Sorumluluk                                                        |
//! |------------------|-------------------------------------------------------------------|
//! | [`FrameRing`]    | 22 karelik NV12 belleği + kare sahipliği (yazma/okuma kiraları)   |
//! | [`Scheduler`]    | Hangi kare hangi pencerede girdi/çıktı olur (saf mantık, piksel yok) |
//! | [`Estimate`]     | Modelin ürettiği onarım tahmini (yalnızca ROI içinde geçerli)     |
//! | [`Pipeline`]     | Yukarıdakileri birleştiren kare giriş/çıkış hattı                 |
//! | [`kernels`]      | Assembly SIMD çekirdekleri + taşınabilir başvuru gerçeklemesi     |
//! | `jni`            | Kotlin köprüsü (harici crate yok, yalnızca doğrudan ByteBuffer)   |
//!
//! ## Pencere matematiği (kesin tanım)
//!
//! Model **her zaman** ardışık 15 kareyi görür (14 Kasım telefon ölçümünden sonra 30 → 15; ara bellek
//! 30 karede 526 MB'tı). İki mod yalnızca *hangi çıktının kullanıldığını* ve *pencerenin ne kadar
//! kaydığını* değiştirir. `e` = sıradaki kesinleşecek karenin sıra numarası olsun:
//!
//! * **Mod B (11+4)** — `B(j)`, `0 ≤ j ≤ 11`: pencere `[e+j-11, e+j+4)`.
//!   Konum `[11-j, 11)` kareleri kesinleşir, konum `[11, 15)` tahminleri bir sonraki
//!   pencereyle harmanlanmak üzere önbelleğe alınır. Olağan akışta `j = 11`:
//!   pencereler `[0,15) → [11,26) → [22,37)…` şeklinde 11 kare kayar; ardışık iki pencere
//!   4 karede örtüşür ve bu 4 karenin iki tahmini `öncelik × güven` ile harmanlanır.
//!   `j = 0` hiçbir kareyi kesinleştirmeyen "köprü" geçişidir (yalnızca önbellek üretir).
//! * **Mod A (7+1+7 hassas)**: pencere `[e-7, e+8)`, yalnızca konum 7 (= `e`)
//!   kesinleşir, pencere 1 kare kayar.
//!
//! **Kanıtlanabilir** güvence (`scheduler_random_streams_keep_invariants` testiyle denetlenir):
//! Mod B'de kesinleşen her kare en az `min(4, sahne başına uzaklık)` geçmiş ve
//! `min(4, sahne sonuna uzaklık)` gelecek kareye erişmiştir (harmanlanan 4 kare iki tahmini
//! üzerinden 11 geçmiş görür); Mod A karesi `min(7, …)` geçmiş ve gelecek. Pencereler sahne
//! geçişini asla aşmaz; eksik konumlar "geçersiz" (dolgu) olarak işaretlenir ve modele
//! geçerlilik maskesiyle bildirilir. Halka kapasitesi 22 karedir (11 geçmiş + 11 karar ufku).
//!
//! ## Dürüst sınırlar
//!
//! * Halka belleği sayfa hizalıdır; bu onu "pinned" veya GPU'ya doğrudan görünür
//!   **yapmaz**. Sıfır kopya hattı AHardwareBuffer/Vulkan aşamasının işidir.
//! * Maskesiz piksellerin bit düzeyinde korunması **kodlayıcıdan önce** garanti
//!   edilir. Kayıplı (H.264/HEVC) yeniden kodlama bu garantiyi doğası gereği bozar.
//! * NV12'de renk 2×2 blok başına tek örnektir: Y pikseli tek tek, UV ise içinde
//!   hiç maskeli piksel bulunmayan bloklar için bit düzeyinde korunur.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]

use std::alloc::{self, Layout};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard};

// ============================================================================
// 1. Sabitler ve hata türü
// ============================================================================

/// Modelin tek geçişte gördüğü kare sayısı (Models/Core.py `WINDOW`).
pub const WINDOW_FRAMES: usize = 15;
/// Mod B'nin olağan kayma miktarı (Models/Core.py `STRIDE`).
pub const STRIDE: usize = 11;
/// Ardışık iki Mod B penceresinin ortak kare sayısı; bu karelerin iki tahmini harmanlanır.
pub const OVERLAP: usize = WINDOW_FRAMES - STRIDE;
/// Halka kapasitesi: karar anında `[e-11, e+11)` — köprü penceresi için 11 geçmiş kare + 11 karelik
/// karar ufku. Pencereden büyüktür (15 < 22); halka yalnızca NV12 kareleri tutar (720p'de ~30 MB).
pub const RING_FRAMES: usize = 2 * STRIDE;
/// Mod A'da hedef karenin pencere içindeki konumu (7 geçmiş + hedef + 7 gelecek).
pub const MODE_A_TARGET: usize = 7;
/// Bu sayıdan az geçerli kare içeren pencereler "düşük bağlam" olarak işaretlenir (15'in üçte biri).
pub const LOW_CONTEXT_FRAMES: u32 = 5;
/// Kotlin köprüsüyle uyum denetimi için sürüm numarası.
pub const ABI_VERSION: i32 = 3; // 3: pencere 15 kare, Mod B 11+4, Mod A 7+1+7, halka 22

/// Kare bayrağı: bu kare yeni bir sahnenin ilk karesidir.
pub const FRAME_FLAG_SCENE_CUT: u32 = 1 << 0;
/// Kare bayrağı: kaynakta anahtar kare (yalnızca bilgi amaçlı).
pub const FRAME_FLAG_KEYFRAME: u32 = 1 << 1;

/// Plan bayrağı: pencere sahnenin ilk karesini içeriyor (geçmiş tarafı dolgu olabilir).
pub const PLAN_FLAG_SEGMENT_START: u32 = 1 << 0;
/// Plan bayrağı: pencere sahnenin son karesine dayanıyor (gelecek tarafı dolgu olabilir).
pub const PLAN_FLAG_SEGMENT_END: u32 = 1 << 1;
/// Plan bayrağı: geçerli kare sayısı [`LOW_CONTEXT_FRAMES`] altında.
pub const PLAN_FLAG_LOW_CONTEXT: u32 = 1 << 2;
/// Plan bayrağı: hiçbir kareyi kesinleştirmeyen köprü geçişi (`B(0)`).
pub const PLAN_FLAG_BRIDGE: u32 = 1 << 3;

/// Çıktı bayrağı: kare bir model tahminiyle üretildi (maske boşsa piksel değişmemiş olabilir).
pub const OUT_FLAG_RESTORED: u32 = 1 << 0;
/// Çıktı bayrağı: Mod A (7+1+7) ile üretildi.
pub const OUT_FLAG_MODE_A: u32 = 1 << 1;
/// Çıktı bayrağı: iki örtüşen pencerenin tahmini harmanlandı.
pub const OUT_FLAG_FUSED: u32 = 1 << 2;
/// Çıktı bayrağı: kare modele sokulmadan aynen geçirildi.
pub const OUT_FLAG_BYPASS: u32 = 1 << 3;
/// Çıktı bayrağı: düşük bağlamlı pencereden üretildi (kısa sahne).
pub const OUT_FLAG_LOW_CONTEXT: u32 = 1 << 4;
/// Çıktı bayrağı: sahnenin ilk karesi.
pub const OUT_FLAG_SCENE_START: u32 = 1 << 5;

/// Çekirdeğin tüm hata durumları. Sayısal değerler JNI üzerinden aynen döner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CoreError {
    /// Genişlik/yükseklik çift değil veya desteklenen aralığın dışında.
    InvalidGeometry = -1,
    /// Bellek ayrılamadı.
    OutOfMemory = -2,
    /// Halkada boş yer yok (22 kare dolu).
    RingFull = -3,
    /// Yazılacak yuva hâlâ bir okuyucu tarafından tutuluyor.
    SlotBusy = -4,
    /// Aynı anda ikinci bir yazma kirası istendi.
    WriteInProgress = -5,
    /// Zaman damgası bir önceki kareden büyük değil.
    NonMonotonicPts = -6,
    /// İstenen kare halkada yok (hiç gelmedi veya bırakıldı).
    UnknownFrame = -7,
    /// Çağrı o anki durumda geçerli değil.
    BadState = -8,
    /// Geçersiz argüman.
    BadArgument = -9,
    /// Verilen tampon çok küçük.
    BufferTooSmall = -10,
    /// Akış bitti; yeni kare kabul edilmiyor.
    StreamFinished = -11,
    /// Geçersiz veya serbest bırakılmış tutamak.
    InvalidHandle = -12,
    /// Beklenmeyen iç hata (yakalanan panik).
    Internal = -13,
    /// Bu cihazda/biçimde desteklenmiyor (ör. donanım tamponu CPU için kilitlenemiyor); çağıran
    /// başka bir yola (CPU düzlemleri) düşmelidir.
    Unsupported = -14,
}

impl CoreError {
    /// JNI sınırında kullanılan negatif hata kodu.
    pub const fn code(self) -> i32 {
        self as i32
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            CoreError::InvalidGeometry => "geçersiz kare boyutu",
            CoreError::OutOfMemory => "bellek ayrılamadı",
            CoreError::RingFull => "halka dolu",
            CoreError::SlotBusy => "yuva bir okuyucu tarafından tutuluyor",
            CoreError::WriteInProgress => "süren bir yazma kirası var",
            CoreError::NonMonotonicPts => "zaman damgası artan sırada değil",
            CoreError::UnknownFrame => "kare halkada bulunamadı",
            CoreError::BadState => "çağrı bu durumda geçerli değil",
            CoreError::BadArgument => "geçersiz argüman",
            CoreError::BufferTooSmall => "tampon çok küçük",
            CoreError::StreamFinished => "akış sonlandı",
            CoreError::InvalidHandle => "geçersiz tutamak",
            CoreError::Internal => "iç hata",
            CoreError::Unsupported => "bu cihazda desteklenmiyor",
        };
        write!(f, "{text} ({})", self.code())
    }
}

impl std::error::Error for CoreError {}

/// Zehirlenmiş kilidi de kabul eden kilitleme (durum tabloları her adımda tutarlıdır).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Sürüm metni.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// 2. Kare geometrisi ve hizalı bellek
// ============================================================================

/// NV12 (4:2:0, Y düzlemi + iç içe UV düzlemi) kare boyutu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGeometry {
    width: u32,
    height: u32,
}

impl FrameGeometry {
    /// Projenin hedef çözünürlüğü: 1280×720.
    pub const HD720: FrameGeometry = FrameGeometry {
        width: 1280,
        height: 720,
    };
    /// Desteklenen en büyük kenar uzunluğu.
    pub const MAX_DIM: usize = 8192;

    /// Genişlik ve yükseklik çift olmalıdır (4:2:0 örnekleme).
    pub fn new(width: usize, height: usize) -> Result<Self, CoreError> {
        let ok = |v: usize| (2..=Self::MAX_DIM).contains(&v) && v.is_multiple_of(2);
        if ok(width) && ok(height) {
            Ok(Self {
                width: width as u32,
                height: height as u32,
            })
        } else {
            Err(CoreError::InvalidGeometry)
        }
    }

    /// Piksel cinsinden genişlik.
    pub const fn width(self) -> usize {
        self.width as usize
    }

    /// Piksel cinsinden yükseklik.
    pub const fn height(self) -> usize {
        self.height as usize
    }

    /// Y düzleminin bayt sayısı (aynı zamanda UV düzleminin başlangıç ofseti).
    pub const fn luma_bytes(self) -> usize {
        self.width() * self.height()
    }

    /// İç içe UV düzleminin bayt sayısı.
    pub const fn chroma_bytes(self) -> usize {
        self.luma_bytes() / 2
    }

    /// Bir NV12 karesinin toplam bayt sayısı (genişlik × yükseklik × 1,5).
    pub const fn frame_bytes(self) -> usize {
        self.luma_bytes() + self.chroma_bytes()
    }
}

/// İşletim sisteminin bellek sayfası boyutu (4 KB veya 16 KB). Derleme zamanında
/// sabitlenmez: aynı ikili dosya her iki sayfa boyutunda da çalışır.
pub fn page_size() -> usize {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn getpagesize() -> std::ffi::c_int;
        }
        // SAFETY: getpagesize() ön koşulu olmayan, yan etkisiz bir libc çağrısıdır.
        let size = unsafe { getpagesize() };
        if size > 0 {
            return size as usize;
        }
    }
    4096
}

const fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

/// Sıfırlanmış, hizalı tek parça bellek.
struct AlignedBlock {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBlock {
    fn zeroed(size: usize, align: usize) -> Result<Self, CoreError> {
        if size == 0 {
            return Err(CoreError::OutOfMemory);
        }
        let layout = Layout::from_size_align(size, align).map_err(|_| CoreError::OutOfMemory)?;
        // SAFETY: layout sıfırdan büyük boyutludur.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        NonNull::new(raw)
            .map(|ptr| Self { ptr, layout })
            .ok_or(CoreError::OutOfMemory)
    }
}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        // SAFETY: ptr bu layout ile alloc_zeroed tarafından ayrıldı ve bir kez serbest bırakılıyor.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: AlignedBlock sahip olunan düz bellektir (Box<[u8]> gibi). İçeriğe erişim
// FrameRing'in kira durum makinesi tarafından eşzamanlanır.
unsafe impl Send for AlignedBlock {}
unsafe impl Sync for AlignedBlock {}

// ============================================================================
// 3. FrameRing — 22 karelik bellek ve kare sahipliği
// ============================================================================

/// Halkadaki bir karenin kimliği ve zaman bilgisi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    /// Gösterim sırasındaki kesintisiz kare numarası (0'dan başlar).
    pub seq: u64,
    /// Kaynak dosyadaki sunum zaman damgası (mikrosaniye). Aynen korunur.
    pub pts_us: i64,
    /// `FRAME_FLAG_*` bitleri.
    pub flags: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Boş; yazılabilir.
    Free,
    /// Tek bir [`WriteLease`] tarafından dolduruluyor.
    Writing,
    /// Değişmez; okunabilir.
    Ready,
    /// Halkadan çıkarıldı ama okuyucusu var; son okuyucu bırakınca `Free` olur.
    Retired,
}

#[derive(Debug, Clone, Copy)]
struct SlotInfo {
    state: SlotState,
    readers: u32,
    meta: FrameMeta,
}

struct RingTable {
    slots: [SlotInfo; RING_FRAMES],
    /// Halkadaki en eski karenin sıra numarası.
    head: u64,
    /// Bir sonraki yazılacak karenin sıra numarası.
    tail: u64,
    last_pts: Option<i64>,
    writing: bool,
}

/// 22 karelik NV12 halka belleği.
///
/// Tek bir sayfa hizalı blok ayrılır (720p için tam 41.472.000 bayt). Kare `seq`,
/// `seq % 30` numaralı yuvada durur. Yuva belleğine yalnızca kiralar üzerinden
/// erişilir:
///
/// * [`WriteLease`] — aynı anda en çok bir tane; yuvaya özel yazma erişimi.
/// * [`ReadLease`] — sınırsız sayıda; paylaşımlı salt-okunur erişim.
///
/// Okuyucusu olan bir yuva halkadan çıkarılsa bile üzerine yazılamaz
/// ([`CoreError::SlotBusy`]); bu yüzden yapı `Sync`'tir ve çözücü, çıkarım ve
/// kodlayıcı iş parçacıkları arasında güvenle paylaşılabilir.
/// Halka belleğinin nerede durduğu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    /// Süreç yığınında tek parça, sayfa hizalı blok (masaüstü, test ve geri dönüş yolu).
    Heap,
    /// Android: her yuva ayrı bir `AHardwareBuffer` (BLOB; CPU okuma/yazma + GPU veri tamponu
    /// kullanımıyla ayrılır, Vulkan dış bellek olarak içe aktarılabilir). Yuvalar ömür boyu
    /// CPU için kilitli tutulur; Java yığınına kopya yoktur.
    HardwareBuffer,
}

/// Halka belleği: yığın bloğu veya (Android'de) yuva başına bir donanım tamponu.
enum RingStorage {
    Heap(AlignedBlock),
    #[cfg(target_os = "android")]
    Hardware(hwbuf::SlotBuffers),
}

impl RingStorage {
    fn kind(&self) -> StorageKind {
        match self {
            RingStorage::Heap(_) => StorageKind::Heap,
            #[cfg(target_os = "android")]
            RingStorage::Hardware(_) => StorageKind::HardwareBuffer,
        }
    }

    fn slot_base(&self, slot: usize, stride: usize) -> *mut u8 {
        match self {
            // SAFETY: slot < 30 olduğundan ofset ayrılan bloğun içindedir.
            RingStorage::Heap(block) => unsafe { block.ptr.as_ptr().add(slot * stride) },
            #[cfg(target_os = "android")]
            RingStorage::Hardware(slots) => slots.ptr(slot),
        }
    }
}

pub struct FrameRing {
    geo: FrameGeometry,
    stride: usize,
    storage: RingStorage,
    table: Mutex<RingTable>,
}

impl FrameRing {
    /// Verilen geometri için 30 yuvalı halka ayırır (yığın belleği).
    pub fn new(geo: FrameGeometry) -> Result<Self, CoreError> {
        Self::with_storage(geo, StorageKind::Heap)
    }

    /// İstenen bellek türüyle halka ayırır. Donanım tamponu ayrılamazsa (Android dışı, sürücü
    /// reddi, hizasız adres) yığın belleğine düşülür; gerçekte kullanılan tür [`Self::storage_kind`].
    pub fn with_storage(geo: FrameGeometry, kind: StorageKind) -> Result<Self, CoreError> {
        let stride = round_up(geo.frame_bytes(), 64);
        let total = stride
            .checked_mul(RING_FRAMES)
            .ok_or(CoreError::OutOfMemory)?;
        let storage = match kind {
            #[cfg(target_os = "android")]
            StorageKind::HardwareBuffer => {
                match hwbuf::SlotBuffers::allocate(stride, RING_FRAMES) {
                    Some(slots) => RingStorage::Hardware(slots),
                    None => RingStorage::Heap(AlignedBlock::zeroed(total, page_size().max(64))?),
                }
            }
            _ => RingStorage::Heap(AlignedBlock::zeroed(total, page_size().max(64))?),
        };
        let empty = SlotInfo {
            state: SlotState::Free,
            readers: 0,
            meta: FrameMeta {
                seq: 0,
                pts_us: 0,
                flags: 0,
            },
        };
        Ok(Self {
            geo,
            stride,
            storage,
            table: Mutex::new(RingTable {
                slots: [empty; RING_FRAMES],
                head: 0,
                tail: 0,
                last_pts: None,
                writing: false,
            }),
        })
    }

    /// Kare geometrisi.
    pub fn geometry(&self) -> FrameGeometry {
        self.geo
    }

    /// Ayrılan toplam bayt sayısı.
    pub fn pool_bytes(&self) -> usize {
        self.stride * RING_FRAMES
    }

    /// İlk yuvanın adresi (hizalama testleri ve tanılama için).
    pub fn base_addr(&self) -> usize {
        self.slot_ptr(0) as usize
    }

    /// Gerçekte kullanılan bellek türü.
    pub fn storage_kind(&self) -> StorageKind {
        self.storage.kind()
    }

    /// Donanım tamponu GPU tarafından da (Vulkan dış bellek) kullanılabilir mi?
    pub fn gpu_importable(&self) -> bool {
        match &self.storage {
            #[cfg(target_os = "android")]
            RingStorage::Hardware(slots) => slots.usage & hwbuf::USAGE_GPU_DATA_BUFFER != 0,
            _ => false,
        }
    }

    /// `[en eski kare, bir sonraki kare)` sıra numarası aralığı.
    pub fn bounds(&self) -> (u64, u64) {
        let t = lock(&self.table);
        (t.head, t.tail)
    }

    /// Halkadaki canlı kare sayısı.
    pub fn len(&self) -> usize {
        let (head, tail) = self.bounds();
        (tail - head) as usize
    }

    /// Halka boş mu?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slot_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < RING_FRAMES);
        self.storage.slot_base(slot, self.stride)
    }

    /// Bir sonraki kare için yazma kirası alır. Kira `commit` edilmeden düşerse
    /// yuva boşa döner ve hiçbir kare eklenmez.
    pub fn begin_write(&self) -> Result<WriteLease<'_>, CoreError> {
        let mut t = lock(&self.table);
        if t.writing {
            return Err(CoreError::WriteInProgress);
        }
        if t.tail - t.head >= RING_FRAMES as u64 {
            return Err(CoreError::RingFull);
        }
        let slot = (t.tail % RING_FRAMES as u64) as usize;
        if t.slots[slot].state != SlotState::Free {
            return Err(CoreError::SlotBusy);
        }
        t.slots[slot].state = SlotState::Writing;
        t.writing = true;
        Ok(WriteLease {
            ring: self,
            slot,
            active: true,
        })
    }

    fn commit(&self, slot: usize, pts_us: i64, flags: u32) -> Result<FrameMeta, CoreError> {
        let mut t = lock(&self.table);
        debug_assert_eq!(t.slots[slot].state, SlotState::Writing);
        t.writing = false;
        if t.last_pts.is_some_and(|last| pts_us <= last) {
            t.slots[slot].state = SlotState::Free;
            return Err(CoreError::NonMonotonicPts);
        }
        let meta = FrameMeta {
            seq: t.tail,
            pts_us,
            flags,
        };
        t.slots[slot] = SlotInfo {
            state: SlotState::Ready,
            readers: 0,
            meta,
        };
        t.tail += 1;
        t.last_pts = Some(pts_us);
        Ok(meta)
    }

    fn abort(&self, slot: usize) {
        let mut t = lock(&self.table);
        t.slots[slot].state = SlotState::Free;
        t.writing = false;
    }

    /// Halkadaki bir kare için paylaşımlı okuma kirası alır.
    pub fn read(&self, seq: u64) -> Result<ReadLease<'_>, CoreError> {
        let mut t = lock(&self.table);
        if seq < t.head || seq >= t.tail {
            return Err(CoreError::UnknownFrame);
        }
        let slot = (seq % RING_FRAMES as u64) as usize;
        let info = &mut t.slots[slot];
        debug_assert!(info.state == SlotState::Ready && info.meta.seq == seq);
        info.readers += 1;
        Ok(ReadLease {
            ring: self,
            slot,
            meta: info.meta,
        })
    }

    fn release_read(&self, slot: usize) {
        let mut t = lock(&self.table);
        let info = &mut t.slots[slot];
        debug_assert!(info.readers > 0);
        info.readers -= 1;
        if info.readers == 0 && info.state == SlotState::Retired {
            info.state = SlotState::Free;
        }
    }

    /// Halkadaki bir karenin bilgisi (kira almadan).
    pub fn meta(&self, seq: u64) -> Option<FrameMeta> {
        let t = lock(&self.table);
        (seq >= t.head && seq < t.tail).then(|| t.slots[(seq % RING_FRAMES as u64) as usize].meta)
    }

    /// `seq`'ten küçük tüm kareleri halkadan çıkarır; çıkarılan kare sayısını döndürür.
    pub fn retire_before(&self, seq: u64) -> usize {
        let mut t = lock(&self.table);
        let target = seq.min(t.tail);
        let mut released = 0;
        while t.head < target {
            let slot = (t.head % RING_FRAMES as u64) as usize;
            let info = &mut t.slots[slot];
            info.state = if info.readers == 0 {
                SlotState::Free
            } else {
                SlotState::Retired
            };
            t.head += 1;
            released += 1;
        }
        released
    }
}

/// Bir yuvaya özel yazma erişimi. `commit` ile kare halkaya eklenir.
pub struct WriteLease<'r> {
    ring: &'r FrameRing,
    slot: usize,
    active: bool,
}

impl WriteLease<'_> {
    /// Yuvanın NV12 belleği (`frame_bytes()` uzunluğunda).
    pub fn data(&mut self) -> &mut [u8] {
        // SAFETY: Yuva `Writing` durumundadır ve bu durum için tek bir WriteLease
        // vardır (`writing` bayrağı). `read()` yalnızca [head, tail) aralığındaki
        // `Ready` karelere kira verir; bu yuva tail konumunda olduğundan ve eski
        // okuyucusu varken `begin_write` SlotBusy döndürdüğünden, başka hiçbir
        // referans bu belleği göremez.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ring.slot_ptr(self.slot),
                self.ring.geo.frame_bytes(),
            )
        }
    }

    /// Kareyi zaman damgasıyla halkaya ekler. Damga artan sırada değilse kare reddedilir.
    pub fn commit(mut self, pts_us: i64, flags: u32) -> Result<FrameMeta, CoreError> {
        self.active = false;
        self.ring.commit(self.slot, pts_us, flags)
    }
}

impl Drop for WriteLease<'_> {
    fn drop(&mut self) {
        if self.active {
            self.ring.abort(self.slot);
        }
    }
}

/// Bir kareye paylaşımlı salt-okunur erişim. Düştüğünde yuva serbest kalır.
pub struct ReadLease<'r> {
    ring: &'r FrameRing,
    slot: usize,
    meta: FrameMeta,
}

impl ReadLease<'_> {
    /// Karenin tamamı (NV12).
    pub fn data(&self) -> &[u8] {
        // SAFETY: Yuvanın okuyucu sayacı bu kira yaşadığı sürece > 0'dır; bu durumda
        // yuva `Writing` durumuna geçemez, dolayısıyla bellek değişmezdir.
        unsafe {
            std::slice::from_raw_parts(self.ring.slot_ptr(self.slot), self.ring.geo.frame_bytes())
        }
    }

    /// Y düzlemi.
    pub fn luma(&self) -> &[u8] {
        &self.data()[..self.ring.geo.luma_bytes()]
    }

    /// İç içe UV düzlemi.
    pub fn chroma(&self) -> &[u8] {
        &self.data()[self.ring.geo.luma_bytes()..]
    }

    /// Karenin kimliği ve zaman damgası.
    pub fn meta(&self) -> FrameMeta {
        self.meta
    }
}

impl Drop for ReadLease<'_> {
    fn drop(&mut self) {
        self.ring.release_read(self.slot);
    }
}

// ============================================================================
// 3b. Android donanım tamponları (AHardwareBuffer, NDK API 26+; lockPlanes API 29+)
// ============================================================================

#[cfg(target_os = "android")]
pub mod hwbuf {
    use std::ffi::c_void;

    /// `AHardwareBuffer` (opak).
    #[repr(C)]
    pub struct AHardwareBuffer {
        _private: [u8; 0],
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct Desc {
        pub width: u32,
        pub height: u32,
        pub layers: u32,
        pub format: u32,
        pub usage: u64,
        pub stride: u32,
        pub rfu0: u32,
        pub rfu1: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Plane {
        pub data: *mut c_void,
        pub pixel_stride: u32,
        pub row_stride: u32,
    }

    #[repr(C)]
    pub struct Planes {
        pub plane_count: u32,
        pub planes: [Plane; 4],
    }

    pub const FORMAT_BLOB: u32 = 0x21;
    pub const FORMAT_Y8CB8CR8_420: u32 = 0x23;
    pub const USAGE_CPU_READ_OFTEN: u64 = 3;
    pub const USAGE_CPU_WRITE_OFTEN: u64 = 3 << 4;
    pub const USAGE_GPU_DATA_BUFFER: u64 = 1 << 24;

    #[link(name = "nativewindow")]
    unsafe extern "C" {
        fn AHardwareBuffer_allocate(desc: *const Desc, out: *mut *mut AHardwareBuffer) -> i32;
        fn AHardwareBuffer_release(buffer: *mut AHardwareBuffer);
        pub fn AHardwareBuffer_describe(buffer: *const AHardwareBuffer, out: *mut Desc);
        fn AHardwareBuffer_lock(
            buffer: *mut AHardwareBuffer,
            usage: u64,
            fence: i32,
            rect: *const c_void,
            out: *mut *mut c_void,
        ) -> i32;
        pub fn AHardwareBuffer_lockPlanes(
            buffer: *mut AHardwareBuffer,
            usage: u64,
            fence: i32,
            rect: *const c_void,
            out: *mut Planes,
        ) -> i32;
        pub fn AHardwareBuffer_unlock(buffer: *mut AHardwareBuffer, fence: *mut i32) -> i32;
    }

    #[link(name = "android")]
    unsafe extern "C" {
        /// `android.hardware.HardwareBuffer` → `AHardwareBuffer*` (başvuru sayısı artmaz).
        pub fn AHardwareBuffer_fromHardwareBuffer(
            env: *mut c_void,
            object: *mut c_void,
        ) -> *mut AHardwareBuffer;
    }

    /// Halkanın 30 yuvası: her biri ayrı BLOB donanım tamponu, ömür boyu CPU için kilitli.
    pub struct SlotBuffers {
        slots: Vec<(*mut AHardwareBuffer, *mut u8)>,
        /// Ayırmada kullanılan kullanım bayrakları (GPU veri tamponu dahil mi?).
        pub usage: u64,
    }

    // SAFETY: tamponlar bu yapıya aittir; içeriğe erişim FrameRing'in kira durum makinesiyle eşzamanlanır.
    unsafe impl Send for SlotBuffers {}
    unsafe impl Sync for SlotBuffers {}

    impl SlotBuffers {
        /// `count` adet `bytes` boyutlu BLOB tamponu ayırıp kilitler. Herhangi biri başarısız olursa
        /// (sürücü BLOB+GPU kullanımını desteklemiyor, bellek yok, adres 64 bayta hizalı değil) `None`.
        pub fn allocate(bytes: usize, count: usize) -> Option<Self> {
            // Önce Vulkan'ın içe aktarabileceği biçimde (GPU veri tamponu), olmazsa yalnızca CPU.
            let cpu = USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN;
            Self::allocate_with(bytes, count, cpu | USAGE_GPU_DATA_BUFFER)
                .or_else(|| Self::allocate_with(bytes, count, cpu))
        }

        fn allocate_with(bytes: usize, count: usize, usage_flags: u64) -> Option<Self> {
            let mut out = Self {
                slots: Vec::with_capacity(count),
                usage: usage_flags,
            };
            for _ in 0..count {
                let desc = Desc {
                    width: u32::try_from(bytes).ok()?,
                    height: 1,
                    layers: 1,
                    format: FORMAT_BLOB,
                    usage: usage_flags,
                    ..Desc::default()
                };
                let mut buffer = std::ptr::null_mut();
                // SAFETY: desc geçerli; çıktı işaretçisi yerel değişkendir.
                if unsafe { AHardwareBuffer_allocate(&desc, &mut buffer) } != 0 || buffer.is_null()
                {
                    return None;
                }
                let mut ptr = std::ptr::null_mut();
                let usage = USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN;
                // SAFETY: buffer az önce ayrıldı; çit yok (-1), tüm tampon.
                let locked =
                    unsafe { AHardwareBuffer_lock(buffer, usage, -1, std::ptr::null(), &mut ptr) };
                if locked != 0 || ptr.is_null() || !(ptr as usize).is_multiple_of(64) {
                    // SAFETY: kilitlenemeyen tamponu bırak.
                    unsafe {
                        if locked == 0 {
                            AHardwareBuffer_unlock(buffer, std::ptr::null_mut());
                        }
                        AHardwareBuffer_release(buffer);
                    }
                    return None;
                }
                // SAFETY: yeni kilitlenen bellek yazılabilir ve `bytes` uzunluğundadır.
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
                out.slots.push((buffer, ptr as *mut u8));
            }
            Some(out)
        }

        pub fn ptr(&self, slot: usize) -> *mut u8 {
            self.slots[slot].1
        }
    }

    impl Drop for SlotBuffers {
        fn drop(&mut self) {
            for &(buffer, _) in &self.slots {
                // SAFETY: her tampon bir kez kilitlendi ve bir kez bırakılıyor.
                unsafe {
                    AHardwareBuffer_unlock(buffer, std::ptr::null_mut());
                    AHardwareBuffer_release(buffer);
                }
            }
        }
    }
}

// ============================================================================
// 4. SIMD çekirdekleri
// ============================================================================

/// Assembly çekirdekleri ve bunların taşınabilir başvuru gerçeklemesi.
///
/// Açık işlevler dilimler üzerinde çalışır: tam 32 baytlık bloklar (varsa)
/// assembly çekirdeğine, kalan kuyruk [`portable`] yoluna gider. Her iki yol bit
/// düzeyinde aynı sonucu üretir (`kernels_match_portable_reference` testi).
pub mod kernels {
    use std::sync::atomic::{AtomicU8, Ordering};

    #[cfg(all(
        target_arch = "x86_64",
        any(target_os = "linux", target_os = "android")
    ))]
    mod native {
        core::arch::global_asm!(include_str!("../Assembly/x86_64.asm"), options(raw));

        pub const NAME: &str = "x86_64-avx2";
        pub const ID: i32 = 1;

        pub fn available() -> bool {
            std::is_x86_feature_detected!("avx2")
        }

        unsafe extern "C" {
            #[link_name = "vc_sad_u8_avx2"]
            pub fn sad_u8(a: *const u8, b: *const u8, blocks: usize) -> u64;
            #[link_name = "vc_select_u8_avx2"]
            pub fn select_u8(
                dst: *mut u8,
                restored: *const u8,
                original: *const u8,
                mask: *const u8,
                blocks: usize,
            );
            #[link_name = "vc_haar_fwd_u8_avx2"]
            pub fn haar_fwd(
                row0: *const u8,
                row1: *const u8,
                ll: *mut i16,
                hl: *mut i16,
                lh: *mut i16,
                hh: *mut i16,
                blocks: usize,
            );
            #[link_name = "vc_haar_inv_i16_avx2"]
            pub fn haar_inv(
                ll: *const i16,
                hl: *const i16,
                lh: *const i16,
                hh: *const i16,
                row0: *mut u8,
                row1: *mut u8,
                blocks: usize,
            );
            #[link_name = "vc_max5_u8_avx2"]
            pub fn max5_u8(
                dst: *mut u8,
                r0: *const u8,
                r1: *const u8,
                r2: *const u8,
                r3: *const u8,
                r4: *const u8,
                blocks: usize,
            );
            #[link_name = "vc_unblend_u16_avx2"]
            pub fn unblend_u16(
                dst: *mut u16,
                img: *const u16,
                fg: *const u16,
                alpha: *const u8,
                blocks: usize,
            );
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    ))]
    mod native {
        core::arch::global_asm!(include_str!("../Assembly/Arm64.S"), options(raw));

        pub const NAME: &str = "arm64-neon";
        pub const ID: i32 = 2;

        /// NEON, AArch64'te zorunludur.
        pub fn available() -> bool {
            true
        }

        unsafe extern "C" {
            #[link_name = "vc_sad_u8_neon"]
            pub fn sad_u8(a: *const u8, b: *const u8, blocks: usize) -> u64;
            #[link_name = "vc_select_u8_neon"]
            pub fn select_u8(
                dst: *mut u8,
                restored: *const u8,
                original: *const u8,
                mask: *const u8,
                blocks: usize,
            );
            #[link_name = "vc_haar_fwd_u8_neon"]
            pub fn haar_fwd(
                row0: *const u8,
                row1: *const u8,
                ll: *mut i16,
                hl: *mut i16,
                lh: *mut i16,
                hh: *mut i16,
                blocks: usize,
            );
            #[link_name = "vc_haar_inv_i16_neon"]
            pub fn haar_inv(
                ll: *const i16,
                hl: *const i16,
                lh: *const i16,
                hh: *const i16,
                row0: *mut u8,
                row1: *mut u8,
                blocks: usize,
            );
            #[link_name = "vc_max5_u8_neon"]
            pub fn max5_u8(
                dst: *mut u8,
                r0: *const u8,
                r1: *const u8,
                r2: *const u8,
                r3: *const u8,
                r4: *const u8,
                blocks: usize,
            );
            #[link_name = "vc_unblend_u16_neon"]
            pub fn unblend_u16(
                dst: *mut u16,
                img: *const u16,
                fg: *const u16,
                alpha: *const u8,
                blocks: usize,
            );
        }
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            any(target_os = "linux", target_os = "android")
        ),
        all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "android")
        )
    )))]
    mod native {
        pub const NAME: &str = "portable";
        pub const ID: i32 = 0;

        pub fn available() -> bool {
            false
        }

        pub unsafe fn sad_u8(_: *const u8, _: *const u8, _: usize) -> u64 {
            unreachable!()
        }
        pub unsafe fn select_u8(_: *mut u8, _: *const u8, _: *const u8, _: *const u8, _: usize) {
            unreachable!()
        }
        pub unsafe fn haar_fwd(
            _: *const u8,
            _: *const u8,
            _: *mut i16,
            _: *mut i16,
            _: *mut i16,
            _: *mut i16,
            _: usize,
        ) {
            unreachable!()
        }
        pub unsafe fn haar_inv(
            _: *const i16,
            _: *const i16,
            _: *const i16,
            _: *const i16,
            _: *mut u8,
            _: *mut u8,
            _: usize,
        ) {
            unreachable!()
        }
        pub unsafe fn max5_u8(
            _: *mut u8,
            _: *const u8,
            _: *const u8,
            _: *const u8,
            _: *const u8,
            _: *const u8,
            _: usize,
        ) {
            unreachable!()
        }
        pub unsafe fn unblend_u16(
            _: *mut u16,
            _: *const u16,
            _: *const u16,
            _: *const u8,
            _: usize,
        ) {
            unreachable!()
        }
    }

    /// 0: henüz belirlenmedi, 1: assembly, 2: taşınabilir.
    static MODE: AtomicU8 = AtomicU8::new(0);

    #[inline]
    fn use_native() -> bool {
        match MODE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let native = native::available();
                MODE.store(if native { 1 } else { 2 }, Ordering::Relaxed);
                native
            }
        }
    }

    /// `true` ise assembly devre dışı bırakılır (hata ayıklama ve karşılaştırma için).
    pub fn force_portable(on: bool) {
        MODE.store(if on { 2 } else { 0 }, Ordering::Relaxed);
    }

    /// Etkin arka ucun adı: `"x86_64-avx2"`, `"arm64-neon"` veya `"portable"`.
    pub fn backend() -> &'static str {
        if use_native() {
            native::NAME
        } else {
            "portable"
        }
    }

    /// Etkin arka ucun sayısal kimliği (0 taşınabilir, 1 AVX2, 2 NEON).
    pub fn backend_id() -> i32 {
        if use_native() { native::ID } else { 0 }
    }

    /// Mutlak farkların toplamı.
    pub fn sad_u8(a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len());
        let mut done = 0;
        let mut total = 0;
        if use_native() {
            let blocks = a.len() / 32;
            if blocks > 0 {
                // SAFETY: her iki dilim de en az blocks*32 bayt uzunluğundadır.
                total = unsafe { native::sad_u8(a.as_ptr(), b.as_ptr(), blocks) };
                done = blocks * 32;
            }
        }
        total + portable::sad_u8(&a[done..], &b[done..])
    }

    /// `dst[i] = mask[i] != 0 ? restored[i] : original[i]` (yalnızca bayt seçimi).
    pub fn select_u8(dst: &mut [u8], restored: &[u8], original: &[u8], mask: &[u8]) {
        let n = dst.len();
        assert!(restored.len() == n && original.len() == n && mask.len() == n);
        let mut done = 0;
        if use_native() {
            let blocks = n / 32;
            if blocks > 0 {
                // SAFETY: dört dilim de en az blocks*32 bayt uzunluğundadır; dst diğerleriyle çakışmaz.
                unsafe {
                    native::select_u8(
                        dst.as_mut_ptr(),
                        restored.as_ptr(),
                        original.as_ptr(),
                        mask.as_ptr(),
                        blocks,
                    )
                };
                done = blocks * 32;
            }
        }
        portable::select_u8(
            &mut dst[done..],
            &restored[done..],
            &original[done..],
            &mask[done..],
        );
    }

    /// Yerinde seçim: `mask[i] != 0` olan baytlar `restored`'dan alınır, diğerlerine dokunulmaz.
    pub fn select_in_place(dst: &mut [u8], restored: &[u8], mask: &[u8]) {
        let n = dst.len();
        assert!(restored.len() == n && mask.len() == n);
        let mut done = 0;
        if use_native() {
            let blocks = n / 32;
            if blocks > 0 {
                let p = dst.as_mut_ptr();
                // SAFETY: dilimler en az blocks*32 bayttır. Çekirdek her bloğu yazmadan
                // önce yüklediği için dst == original kullanımı tanımlıdır.
                unsafe {
                    native::select_u8(p, restored.as_ptr(), p as *const u8, mask.as_ptr(), blocks)
                };
                done = blocks * 32;
            }
        }
        for i in done..n {
            if mask[i] != 0 {
                dst[i] = restored[i];
            }
        }
    }

    /// Bir satır çifti için kayıpsız tamsayı 2×2 Haar ileri dönüşümü.
    /// `row0.len() == row1.len() == 2 * ll.len()`.
    pub fn haar_fwd_rows(
        row0: &[u8],
        row1: &[u8],
        ll: &mut [i16],
        hl: &mut [i16],
        lh: &mut [i16],
        hh: &mut [i16],
    ) {
        let n = ll.len();
        assert!(
            row0.len() == 2 * n
                && row1.len() == 2 * n
                && hl.len() == n
                && lh.len() == n
                && hh.len() == n
        );
        let mut done = 0;
        if use_native() {
            let blocks = n / 16;
            if blocks > 0 {
                // SAFETY: girişler en az blocks*32 bayt, çıkışlar en az blocks*16 katsayıdır.
                unsafe {
                    native::haar_fwd(
                        row0.as_ptr(),
                        row1.as_ptr(),
                        ll.as_mut_ptr(),
                        hl.as_mut_ptr(),
                        lh.as_mut_ptr(),
                        hh.as_mut_ptr(),
                        blocks,
                    )
                };
                done = blocks * 16;
            }
        }
        portable::haar_fwd_rows(
            &row0[2 * done..],
            &row1[2 * done..],
            &mut ll[done..],
            &mut hl[done..],
            &mut lh[done..],
            &mut hh[done..],
        );
    }

    /// [`haar_fwd_rows`] dönüşümünün tersi (doygun aritmetik, 0..255 kırpma).
    pub fn haar_inv_rows(
        ll: &[i16],
        hl: &[i16],
        lh: &[i16],
        hh: &[i16],
        row0: &mut [u8],
        row1: &mut [u8],
    ) {
        let n = ll.len();
        assert!(
            row0.len() == 2 * n
                && row1.len() == 2 * n
                && hl.len() == n
                && lh.len() == n
                && hh.len() == n
        );
        let mut done = 0;
        if use_native() {
            let blocks = n / 16;
            if blocks > 0 {
                // SAFETY: girişler en az blocks*16 katsayı, çıkışlar en az blocks*32 bayttır.
                unsafe {
                    native::haar_inv(
                        ll.as_ptr(),
                        hl.as_ptr(),
                        lh.as_ptr(),
                        hh.as_ptr(),
                        row0.as_mut_ptr(),
                        row1.as_mut_ptr(),
                        blocks,
                    )
                };
                done = blocks * 16;
            }
        }
        portable::haar_inv_rows(
            &ll[done..],
            &hl[done..],
            &lh[done..],
            &hh[done..],
            &mut row0[2 * done..],
            &mut row1[2 * done..],
        );
    }

    /// `dst[i] = max(r[0][i], …, r[4][i])` — 5×5 maske genişletmenin yapı taşı.
    pub fn max5_u8(dst: &mut [u8], rows: [&[u8]; 5]) {
        let n = dst.len();
        assert!(rows.iter().all(|r| r.len() == n));
        let mut done = 0;
        if use_native() {
            let blocks = n / 32;
            if blocks > 0 {
                // SAFETY: altı dilim de en az blocks*32 bayttır; dst diğerleriyle çakışmaz (&mut).
                unsafe {
                    native::max5_u8(
                        dst.as_mut_ptr(),
                        rows[0].as_ptr(),
                        rows[1].as_ptr(),
                        rows[2].as_ptr(),
                        rows[3].as_ptr(),
                        rows[4].as_ptr(),
                        blocks,
                    )
                };
                done = blocks * 32;
            }
        }
        portable::max5_u8(
            &mut dst[done..],
            [
                &rows[0][done..],
                &rows[1][done..],
                &rows[2][done..],
                &rows[3][done..],
                &rows[4][done..],
            ],
        );
    }

    /// Analitik ters karışım (doğrusal ışık, Q16): `B = round_even((255·I − α·F) / (255 − α))`,
    /// [0, 65535]. Ara değerler < 2^24 olduğundan f32'de tamdır; yalnızca bölme yuvarlanır. `alpha` değerleri ≤ 254 olmalıdır (çağıran eşik üstünü 0 yapar veya hiç vermez).
    pub fn unblend_u16(dst: &mut [u16], img: &[u16], fg: &[u16], alpha: &[u8]) {
        let n = dst.len();
        assert!(img.len() == n && fg.len() == n && alpha.len() == n);
        assert!(alpha.iter().all(|&a| a < 255), "α = 255 ters çevrilemez");
        let mut done = 0;
        if use_native() {
            let blocks = n / 16;
            if blocks > 0 {
                // SAFETY: dilimler en az blocks*16 öğedir; dst diğerleriyle çakışmaz (&mut).
                unsafe {
                    native::unblend_u16(
                        dst.as_mut_ptr(),
                        img.as_ptr(),
                        fg.as_ptr(),
                        alpha.as_ptr(),
                        blocks,
                    )
                };
                done = blocks * 16;
            }
        }
        portable::unblend_u16(&mut dst[done..], &img[done..], &fg[done..], &alpha[done..]);
    }

    /// Düz Rust başvuru gerçeklemesi. Assembly çekirdeklerinin doğruluk ölçütü ve
    /// assembly bulunmayan hedeflerdeki çalışma yoludur.
    pub mod portable {
        /// Mutlak farkların toplamı.
        pub fn sad_u8(a: &[u8], b: &[u8]) -> u64 {
            a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y) as u64).sum()
        }

        /// Maskeye göre bayt seçimi.
        pub fn select_u8(dst: &mut [u8], restored: &[u8], original: &[u8], mask: &[u8]) {
            for i in 0..dst.len() {
                dst[i] = if mask[i] != 0 {
                    restored[i]
                } else {
                    original[i]
                };
            }
        }

        /// Beş satırın bayt bazında en büyüğü.
        pub fn max5_u8(dst: &mut [u8], rows: [&[u8]; 5]) {
            for i in 0..dst.len() {
                dst[i] = rows[0][i]
                    .max(rows[1][i])
                    .max(rows[2][i])
                    .max(rows[3][i])
                    .max(rows[4][i]);
            }
        }

        /// Ters karışım; assembly ile aynı işlem sırası (FMA yok, en yakın çifte yuvarlama).
        pub fn unblend_u16(dst: &mut [u16], img: &[u16], fg: &[u16], alpha: &[u8]) {
            for i in 0..dst.len() {
                let a = alpha[i] as f32;
                let num = 255.0f32 * img[i] as f32 - a * fg[i] as f32;
                let q = (num / (255.0f32 - a)).round_ties_even();
                dst[i] = q.clamp(0.0, 65535.0) as u16;
            }
        }

        /// 2×2 Haar ileri dönüşümü.
        pub fn haar_fwd_rows(
            row0: &[u8],
            row1: &[u8],
            ll: &mut [i16],
            hl: &mut [i16],
            lh: &mut [i16],
            hh: &mut [i16],
        ) {
            for i in 0..ll.len() {
                let a = row0[2 * i] as i16;
                let b = row0[2 * i + 1] as i16;
                let c = row1[2 * i] as i16;
                let d = row1[2 * i + 1] as i16;
                ll[i] = a + b + c + d;
                hl[i] = a - b + c - d;
                lh[i] = a + b - c - d;
                hh[i] = a - b - c + d;
            }
        }

        /// `(x + 2) >> 2` — taşmasız.
        #[inline]
        fn rsh2(x: i16) -> i16 {
            (x >> 2) + ((x >> 1) & 1)
        }

        /// 2×2 Haar ters dönüşümü.
        pub fn haar_inv_rows(
            ll: &[i16],
            hl: &[i16],
            lh: &[i16],
            hh: &[i16],
            row0: &mut [u8],
            row1: &mut [u8],
        ) {
            for i in 0..ll.len() {
                let t0 = ll[i].saturating_add(lh[i]);
                let t1 = ll[i].saturating_sub(lh[i]);
                let t2 = hl[i].saturating_add(hh[i]);
                let t3 = hl[i].saturating_sub(hh[i]);
                let px = |v: i16| rsh2(v).clamp(0, 255) as u8;
                row0[2 * i] = px(t0.saturating_add(t2));
                row0[2 * i + 1] = px(t0.saturating_sub(t2));
                row1[2 * i] = px(t1.saturating_add(t3));
                row1[2 * i + 1] = px(t1.saturating_sub(t3));
            }
        }
    }
}

/// Bir düzlemin tek seviyeli 2×2 Haar alt bantları.
///
/// Dört bant toplamda girişle aynı sayıda katsayı tutar; dönüşüm belleği
/// küçültmez. Kenar/detay enerjisi ölçümü ve frekans temelli denetim içindir.
pub struct HaarBands {
    /// Bant genişliği (giriş genişliğinin yarısı).
    pub width: usize,
    /// Bant yüksekliği (giriş yüksekliğinin yarısı).
    pub height: usize,
    /// Alçak-alçak bant: `a+b+c+d`.
    pub ll: Vec<i16>,
    /// Yatay detay: `a-b+c-d`.
    pub hl: Vec<i16>,
    /// Dikey detay: `a+b-c-d`.
    pub lh: Vec<i16>,
    /// Çapraz detay: `a-b-c+d`.
    pub hh: Vec<i16>,
}

impl HaarBands {
    /// `width × height` (ikisi de çift) boyutlu bir düzlem için bant belleği ayırır.
    pub fn new(width: usize, height: usize) -> Result<Self, CoreError> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(CoreError::InvalidGeometry);
        }
        let n = (width / 2) * (height / 2);
        Ok(Self {
            width: width / 2,
            height: height / 2,
            ll: vec![0; n],
            hl: vec![0; n],
            lh: vec![0; n],
            hh: vec![0; n],
        })
    }

    /// `plane` (satır aralığı `stride`) üzerinde ileri dönüşüm.
    pub fn forward(&mut self, plane: &[u8], stride: usize) {
        let (w2, w) = (self.width, self.width * 2);
        for y in 0..self.height {
            let row0 = &plane[2 * y * stride..][..w];
            let row1 = &plane[(2 * y + 1) * stride..][..w];
            let o = y * w2..(y + 1) * w2;
            kernels::haar_fwd_rows(
                row0,
                row1,
                &mut self.ll[o.clone()],
                &mut self.hl[o.clone()],
                &mut self.lh[o.clone()],
                &mut self.hh[o],
            );
        }
    }

    /// Bantlardan düzlemi yeniden kurar. İleri dönüşümün çıktısı için kayıpsızdır.
    pub fn inverse(&self, plane: &mut [u8], stride: usize) {
        let (w2, w) = (self.width, self.width * 2);
        for y in 0..self.height {
            let (top, bottom) = plane[2 * y * stride..].split_at_mut(stride);
            let o = y * w2..(y + 1) * w2;
            kernels::haar_inv_rows(
                &self.ll[o.clone()],
                &self.hl[o.clone()],
                &self.lh[o.clone()],
                &self.hh[o],
                &mut top[..w],
                &mut bottom[..w],
            );
        }
    }

    /// Katsayı başına ortalama yüksek frekans enerjisi `(|HL| + |LH| + |HH|)`.
    /// Onarılan bölgenin çevresine göre bulanık/aşırı keskin olup olmadığını
    /// denetlemek için kullanılır.
    pub fn detail_energy(&self) -> f64 {
        let n = self.ll.len();
        if n == 0 {
            return 0.0;
        }
        let sum: u64 = (0..n)
            .map(|i| {
                self.hl[i].unsigned_abs() as u64
                    + self.lh[i].unsigned_abs() as u64
                    + self.hh[i].unsigned_abs() as u64
            })
            .sum();
        sum as f64 / n as f64
    }
}

// ============================================================================
// 4b. Maske genişletme ve analitik alfa çözümü (deterministik kurtarma)
// ============================================================================

/// 5×5 kare yapı elemanıyla ikili/gri maske genişletme (dilation). Kenar dışı 0 sayılır.
/// Ayrılabilir: önce yatay 5'li en büyük, sonra dikey 5'li en büyük ([`kernels::max5_u8`]).
pub fn dilate5x5(mask: &[u8], width: usize, height: usize, out: &mut [u8]) {
    assert!(mask.len() >= width * height && out.len() >= width * height);
    if width == 0 || height == 0 {
        return;
    }
    let mut horiz = vec![0u8; width * height];
    let mut padded = vec![0u8; width + 4];
    for y in 0..height {
        padded[2..2 + width].copy_from_slice(&mask[y * width..(y + 1) * width]);
        let rows = [
            &padded[0..width],
            &padded[1..1 + width],
            &padded[2..2 + width],
            &padded[3..3 + width],
            &padded[4..4 + width],
        ];
        kernels::max5_u8(&mut horiz[y * width..(y + 1) * width], rows);
    }
    let zero = vec![0u8; width];
    for y in 0..height {
        let row = |dy: isize| -> &[u8] {
            let yy = y as isize + dy;
            if yy < 0 || yy >= height as isize {
                &zero
            } else {
                &horiz[yy as usize * width..(yy as usize + 1) * width]
            }
        };
        let rows = [row(-2), row(-1), row(0), row(1), row(2)];
        kernels::max5_u8(&mut out[y * width..(y + 1) * width], rows);
    }
}

/// Yarı saydam yazının analitik olarak geri alınması: `I = α·F + (1 − α)·B` karışımı doğrusal
/// ışıkta tersine çevrilir: `B = (I − α·F) / (1 − α)`. Gama (BT.1886, γ = 2,4) tabloyla kaldırılır,
/// hesap [`kernels::unblend_u16`] ile Q16 doğrusal değerlerde yapılır, sonra tekrar gama uygulanır.
///
/// Y düzlemine uygulanır; renk farkı (U, V) düzlemleri için `linear = false` ile sinyal düzleminde
/// doğrusal karışım varsayılır.
pub struct AlphaUnblend {
    to_linear: [u16; 256],
    from_linear: Vec<u8>,
}

impl Default for AlphaUnblend {
    fn default() -> Self {
        Self::new()
    }
}

impl AlphaUnblend {
    /// En yüksek güvenle çözülecek alfa (≈ 0,85 · 255). Üstü modele bırakılır.
    pub const MAX_ALPHA: u8 = 216;

    pub fn new() -> Self {
        let mut to_linear = [0u16; 256];
        for (v, slot) in to_linear.iter_mut().enumerate() {
            *slot = ((v as f64 / 255.0).powf(2.4) * 65535.0).round() as u16;
        }
        let from_linear = (0..65536u32)
            .map(|l| ((l as f64 / 65535.0).powf(1.0 / 2.4) * 255.0).round() as u8)
            .collect();
        Self {
            to_linear,
            from_linear,
        }
    }

    /// `plane` yerinde çözülür. `alpha` (0..255) sıfırsa veya [`Self::MAX_ALPHA`] üstündeyse piksel
    /// değişmez. Döner: (çözülen piksel, modele bırakılan piksel).
    pub fn unblend_plane(
        &self,
        plane: &mut [u8],
        alpha: &[u8],
        overlay: u8,
        linear: bool,
        max_alpha: u8,
    ) -> (usize, usize) {
        assert_eq!(plane.len(), alpha.len());
        let lin = |v: u8| {
            if linear {
                self.to_linear[v as usize]
            } else {
                (v as u16) * 257
            }
        };
        let back = |l: u16| {
            if linear {
                self.from_linear[l as usize]
            } else {
                ((l as u32 + 128) / 257) as u8
            }
        };
        let max_alpha = max_alpha.min(254);
        let mut resolved = 0;
        let mut deferred = 0;
        let a: Vec<u8> = alpha
            .iter()
            .map(|&x| {
                if x > max_alpha {
                    if x > 0 {
                        deferred += 1;
                    }
                    0
                } else {
                    if x > 0 {
                        resolved += 1;
                    }
                    x
                }
            })
            .collect();
        let img: Vec<u16> = plane.iter().map(|&v| lin(v)).collect();
        let fg = vec![lin(overlay); plane.len()];
        let mut out = vec![0u16; plane.len()];
        kernels::unblend_u16(&mut out, &img, &fg, &a);
        for i in 0..plane.len() {
            if a[i] != 0 {
                plane[i] = back(out[i]);
            }
        }
        (resolved, deferred)
    }
}

// ============================================================================
// 5. Sahne geçişi algılama
// ============================================================================

/// Sahne geçişi eşikleri. Varsayılanlar başlangıç değerleridir; gerçek video
/// kümesi üzerinde ayarlanmalıdır.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneCutConfig {
    /// Küçük resimlerin ortalama mutlak farkı (0..255) için alt eşik.
    pub thumb_threshold: f32,
    /// Parlaklık histogramı uzaklığı (0..1) için alt eşik.
    pub hist_threshold: f32,
    /// Fark, son karelerin hareket ortalamasının en az bu katı olmalıdır.
    pub adaptive_ratio: f32,
    /// İki geçiş arasındaki en az kare sayısı.
    pub min_scene_len: u32,
}

impl Default for SceneCutConfig {
    fn default() -> Self {
        Self {
            thumb_threshold: 20.0,
            hist_threshold: 0.25,
            adaptive_ratio: 4.0,
            min_scene_len: 2,
        }
    }
}

/// [`SceneDetector::feed`] sonucu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneVerdict {
    /// Bu kare yeni bir sahnenin ilk karesi mi?
    pub cut: bool,
    /// Küçük resimlerin ortalama mutlak farkı (0..255).
    pub thumb_diff: f32,
    /// Histogram uzaklığı (0..1).
    pub hist_diff: f32,
}

/// Y düzleminin küçük resmi (720p için 80×45) ve histogramı üzerinden sert
/// sahne geçişlerini bulur. Küçük resim farkı SAD çekirdeğiyle hesaplanır.
pub struct SceneDetector {
    cfg: SceneCutConfig,
    width: usize,
    tw: usize,
    th: usize,
    bw: usize,
    bh: usize,
    prev: Vec<u8>,
    cur: Vec<u8>,
    acc: Vec<u32>,
    prev_hist: [u32; 32],
    primed: bool,
    ema: f32,
    since_cut: u32,
}

impl SceneDetector {
    /// Verilen geometri için algılayıcı kurar.
    pub fn new(geo: FrameGeometry, cfg: SceneCutConfig) -> Self {
        let tw = (geo.width() / 16).clamp(1, 512);
        let th = (geo.height() / 16).clamp(1, 512);
        Self {
            cfg,
            width: geo.width(),
            tw,
            th,
            bw: geo.width() / tw,
            bh: geo.height() / th,
            prev: vec![0; tw * th],
            cur: vec![0; tw * th],
            acc: vec![0; tw],
            prev_hist: [0; 32],
            primed: false,
            ema: 0.0,
            since_cut: 0,
        }
    }

    /// Geçmişi unutur (akış yeniden başlatıldığında).
    pub fn reset(&mut self) {
        self.primed = false;
        self.ema = 0.0;
        self.since_cut = 0;
    }

    /// Sıradaki karenin Y düzlemini değerlendirir.
    pub fn feed(&mut self, luma: &[u8]) -> SceneVerdict {
        let (w, tw, th, bw, bh) = (self.width, self.tw, self.th, self.bw, self.bh);
        let step = if bh >= 2 { 2 } else { 1 };
        let div = (bw * bh.div_ceil(step)) as u32;
        for ty in 0..th {
            self.acc.fill(0);
            let mut y = ty * bh;
            while y < (ty + 1) * bh {
                let row = &luma[y * w..y * w + tw * bw];
                for (acc, cell) in self.acc.iter_mut().zip(row.chunks_exact(bw)) {
                    *acc += cell.iter().map(|&v| v as u32).sum::<u32>();
                }
                y += step;
            }
            for tx in 0..tw {
                self.cur[ty * tw + tx] = ((self.acc[tx] + div / 2) / div).min(255) as u8;
            }
        }
        let mut hist = [0u32; 32];
        for &v in &self.cur {
            hist[(v >> 3) as usize] += 1;
        }

        let n = (tw * th) as f32;
        let mut verdict = SceneVerdict {
            cut: false,
            thumb_diff: 0.0,
            hist_diff: 0.0,
        };
        if self.primed {
            verdict.thumb_diff = kernels::sad_u8(&self.cur, &self.prev) as f32 / n;
            let l1: u32 = hist
                .iter()
                .zip(&self.prev_hist)
                .map(|(&a, &b)| a.abs_diff(b))
                .sum();
            verdict.hist_diff = l1 as f32 / (2.0 * n);
            let floor = self
                .cfg
                .thumb_threshold
                .max(self.cfg.adaptive_ratio * self.ema);
            verdict.cut = self.since_cut >= self.cfg.min_scene_len
                && verdict.hist_diff >= self.cfg.hist_threshold
                && verdict.thumb_diff >= floor;
        }
        if verdict.cut {
            self.ema = 0.0;
            self.since_cut = 0;
        } else {
            self.ema = if self.primed {
                0.9 * self.ema + 0.1 * verdict.thumb_diff
            } else {
                0.0
            };
            self.since_cut = self.since_cut.saturating_add(1);
        }
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.prev_hist = hist;
        self.primed = true;
        verdict
    }
}

// ============================================================================
// 6. Tahminler, harmanlama ve koruma kapısı
// ============================================================================

/// Kare içi dikdörtgen bölge (piksel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// Sol kenar.
    pub x: usize,
    /// Üst kenar.
    pub y: usize,
    /// Genişlik.
    pub w: usize,
    /// Yükseklik.
    pub h: usize,
}

impl Rect {
    /// Boş bölge.
    pub const EMPTY: Rect = Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    /// Yeni dikdörtgen.
    pub const fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }

    /// Alanı sıfır mı?
    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Kareye kırpılmış ve 2×2 renk bloklarına hizalanmış (çift sınırlı) dikdörtgen.
    pub fn aligned(self, geo: FrameGeometry) -> Option<Rect> {
        if self.is_empty() || self.x >= geo.width() || self.y >= geo.height() {
            return None;
        }
        let x0 = self.x & !1;
        let y0 = self.y & !1;
        let x1 = (self.x.saturating_add(self.w).saturating_add(1) & !1).min(geo.width());
        let y1 = (self.y.saturating_add(self.h).saturating_add(1) & !1).min(geo.height());
        Some(Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        })
    }

    /// İki dikdörtgeni kapsayan en küçük dikdörtgen.
    pub fn union(self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w).max(other.x + other.w);
        let y1 = (self.y + self.h).max(other.y + other.h);
        Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }
}

/// Bir kare için model tahmini: onarılmış pikseller + piksel başına ağırlık.
///
/// * `weight == 0`  → piksel **korunur** (orijinal bayt aynen çıkar).
/// * `weight 1..=255` → piksel değiştirilir; değer aynı zamanda güven puanıdır ve
///   örtüşen iki pencerenin tahminleri harmanlanırken kullanılır.
///
/// Tahmin yalnızca [`Estimate::begin`] ile bildirilen ROI içinde geçerlidir ve
/// yalnızca o bölgeye yazılabilir. Tamponlar kare boyutunda ayrılır ama işletim
/// sistemi yalnızca dokunulan sayfaları gerçekten ayırdığından fiziksel bellek
/// kullanımı ROI alanıyla orantılıdır.
pub struct Estimate {
    geo: FrameGeometry,
    nv12: Box<[u8]>,
    weight: Box<[u8]>,
    roi: Rect,
    prior: u8,
}

impl Estimate {
    /// Boş tahmin (hiçbir pikseli değiştirmez).
    pub fn new(geo: FrameGeometry) -> Self {
        Self {
            geo,
            nv12: vec![0u8; geo.frame_bytes()].into_boxed_slice(),
            weight: vec![0u8; geo.luma_bytes()].into_boxed_slice(),
            roi: Rect::EMPTY,
            prior: 1,
        }
    }

    /// Tahmini boşaltır (önceki ROI'deki ağırlıklar sıfırlanır).
    pub fn clear(&mut self) {
        let r = self.roi;
        let w = self.geo.width();
        for y in r.y..r.y + r.h {
            self.weight[y * w + r.x..y * w + r.x + r.w].fill(0);
        }
        self.roi = Rect::EMPTY;
    }

    /// Yeni bir tahmine başlar ve yazılabilir bölgeyi belirler. Bölge kareye
    /// kırpılır, çift sınırlara genişletilir ve bu hâliyle döndürülür.
    pub fn begin(&mut self, roi: Rect) -> Result<Rect, CoreError> {
        self.clear();
        let aligned = roi.aligned(self.geo).ok_or(CoreError::BadArgument)?;
        self.roi = aligned;
        Ok(aligned)
    }

    /// Geçerli bölge (boşsa tahmin hiçbir pikseli değiştirmez).
    pub fn roi(&self) -> Rect {
        self.roi
    }

    /// Bu tahminin pencere içindeki konumundan gelen bağlam önceliği
    /// (`1 + min(geçerli geçmiş, geçerli gelecek)`, 1..=15).
    pub fn prior(&self) -> u8 {
        self.prior
    }

    fn luma_range(&self, y: usize) -> Range<usize> {
        let r = self.roi;
        assert!(y >= r.y && y < r.y + r.h, "satır ROI dışında");
        let start = y * self.geo.width() + r.x;
        start..start + r.w
    }

    fn chroma_range(&self, cy: usize) -> Range<usize> {
        let r = self.roi;
        assert!(
            cy >= r.y / 2 && cy < (r.y + r.h) / 2,
            "renk satırı ROI dışında"
        );
        let start = self.geo.luma_bytes() + cy * self.geo.width() + r.x;
        start..start + r.w
    }

    /// ROI içindeki `y` satırının Y değerleri.
    pub fn y_row(&self, y: usize) -> &[u8] {
        &self.nv12[self.luma_range(y)]
    }

    /// ROI içindeki `y` satırının ağırlıkları.
    pub fn weight_row(&self, y: usize) -> &[u8] {
        &self.weight[self.luma_range(y)]
    }

    /// ROI içindeki `cy` renk satırı (`U0 V0 U1 V1 …`, ROI genişliği kadar bayt).
    pub fn uv_row(&self, cy: usize) -> &[u8] {
        &self.nv12[self.chroma_range(cy)]
    }

    /// `y` satırının yazılabilir `(Y, ağırlık)` dilimleri.
    pub fn luma_rows_mut(&mut self, y: usize) -> (&mut [u8], &mut [u8]) {
        let range = self.luma_range(y);
        (&mut self.nv12[range.clone()], &mut self.weight[range])
    }

    /// `cy` renk satırının yazılabilir dilimi.
    pub fn uv_row_mut(&mut self, cy: usize) -> &mut [u8] {
        let range = self.chroma_range(cy);
        &mut self.nv12[range]
    }

    /// Bölgeyi sabit bir renk ve ağırlıkla doldurur (testler ve basit onarıcılar için).
    pub fn paint(&mut self, roi: Rect, y: u8, u: u8, v: u8, weight: u8) -> Result<Rect, CoreError> {
        let r = self.begin(roi)?;
        for row in r.y..r.y + r.h {
            let (ys, ws) = self.luma_rows_mut(row);
            ys.fill(y);
            ws.fill(weight);
        }
        for cy in r.y / 2..(r.y + r.h) / 2 {
            for pair in self.uv_row_mut(cy).as_chunks_mut::<2>().0 {
                pair[0] = u;
                pair[1] = v;
            }
        }
        Ok(r)
    }
}

/// Geri yazmanın son dokunuşları: maske kenarı alfa yumuşatması ve kumlanma (grain) eşleme.
///
/// * **Yumuşatma**: onarım bölgesinin (tahmin ağırlığı > 0) dış kenarından içe doğru mesafe ölçülür
///   (3-4 chamfer); kenardan `feather` piksel içeride tam onarım, arada kosinüs rampa ile orijinale geçiş.
///   Bölge DIŞINDAKİ piksellere hiç dokunulmaz (bit-exact kapı korunur); onarıcı bölgeyi maskeden
///   `feather` kadar fazla genişletmelidir ki yazı pikselleri tam onarım alanında kalsın.
/// * **Kumlanma eşleme**: 20–40M parametreli onarıcılar "plastik" (fazla pürüzsüz) doku üretir. ROI'nin
///   onarılmayan kısmından kameranın yüksek frekanslı gürültüsü ölçülür (y − 3×3 ortalama; σ = 1,4826 ·
///   ortanca mutlak sapma), onarılan bölgenin kendi gürültüsü düşülür ve eksik kalan σ deterministik
///   (kare sırasıyla tohumlanmış) gürültü olarak yalnızca parlaklığa eklenir.
pub mod finish {
    /// Bölge içi pikseller için dış kenara chamfer (3-4) mesafesi, piksel cinsinden (bölge dışı 0).
    ///
    /// `frame_edge` = [sol, üst, sağ, alt]: ROI'nin o kenarı karenin kenarıysa ötesi bölge İÇİ sayılır
    /// (köşedeki filigran kare kenarına doğru yarım onarılmasın); değilse ötesi bölge dışıdır.
    pub fn inner_distance(region: &[bool], w: usize, h: usize, frame_edge: [bool; 4]) -> Vec<f32> {
        const INF: u32 = u32::MAX / 4;
        let mut d: Vec<u32> = region.iter().map(|&r| if r { INF } else { 0 }).collect();
        let at = |x: isize, y: isize, d: &Vec<u32>| -> u32 {
            let beyond = [x < 0, y < 0, x >= w as isize, y >= h as isize];
            if beyond.iter().any(|&b| b) {
                return if beyond.iter().zip(frame_edge).any(|(&b, e)| b && !e) {
                    0
                } else {
                    INF
                };
            }
            d[y as usize * w + x as usize]
        };
        for y in 0..h as isize {
            for x in 0..w as isize {
                let i = y as usize * w + x as usize;
                if d[i] == 0 {
                    continue;
                }
                let best = (at(x - 1, y, &d) + 3)
                    .min(at(x, y - 1, &d) + 3)
                    .min(at(x - 1, y - 1, &d) + 4)
                    .min(at(x + 1, y - 1, &d) + 4);
                d[i] = d[i].min(best);
            }
        }
        for y in (0..h as isize).rev() {
            for x in (0..w as isize).rev() {
                let i = y as usize * w + x as usize;
                if d[i] == 0 {
                    continue;
                }
                let best = (at(x + 1, y, &d) + 3)
                    .min(at(x, y + 1, &d) + 3)
                    .min(at(x + 1, y + 1, &d) + 4)
                    .min(at(x - 1, y + 1, &d) + 4);
                d[i] = d[i].min(best);
            }
        }
        d.iter().map(|&v| v as f32 / 3.0).collect()
    }

    /// Mesafeden kapsama alfası (0..255): kenar komşusu küçük, `feather` + 1 pikselden derinde 255.
    pub fn ramp(distance: f32, feather: u32) -> u8 {
        if distance <= 0.0 {
            return 0;
        }
        if feather == 0 || distance > feather as f32 {
            return 255;
        }
        let t = distance / (feather as f32 + 1.0);
        (127.5 * (1.0 - (std::f32::consts::PI * t).cos()))
            .round()
            .clamp(1.0, 255.0) as u8
    }

    /// Bir düzlemin yüksek frekanslı gürültü sigması (y − 3×3 ortalama, sağlam MAD), `sel` piksellerinde.
    pub fn noise_sigma(
        plane: &[u8],
        w: usize,
        h: usize,
        sel: &dyn Fn(usize) -> bool,
    ) -> Option<f32> {
        let mut hist = [0u32; 256];
        let mut n = 0u32;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if !sel(i) {
                    continue;
                }
                let mut sum = 0i32;
                for dy in [-1isize, 0, 1] {
                    for dx in [-1isize, 0, 1] {
                        sum += plane[(i as isize + dy * w as isize + dx) as usize] as i32;
                    }
                }
                let r = (plane[i] as i32 * 9 - sum).abs() / 9;
                hist[r.min(255) as usize] += 1;
                n += 1;
            }
        }
        if n < 64 {
            return None;
        }
        let mut acc = 0;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc * 2 >= n {
                // 3×3 ortalamadan sapma, saf gürültü σ'sının √(8/9) katıdır; MAD → σ için 1,4826.
                return Some(1.4826 * v as f32 / (8.0f32 / 9.0).sqrt());
            }
        }
        None
    }

    /// Deterministik, yaklaşık normal (Irwin–Hall, 4 tekdüze) gürültü üreteci.
    pub struct Grain(u64);

    impl Grain {
        pub fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
        /// Ortalama 0, σ ≈ 1.
        pub fn normal(&mut self) -> f32 {
            let s: f32 = (0..4).map(|_| (self.next() as f32) / u32::MAX as f32).sum();
            (s - 2.0) * (3.0f32).sqrt()
        }
    }
}

/// Bileşimden sonra ROI içinde yumuşatma + kumlanma uygular (bkz. [`finish`]). `orig` girdinin, `dst`
/// bileşimin NV12 kopyasıdır; bölge = ağırlığı > 0 olan pikseller. Eklenen σ'yı döndürür.
fn finish_roi(
    geo: FrameGeometry,
    orig: &[u8],
    dst: &mut [u8],
    region_of: &dyn Fn(usize) -> bool,
    roi: Rect,
    feather: u32,
    grain: bool,
    seed: u64,
) -> f32 {
    if roi.is_empty() || (feather == 0 && !grain) {
        return 0.0;
    }
    let w = geo.width();
    let (rw, rh) = (roi.w, roi.h);
    let region: Vec<bool> = (0..rw * rh)
        .map(|k| region_of((roi.y + k / rw) * w + roi.x + k % rw))
        .collect();
    let edge = [
        roi.x == 0,
        roi.y == 0,
        roi.x + rw >= w,
        roi.y + rh >= geo.height(),
    ];
    let dist = finish::inner_distance(&region, rw, rh, edge);
    let alpha: Vec<u8> = dist.iter().map(|&d| finish::ramp(d, feather)).collect();
    let uv_base = geo.luma_bytes();
    let mut added = 0.0;
    if grain {
        let mut src_y = vec![0u8; rw * rh];
        let mut out_y = vec![0u8; rw * rh];
        for r in 0..rh {
            let row = (roi.y + r) * w + roi.x;
            src_y[r * rw..(r + 1) * rw].copy_from_slice(&orig[row..row + rw]);
            out_y[r * rw..(r + 1) * rw].copy_from_slice(&dst[row..row + rw]);
        }
        let s_src = finish::noise_sigma(&src_y, rw, rh, &|k| !region[k]);
        let s_out = finish::noise_sigma(&out_y, rw, rh, &|k| alpha[k] == 255);
        if let (Some(a), Some(b)) = (s_src, s_out) {
            added = (a * a - b * b).max(0.0).sqrt();
        }
        if added > 0.25 {
            let mut rng = finish::Grain::new(seed);
            for (k, &inside) in region.iter().enumerate() {
                if inside {
                    let i = (roi.y + k / rw) * w + roi.x + k % rw;
                    dst[i] = (dst[i] as f32 + added * rng.normal())
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    if feather > 0 {
        let mix = |o: u8, e: u8, a: u32| ((e as u32 * a + o as u32 * (255 - a) + 127) / 255) as u8;
        for (k, &a) in alpha.iter().enumerate() {
            if a != 0 && a != 255 {
                let i = (roi.y + k / rw) * w + roi.x + k % rw;
                dst[i] = mix(orig[i], dst[i], a as u32);
            }
        }
        for cy in 0..rh / 2 {
            for cx in 0..rw / 2 {
                let ks = [
                    2 * cy * rw + 2 * cx,
                    2 * cy * rw + 2 * cx + 1,
                    (2 * cy + 1) * rw + 2 * cx,
                    (2 * cy + 1) * rw + 2 * cx + 1,
                ];
                if ks.iter().all(|&k| alpha[k] == 0) {
                    continue;
                }
                let a = ks.iter().map(|&k| alpha[k] as u32).sum::<u32>() / 4;
                if a == 255 {
                    continue;
                }
                let i = uv_base + ((roi.y / 2) + cy) * w + roi.x + 2 * cx;
                for c in 0..2 {
                    dst[i + c] = mix(orig[i + c], dst[i + c], a);
                }
            }
        }
    }
    added
}

/// `dst` (orijinal karenin kopyası) üzerine tahminleri uygular; değiştirilen Y
/// pikseli sayısını döndürür.
///
/// Tek tahmin varsa maskeli pikseller doğrudan seçilir (assembly çekirdeği).
/// İki tahmin varsa her piksel `öncelik × ağırlık` oranında harmanlanır. Her iki
/// tahminde de ağırlığı 0 olan piksele **hiç yazılmaz**.
pub fn compose(
    geo: FrameGeometry,
    dst: &mut [u8],
    est: &Estimate,
    cached: Option<&Estimate>,
) -> u64 {
    let cached = cached.filter(|c| !c.roi.is_empty());
    match (est.roi.is_empty(), cached) {
        (true, None) => 0,
        (false, None) => compose_single(geo, dst, est),
        (true, Some(c)) => compose_single(geo, dst, c),
        (false, Some(c)) => compose_fused(geo, dst, est, c),
    }
}

fn compose_single(geo: FrameGeometry, dst: &mut [u8], est: &Estimate) -> u64 {
    let w = geo.width();
    let r = est.roi;
    let uv_base = geo.luma_bytes();
    let (dst_y, dst_uv) = dst.split_at_mut(uv_base);
    let mut modified = 0u64;
    for y in r.y..r.y + r.h {
        let range = y * w + r.x..y * w + r.x + r.w;
        let weights = &est.weight[range.clone()];
        kernels::select_in_place(&mut dst_y[range.clone()], &est.nv12[range], weights);
        modified += weights.iter().filter(|&&v| v != 0).count() as u64;
    }
    for cy in r.y / 2..(r.y + r.h) / 2 {
        let w0 = &est.weight[2 * cy * w + r.x..][..r.w];
        let w1 = &est.weight[(2 * cy + 1) * w + r.x..][..r.w];
        let src = &est.nv12[uv_base + cy * w + r.x..][..r.w];
        let out = &mut dst_uv[cy * w + r.x..][..r.w];
        for i in (0..r.w).step_by(2) {
            if (w0[i] | w0[i + 1] | w1[i] | w1[i + 1]) != 0 {
                out[i] = src[i];
                out[i + 1] = src[i + 1];
            }
        }
    }
    modified
}

fn compose_fused(geo: FrameGeometry, dst: &mut [u8], a: &Estimate, b: &Estimate) -> u64 {
    let w = geo.width();
    let r = a.roi.union(b.roi);
    let uv_base = geo.luma_bytes();
    let (pa, pb) = (a.prior.max(1) as u32, b.prior.max(1) as u32);
    let (dst_y, dst_uv) = dst.split_at_mut(uv_base);
    let mut modified = 0u64;
    for cy in r.y / 2..(r.y + r.h) / 2 {
        for cx in r.x / 2..(r.x + r.w) / 2 {
            let (mut sum_a, mut sum_b) = (0u32, 0u32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (2 * cy + dy) * w + 2 * cx + dx;
                    let (wa, wb) = (a.weight[i] as u32, b.weight[i] as u32);
                    sum_a += wa;
                    sum_b += wb;
                    let (ka, kb) = (pa * wa, pb * wb);
                    let k = ka + kb;
                    // k == 0: piksel iki tahminde de maskesiz → hiç yazılmaz.
                    if let Some(value) =
                        (a.nv12[i] as u32 * ka + b.nv12[i] as u32 * kb + k / 2).checked_div(k)
                    {
                        dst_y[i] = value as u8;
                        modified += 1;
                    }
                }
            }
            let (ka, kb) = (pa * sum_a, pb * sum_b);
            let k = ka + kb;
            let i = cy * w + 2 * cx;
            for c in 0..2 {
                let (va, vb) = (
                    a.nv12[uv_base + i + c] as u32,
                    b.nv12[uv_base + i + c] as u32,
                );
                // k == 0: bloğun dört pikseli de maskesiz → renk örneği hiç yazılmaz.
                if let Some(value) = (va * ka + vb * kb + k / 2).checked_div(k) {
                    dst_uv[i + c] = value as u8;
                }
            }
        }
    }
    modified
}

/// Koruma kapısının denetimi: tahminlerin dokunmadığı her bayt orijinalle aynı
/// mı? İhlal sayısını döndürür (beklenen: 0).
///
/// * Y: iki tahminde de ağırlığı 0 olan her piksel.
/// * UV: dört pikselinin hiçbiri maskeli olmayan her 2×2 blok.
pub fn verify_preservation(
    geo: FrameGeometry,
    original: &[u8],
    output: &[u8],
    est: &Estimate,
    cached: Option<&Estimate>,
) -> u64 {
    let (w, h) = (geo.width(), geo.height());
    let uv_base = geo.luma_bytes();
    let touched = est.roi.union(cached.map_or(Rect::EMPTY, |c| c.roi));
    let weight = |i: usize| est.weight[i] as u32 + cached.map_or(0, |c| c.weight[i] as u32);
    let mismatches = |range: Range<usize>| -> u64 {
        if original[range.clone()] == output[range.clone()] {
            0
        } else {
            range.filter(|&i| original[i] != output[i]).count() as u64
        }
    };
    let mut bad = 0u64;
    for y in 0..h {
        if y < touched.y || y >= touched.y + touched.h {
            bad += mismatches(y * w..(y + 1) * w);
        } else {
            bad += (0..w)
                .filter(|&x| weight(y * w + x) == 0 && original[y * w + x] != output[y * w + x])
                .count() as u64;
        }
    }
    for cy in 0..h / 2 {
        let row = uv_base + cy * w;
        if 2 * cy < touched.y || 2 * cy >= touched.y + touched.h {
            bad += mismatches(row..row + w);
        } else {
            for cx in 0..w / 2 {
                let i = 2 * cy * w + 2 * cx;
                let any = weight(i) + weight(i + 1) + weight(i + w) + weight(i + w + 1);
                let j = row + 2 * cx;
                if any == 0 && (original[j] != output[j] || original[j + 1] != output[j + 1]) {
                    bad += 1;
                }
            }
        }
    }
    bad
}

// ============================================================================
// 7. Scheduler — pencere planlama (saf mantık)
// ============================================================================

/// Çalışma modu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 14 geçmiş + 1 hedef + 15 gelecek; yalnızca hedef kare kesinleşir.
    A,
    /// 11+4 blok; `slide` kadar kare kesinleşir, son 4 konum önbelleğe alınır.
    B,
}

/// Kare başına işleme ipucu. Detector çıktısından türetilir ve kare
/// karara bağlanmadan önce [`Pipeline::set_hint`] ile verilir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hint {
    /// Olağan: Mod B.
    #[default]
    Normal,
    /// Zor kare (büyük maske, hızlı hareket, yoğun örtüşme): Mod A.
    Precision,
    /// Silinecek bir şey yok: modele sokmadan aynen geçir.
    Bypass,
}

impl Hint {
    /// JNI tamsayı değerinden dönüşüm.
    pub fn from_i32(v: i32) -> Option<Hint> {
        match v {
            0 => Some(Hint::Normal),
            1 => Some(Hint::Precision),
            2 => Some(Hint::Bypass),
            _ => None,
        }
    }
}

/// Modelin çalıştırılacağı 15 konumluk pencerenin tanımı.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPlan {
    /// Artan plan numarası.
    pub id: u64,
    /// Çalışma modu.
    pub mode: Mode,
    /// Bu pencereyle kesinleşen kare sayısı (Mod A: 1, Mod B: 0..=11).
    pub slide: u8,
    /// Konum 0'a karşılık gelen sıra numarası. Sahne başında negatif olabilir
    /// (o konumlar dolgudur).
    pub base_seq: i64,
    /// Geçerlilik maskesi: bit `p` = konum `p` gerçek bir kare içeriyor.
    pub valid: u32,
    /// Çıktısı kesinleşecek konumlar `[başlangıç, bitiş)`.
    pub emit: (u8, u8),
    /// Tahmini önbelleğe alınacak konumlar `[başlangıç, bitiş)`.
    pub cache: (u8, u8),
    /// `PLAN_FLAG_*` bitleri.
    pub flags: u32,
}

impl WindowPlan {
    /// Konum gerçek bir kare içeriyor mu?
    pub fn is_valid(&self, pos: usize) -> bool {
        pos < WINDOW_FRAMES && self.valid & (1 << pos) != 0
    }

    /// Konumdaki karenin sıra numarası (dolgu ise `None`).
    pub fn seq_at(&self, pos: usize) -> Option<u64> {
        self.is_valid(pos)
            .then(|| (self.base_seq + pos as i64) as u64)
    }

    /// Geçerli kare sayısı.
    pub fn valid_count(&self) -> u32 {
        self.valid.count_ones()
    }

    /// Konumdan önceki geçerli kare sayısı (geçmiş bağlam).
    pub fn past_context(&self, pos: usize) -> u32 {
        (self.valid & ((1u32 << pos) - 1)).count_ones()
    }

    /// Konumdan sonraki geçerli kare sayısı (gelecek bağlam).
    pub fn future_context(&self, pos: usize) -> u32 {
        (self.valid >> (pos + 1)).count_ones()
    }

    /// Bu konum için model çıktısı isteniyor mu?
    pub fn wants_estimate(&self, pos: usize) -> bool {
        let p = pos as u8;
        (p >= self.emit.0 && p < self.emit.1) || (p >= self.cache.0 && p < self.cache.1)
    }

    /// Harmanlama önceliği: `1 + min(geçmiş, gelecek)` (15 karelik pencerede en çok 8). Olağan Mod B
    /// akışında örtüşen 4 karede iki tahmin arasında doğrusal geçiş üretir (Models/Core.py ile aynı).
    pub fn prior_at(&self, pos: usize) -> u8 {
        (1 + self.past_context(pos).min(self.future_context(pos))) as u8
    }
}

/// Kesinleşen bir karenin piksel kaynağı.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSource {
    /// Orijinal kare aynen (bypass).
    Original,
    /// Model tahmini; `fuse_index` varsa önceki pencerenin önbelleğiyle harmanlanır.
    Estimate {
        /// Kesinleşen tahminler dizisindeki sıra.
        emit_index: u8,
        /// Önceki önbellek dizisindeki sıra (harmanlama yapılacaksa).
        fuse_index: Option<u8>,
    },
}

/// Çıkış kuyruğundaki kare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputItem {
    /// Karenin sıra numarası.
    pub seq: u64,
    /// Piksel kaynağı.
    pub source: OutputSource,
    /// `OUT_FLAG_*` bitleri.
    pub flags: u32,
}

/// Sürücünün atması gereken sıradaki adım.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// En çok bu kadar kare itin (kaynak bittiyse `finish` çağırın).
    NeedFrames(usize),
    /// Bu pencere için modeli çalıştırıp `complete_window` çağırın.
    Window(WindowPlan),
    /// Bu kadar kare çıkışa hazır; `pop_output` ile alın.
    Output(usize),
    /// Tüm kareler çıktı.
    Finished,
}

/// Zamanlayıcı sayaçları.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedStats {
    /// Tamamlanan Mod A pencereleri.
    pub windows_a: u64,
    /// Tamamlanan (kare kesinleştiren) Mod B pencereleri.
    pub windows_b: u64,
    /// Köprü geçişleri (`B(0)`).
    pub bridges: u64,
    /// Modele sokulmadan geçirilen kareler.
    pub bypassed: u64,
    /// İki tahmini harmanlanan kareler.
    pub fused: u64,
    /// Tamamlanan sahne (segment) sayısı.
    pub segments: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    A,
    B(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheRange {
    /// Önbellek dizisinin 0. elemanına karşılık gelen sıra numarası.
    base: u64,
    lo: u64,
    hi: u64,
}

/// Pencere zamanlayıcı: yalnızca sıra numaralarıyla çalışan durum makinesi.
///
/// Her kararın alındığı "kanonik durum"da halka en çok `[e-11, e+11)` aralığını tutar
/// (`e` = sıradaki kesinleşecek kare): 11 kesinleşmiş kare geçmiş bağlam (köprü penceresi
/// `[e-11, e+4)` için), 11 karelik karar ufku. Buradan Mod A 1 kare, Mod B `j` kare kaydırarak
/// aynı duruma döner; bu yüzden iki mod kare kare karıştırılabilir ve halka hiçbir zaman
/// `RING_FRAMES` (22) kareyi aşmaz.
pub struct Scheduler {
    pushed: u64,
    eos: bool,
    cuts: VecDeque<u64>,
    seg_start: u64,
    emit_next: u64,
    retired: u64,
    decided_until: u64,
    cache: Option<CacheRange>,
    hints: [Hint; RING_FRAMES],
    pending: Option<Action>,
    issued: Option<WindowPlan>,
    outputs: VecDeque<OutputItem>,
    next_plan_id: u64,
    stats: SchedStats,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    /// Boş zamanlayıcı.
    pub fn new() -> Self {
        Self {
            pushed: 0,
            eos: false,
            cuts: VecDeque::new(),
            seg_start: 0,
            emit_next: 0,
            retired: 0,
            decided_until: 0,
            cache: None,
            hints: [Hint::Normal; RING_FRAMES],
            pending: None,
            issued: None,
            outputs: VecDeque::new(),
            next_plan_id: 0,
            stats: SchedStats::default(),
        }
    }

    /// İtilen toplam kare sayısı.
    pub fn pushed(&self) -> u64 {
        self.pushed
    }

    /// Kesinleşen toplam kare sayısı.
    pub fn emitted(&self) -> u64 {
        self.emit_next
    }

    /// Bu numaradan küçük kareler artık gerekmez; halkadan çıkarılabilir.
    pub fn retired(&self) -> u64 {
        self.retired
    }

    /// Akış sonu bildirildi mi?
    pub fn is_eos(&self) -> bool {
        self.eos
    }

    /// Halkada kalan boş yuva sayısı.
    pub fn capacity_left(&self) -> usize {
        RING_FRAMES - (self.pushed - self.retired) as usize
    }

    /// Verilmiş ama henüz tamamlanmamış plan.
    pub fn issued(&self) -> Option<WindowPlan> {
        self.issued
    }

    /// Sayaçlar.
    pub fn stats(&self) -> SchedStats {
        self.stats
    }

    fn hint(&self, seq: u64) -> Hint {
        self.hints[(seq % RING_FRAMES as u64) as usize]
    }

    /// Yeni bir kare geldiğini bildirir. `cut`, karenin yeni bir sahnenin ilk
    /// karesi olduğunu söyler. Karenin sıra numarasını döndürür.
    pub fn on_push(&mut self, cut: bool) -> Result<u64, CoreError> {
        if self.eos {
            return Err(CoreError::StreamFinished);
        }
        if self.capacity_left() == 0 {
            return Err(CoreError::RingFull);
        }
        let seq = self.pushed;
        if cut && seq > 0 {
            self.cuts.push_back(seq);
        }
        self.hints[(seq % RING_FRAMES as u64) as usize] = Hint::Normal;
        self.pushed += 1;
        Ok(seq)
    }

    /// Kaynağın bittiğini bildirir.
    pub fn on_eos(&mut self) {
        self.eos = true;
    }

    /// Henüz karara bağlanmamış bir kare için ipucu verir.
    pub fn set_hint(&mut self, seq: u64, hint: Hint) -> Result<(), CoreError> {
        if seq >= self.pushed {
            return Err(CoreError::UnknownFrame);
        }
        if seq < self.decided_until {
            return Err(CoreError::BadState);
        }
        self.hints[(seq % RING_FRAMES as u64) as usize] = hint;
        Ok(())
    }

    fn segment_end(&self) -> Option<u64> {
        self.cuts
            .front()
            .copied()
            .or(self.eos.then_some(self.pushed))
    }

    fn retire_before(&mut self, seq: u64) {
        self.retired = self.retired.max(seq.min(self.pushed));
    }

    fn trim_cache(&mut self) {
        if let Some(c) = &mut self.cache {
            c.lo = c.lo.max(self.emit_next);
            if c.lo >= c.hi {
                self.cache = None;
            }
        }
    }

    /// Sıradaki adımı döndürür. Aynı durumda tekrar çağrılması güvenlidir.
    pub fn poll(&mut self) -> Step {
        if !self.outputs.is_empty() {
            return Step::Output(self.outputs.len());
        }
        if let Some(plan) = self.issued {
            return Step::Window(plan);
        }
        loop {
            let end = self.segment_end();
            let e = self.emit_next;
            if end == Some(e) {
                // Sahnenin tüm kareleri çıktı: kalan bağlam kareleri artık gereksiz.
                self.retire_before(e);
                if self.cuts.front() == Some(&e) {
                    self.cuts.pop_front();
                    self.seg_start = e;
                    self.cache = None;
                    self.pending = None;
                    self.stats.segments += 1;
                    continue;
                }
                return Step::Finished;
            }
            let s = self.seg_start;

            if self.pending.is_none() {
                // Karar için sıradaki 11 karenin (veya sahne sonuna kadarının) gelmiş olması gerekir.
                let horizon = end.map_or(e + STRIDE as u64, |c| c.min(e + STRIDE as u64));
                if self.pushed < horizon {
                    self.retire_before(e.saturating_sub(STRIDE as u64).max(s));
                    return Step::NeedFrames(
                        ((horizon - self.pushed) as usize).min(self.capacity_left()),
                    );
                }
                if self.decide(e, s, horizon) {
                    return Step::Output(self.outputs.len());
                }
            }

            let action = self.pending.expect("karar verilmiş olmalı");
            let (w_lo, w_hi) = match action {
                Action::A => (
                    e as i64 - MODE_A_TARGET as i64,
                    e + (WINDOW_FRAMES - MODE_A_TARGET) as u64,
                ),
                Action::B(j) => (
                    e as i64 + j as i64 - STRIDE as i64,
                    e + j as u64 + OVERLAP as u64,
                ),
            };
            let need_hi = end.map_or(w_hi, |c| c.min(w_hi));
            // Halkadan çıkarma sınırı: pencerenin başı, ama sonraki karenin olası köprü penceresinin
            // ([e'-11, e'+4), e' = bu pencereden sonra kesinleşecek kare) başından ileri değil. Mod A
            // penceresi yalnızca 7 geçmiş kare kullanır; ardından gelen köprü 11 ister.
            let slide = match action {
                Action::A => 1,
                Action::B(j) => j as i64,
            };
            let keep_from = w_lo.min(e as i64 + slide - STRIDE as i64);
            self.retire_before(keep_from.max(s as i64) as u64);
            if self.pushed < need_hi {
                return Step::NeedFrames(
                    ((need_hi - self.pushed) as usize).min(self.capacity_left()),
                );
            }
            let plan = self.build_plan(action, w_lo, need_hi, s, end);
            self.pending = None;
            self.issued = Some(plan);
            return Step::Window(plan);
        }
    }

    /// `e` konumundaki kare için eylem seçer. Bypass kareleri hemen çıkışa
    /// koyar ve `true` döndürür; aksi hâlde `pending` atanır.
    fn decide(&mut self, e: u64, s: u64, horizon: u64) -> bool {
        let run_of = |this: &Self, hint: Hint| -> u64 {
            let mut n = 0;
            while e + n < horizon && n < STRIDE as u64 && this.hint(e + n) == hint {
                n += 1;
            }
            n
        };
        match self.hint(e) {
            Hint::Bypass => {
                let run = run_of(self, Hint::Bypass);
                for k in 0..run {
                    self.outputs.push_back(OutputItem {
                        seq: e + k,
                        source: OutputSource::Original,
                        flags: OUT_FLAG_BYPASS,
                    });
                }
                self.emit_next = e + run;
                self.decided_until = self.emit_next;
                self.stats.bypassed += run;
                self.trim_cache();
                true
            }
            Hint::Precision => {
                self.pending = Some(Action::A);
                self.decided_until = e + 1;
                false
            }
            Hint::Normal => {
                let run = run_of(self, Hint::Normal);
                let covered = self
                    .cache
                    .filter(|c| c.lo == e && c.hi > e)
                    .map(|c| c.hi - e);
                let slide = match covered {
                    // Önceki pencerenin örtüşen tahminleri hazır (en çok 4 kare): bu `k` kare harmanlanır,
                    // sonrakiler yalnızca bu pencereden kesinleşir. Tek tahminli kare pencerede en az
                    // OVERLAP konum geçmiş görmeli: konumu 11-j+k ≥ 4 → j ≤ 7 + k (Mod A sonrası kırpılmış
                    // önbellekte kayma kısalır; olağan akışta k = 4 → j = 11).
                    Some(k) => run.min((STRIDE - OVERLAP) as u64 + k),
                    // Geçmiş kare var ama önbellek yok: önce köprü geçişi.
                    None if e > s => 0,
                    // Sahne başı: geçmiş yok, harmanlanacak bir şey de yok.
                    None => run,
                };
                self.pending = Some(Action::B(slide as u8));
                self.decided_until = e + slide;
                false
            }
        }
    }

    fn build_plan(
        &mut self,
        action: Action,
        w_lo: i64,
        need_hi: u64,
        s: u64,
        end: Option<u64>,
    ) -> WindowPlan {
        let mut valid = 0u32;
        for p in 0..WINDOW_FRAMES {
            let q = w_lo + p as i64;
            if q >= s as i64 && (q as u64) < need_hi {
                valid |= 1 << p;
            }
        }
        let tail = (valid >> STRIDE).count_ones() as u8;
        let (mode, slide, emit, cache) = match action {
            Action::A => (
                Mode::A,
                1,
                (MODE_A_TARGET as u8, MODE_A_TARGET as u8 + 1),
                (0, 0),
            ),
            Action::B(j) => (
                Mode::B,
                j,
                (STRIDE as u8 - j, STRIDE as u8),
                (STRIDE as u8, STRIDE as u8 + tail),
            ),
        };
        let mut flags = 0;
        if w_lo <= s as i64 {
            flags |= PLAN_FLAG_SEGMENT_START;
        }
        if end == Some(need_hi) {
            flags |= PLAN_FLAG_SEGMENT_END;
        }
        if valid.count_ones() < LOW_CONTEXT_FRAMES {
            flags |= PLAN_FLAG_LOW_CONTEXT;
        }
        if action == Action::B(0) {
            flags |= PLAN_FLAG_BRIDGE;
        }
        let plan = WindowPlan {
            id: self.next_plan_id,
            mode,
            slide,
            base_seq: w_lo,
            valid,
            emit,
            cache,
            flags,
        };
        self.next_plan_id += 1;
        debug_assert!((emit.0..emit.1).all(|p| plan.is_valid(p as usize)));
        plan
    }

    /// Verilmiş planın model çıktılarının yazıldığını bildirir. Mod B için `true`
    /// döner: çağıran, "önbellek" ve "yeni önbellek" tahmin dizilerini yer
    /// değiştirmelidir (bkz. [`Pipeline::complete_window`]).
    pub fn complete(&mut self, plan_id: u64) -> Result<bool, CoreError> {
        let plan = self.issued.ok_or(CoreError::BadState)?;
        if plan.id != plan_id {
            return Err(CoreError::BadArgument);
        }
        self.issued = None;
        let e = self.emit_next;
        let low = if plan.flags & PLAN_FLAG_LOW_CONTEXT != 0 {
            OUT_FLAG_LOW_CONTEXT
        } else {
            0
        };
        match plan.mode {
            Mode::A => {
                self.outputs.push_back(OutputItem {
                    seq: e,
                    source: OutputSource::Estimate {
                        emit_index: 0,
                        fuse_index: None,
                    },
                    flags: OUT_FLAG_RESTORED | OUT_FLAG_MODE_A | low,
                });
                self.emit_next = e + 1;
                self.stats.windows_a += 1;
                self.trim_cache();
                Ok(false)
            }
            Mode::B => {
                let j = plan.slide as u64;
                for k in 0..j {
                    let seq = e + k;
                    let fuse = self
                        .cache
                        .filter(|c| seq >= c.lo && seq < c.hi)
                        .map(|c| (seq - c.base) as u8);
                    let mut flags = OUT_FLAG_RESTORED | low;
                    if fuse.is_some() {
                        flags |= OUT_FLAG_FUSED;
                        self.stats.fused += 1;
                    }
                    self.outputs.push_back(OutputItem {
                        seq,
                        source: OutputSource::Estimate {
                            emit_index: k as u8,
                            fuse_index: fuse,
                        },
                        flags,
                    });
                }
                let lo = e + j;
                let count = (plan.cache.1 - plan.cache.0) as u64;
                self.cache = (count > 0).then_some(CacheRange {
                    base: lo,
                    lo,
                    hi: lo + count,
                });
                self.emit_next = lo;
                if j == 0 {
                    self.stats.bridges += 1;
                } else {
                    self.stats.windows_b += 1;
                }
                Ok(true)
            }
        }
    }

    /// Çıkış kuyruğunun başındaki kare (çıkarmadan).
    pub fn peek_output(&self) -> Option<OutputItem> {
        self.outputs.front().copied()
    }

    /// Çıkış kuyruğunun başındaki kareyi çıkarır.
    pub fn pop_output(&mut self) -> Option<OutputItem> {
        self.outputs.pop_front()
    }
}

// ============================================================================
// 8. Pipeline — kare giriş/çıkış hattı
// ============================================================================

/// Çıkan bir karenin bilgisi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputMeta {
    /// Karenin sıra numarası (girişle aynı).
    pub seq: u64,
    /// Zaman damgası (girişle aynı).
    pub pts_us: i64,
    /// `OUT_FLAG_*` bitleri.
    pub flags: u32,
    /// Değiştirilen Y pikseli sayısı.
    pub modified_pixels: u64,
}

/// Hat sayaçları ve bütünlük göstergeleri.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Kabul edilen kare sayısı.
    pub frames_in: u64,
    /// Çıkan kare sayısı (akış sonunda `frames_in` ile eşit olmalıdır).
    pub frames_out: u64,
    /// Algılanan/bildirilen sahne geçişi sayısı.
    pub scene_cuts: u64,
    /// Değiştirilen toplam Y pikseli.
    pub modified_pixels: u64,
    /// Koruma kapısı ihlali (bütünlük denetimi açıkken; beklenen 0).
    pub preservation_violations: u64,
    /// Sıra dışı çıkan kare sayısı (beklenen 0).
    pub order_violations: u64,
    /// Halkanın ulaştığı en yüksek doluluk.
    pub max_occupancy: u32,
    /// Zamanlayıcı sayaçları.
    pub sched: SchedStats,
}

/// Bir pencere için modelin okuyacağı kareler ve yazacağı tahminler.
pub struct WindowIo<'p> {
    plan: WindowPlan,
    geo: FrameGeometry,
    ring: &'p FrameRing,
    emit: &'p mut [Estimate],
    tail: &'p mut [Estimate],
}

impl<'p> WindowIo<'p> {
    /// Pencere planı.
    pub fn plan(&self) -> &WindowPlan {
        &self.plan
    }

    /// Kare geometrisi.
    pub fn geometry(&self) -> FrameGeometry {
        self.geo
    }

    /// Pencere konumundaki giriş karesi (dolgu ise `None`).
    pub fn frame(&self, pos: usize) -> Option<ReadLease<'p>> {
        self.ring.read(self.plan.seq_at(pos)?).ok()
    }

    /// Pencere konumu için yazılacak tahmin (çıktı istenmeyen konumlarda `None`).
    pub fn estimate_mut(&mut self, pos: usize) -> Option<&mut Estimate> {
        let p = pos as u8;
        if p >= self.plan.emit.0 && p < self.plan.emit.1 {
            Some(&mut self.emit[(p - self.plan.emit.0) as usize])
        } else if p >= self.plan.cache.0 && p < self.plan.cache.1 {
            Some(&mut self.tail[(p - STRIDE as u8) as usize])
        } else {
            None
        }
    }
}

/// Model çıkarımının çekirdeğe bağlandığı arayüz (Detector → Flow → Mamba zinciri
/// tek bir `restore` çağrısının arkasındadır).
pub trait Restorer {
    /// Planın istediği her konum için tahmin yazar. Yazılmayan tahmin boş kalır
    /// ve o kare orijinal hâliyle çıkar.
    fn restore(&mut self, io: &mut WindowIo<'_>) -> Result<(), CoreError>;
}

/// Hiçbir pikseli değiştirmeyen onarıcı (altyapı testleri ve model öncesi aşama için).
pub struct IdentityRestorer;

impl Restorer for IdentityRestorer {
    fn restore(&mut self, _io: &mut WindowIo<'_>) -> Result<(), CoreError> {
        Ok(())
    }
}

/// Kare giriş/çıkış hattı: halka + zamanlayıcı + tahmin bellekleri + sahne algılama.
///
/// ```text
/// loop {
///     match hat.poll() {
///         Step::NeedFrames(n) => { /* en çok n kare it; kaynak bittiyse finish() */ }
///         Step::Window(_)     => { /* window_io() ile modeli çalıştır, complete_window() */ }
///         Step::Output(_)     => { /* pop_output() ile kareleri kodlayıcıya ver */ }
///         Step::Finished      => break,
///     }
/// }
/// ```
pub struct Pipeline {
    geo: FrameGeometry,
    ring: FrameRing,
    sched: Scheduler,
    scene: SceneDetector,
    est_emit: Vec<Estimate>,
    est_tail: Vec<Estimate>,
    est_cache: Vec<Estimate>,
    prepared: Option<u64>,
    auto_scene_cut: bool,
    integrity_check: bool,
    /// Maske kenarı alfa yumuşatması (piksel; 0 = kapalı). Bkz. [`finish`].
    feather_px: u32,
    /// Kumlanma eşleme açık mı.
    grain: bool,
    stats: PipelineStats,
}

impl Pipeline {
    /// Verilen geometri için hat kurar (yığın belleği).
    pub fn new(geo: FrameGeometry) -> Result<Self, CoreError> {
        Self::with_storage(geo, StorageKind::Heap)
    }

    /// Halka belleğinin türünü seçerek hat kurar (bkz. [`FrameRing::with_storage`]).
    pub fn with_storage(geo: FrameGeometry, kind: StorageKind) -> Result<Self, CoreError> {
        let estimates = |n: usize| (0..n).map(|_| Estimate::new(geo)).collect::<Vec<_>>();
        Ok(Self {
            geo,
            ring: FrameRing::with_storage(geo, kind)?,
            sched: Scheduler::new(),
            scene: SceneDetector::new(geo, SceneCutConfig::default()),
            est_emit: estimates(STRIDE),
            est_tail: estimates(OVERLAP),
            est_cache: estimates(OVERLAP),
            prepared: None,
            auto_scene_cut: true,
            integrity_check: false,
            feather_px: 0,
            grain: false,
            stats: PipelineStats::default(),
        })
    }

    /// Kare geometrisi.
    pub fn geometry(&self) -> FrameGeometry {
        self.geo
    }

    /// Halka belleği (salt-okunur erişim ve tanılama için).
    pub fn ring(&self) -> &FrameRing {
        &self.ring
    }

    /// Otomatik sahne geçişi algılamayı açar/kapatır (varsayılan: açık).
    pub fn set_auto_scene_cut(&mut self, on: bool) {
        self.auto_scene_cut = on;
    }

    /// Sahne geçişi eşiklerini değiştirir.
    pub fn set_scene_config(&mut self, cfg: SceneCutConfig) {
        self.scene = SceneDetector::new(self.geo, cfg);
    }

    /// Her çıkan karede koruma kapısını yeniden denetler (yavaş; test/tanılama için).
    pub fn set_integrity_check(&mut self, on: bool) {
        self.integrity_check = on;
    }

    /// Maske kenarı alfa yumuşatması (0..=16 piksel; 0 = sert geçiş). Onarıcı, bölgeyi maskeden bu kadar
    /// fazla genişletmelidir. Üretim değeri 4.
    pub fn set_feather(&mut self, px: u32) -> Result<(), CoreError> {
        if px > 16 {
            return Err(CoreError::BadArgument);
        }
        self.feather_px = px;
        Ok(())
    }

    /// Kumlanma eşleme (onarılan bölgeye kameranın ölçülen gürültüsü eklenir).
    pub fn set_grain(&mut self, on: bool) {
        self.grain = on;
    }

    /// Sayaçlar.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            sched: self.sched.stats(),
            ..self.stats
        }
    }

    /// Henüz karara bağlanmamış bir kare için işleme ipucu.
    pub fn set_hint(&mut self, seq: u64, hint: Hint) -> Result<(), CoreError> {
        self.sched.set_hint(seq, hint)
    }

    /// Kaynağın bittiğini bildirir; kalan kareler dolgulu pencerelerle çıkarılır.
    pub fn finish(&mut self) {
        self.sched.on_eos();
    }

    /// Sıradaki adım.
    pub fn poll(&mut self) -> Step {
        let step = self.sched.poll();
        self.ring.retire_before(self.sched.retired());
        if let Step::Window(plan) = step
            && self.prepared != Some(plan.id)
        {
            for p in plan.emit.0..plan.emit.1 {
                let est = &mut self.est_emit[(p - plan.emit.0) as usize];
                est.clear();
                est.prior = plan.prior_at(p as usize);
            }
            for p in plan.cache.0..plan.cache.1 {
                let est = &mut self.est_tail[(p - STRIDE as u8) as usize];
                est.clear();
                est.prior = plan.prior_at(p as usize);
            }
            self.prepared = Some(plan.id);
        }
        step
    }

    /// Halkadaki boş yuvaya doğrudan yazarak kare ekler. `fill`, NV12 belleğini
    /// doldurup `(zaman damgası, bayraklar)` döndürür; `None` dönerse kare eklenmez.
    pub fn push_with<F>(&mut self, fill: F) -> Result<Option<FrameMeta>, CoreError>
    where
        F: FnOnce(&mut [u8]) -> Result<Option<(i64, u32)>, CoreError>,
    {
        if self.sched.is_eos() {
            return Err(CoreError::StreamFinished);
        }
        if self.sched.capacity_left() == 0 {
            return Err(CoreError::RingFull);
        }
        let mut lease = self.ring.begin_write()?;
        let Some((pts_us, mut flags)) = fill(lease.data())? else {
            return Ok(None);
        };
        let verdict = self.scene.feed(&lease.data()[..self.geo.luma_bytes()]);
        let first = self.sched.pushed() == 0;
        if first || (self.auto_scene_cut && verdict.cut) {
            flags |= FRAME_FLAG_SCENE_CUT;
        }
        let meta = lease.commit(pts_us, flags)?;
        let cut = flags & FRAME_FLAG_SCENE_CUT != 0 && !first;
        let seq = self.sched.on_push(cut)?;
        debug_assert_eq!(seq, meta.seq);
        self.stats.frames_in += 1;
        if cut {
            self.stats.scene_cuts += 1;
        }
        self.stats.max_occupancy = self.stats.max_occupancy.max(self.ring.len() as u32);
        Ok(Some(meta))
    }

    /// Hazır bir NV12 karesini kopyalayarak ekler.
    pub fn push_frame(
        &mut self,
        nv12: &[u8],
        pts_us: i64,
        flags: u32,
    ) -> Result<FrameMeta, CoreError> {
        let bytes = self.geo.frame_bytes();
        if nv12.len() < bytes {
            return Err(CoreError::BufferTooSmall);
        }
        let meta = self.push_with(|dst| {
            dst.copy_from_slice(&nv12[..bytes]);
            Ok(Some((pts_us, flags)))
        })?;
        meta.ok_or(CoreError::Internal)
    }

    /// Verilmiş pencerenin giriş kareleri ve yazılacak tahminleri.
    pub fn window_io(&mut self) -> Result<WindowIo<'_>, CoreError> {
        let plan = self.sched.issued().ok_or(CoreError::BadState)?;
        if self.prepared != Some(plan.id) {
            return Err(CoreError::BadState);
        }
        Ok(WindowIo {
            plan,
            geo: self.geo,
            ring: &self.ring,
            emit: &mut self.est_emit,
            tail: &mut self.est_tail,
        })
    }

    /// Model çıktılarının yazıldığını bildirir; kesinleşen kareler çıkış kuyruğuna girer.
    pub fn complete_window(&mut self) -> Result<(), CoreError> {
        let plan = self.sched.issued().ok_or(CoreError::BadState)?;
        if self.sched.complete(plan.id)? {
            // Yeni ikinci yarı tahminleri önbellek olur; eski önbellek (est_tail)
            // çıkış kuyruğu boşalana kadar harmanlama kaynağı olarak kalır.
            std::mem::swap(&mut self.est_cache, &mut self.est_tail);
        }
        self.prepared = None;
        Ok(())
    }

    /// Sıradaki kesinleşmiş kareyi `dst` içine NV12 olarak yazar.
    ///
    /// Çıktı her zaman orijinal karenin kopyasıyla başlar; yalnızca tahminlerin
    /// işaretlediği pikseller üzerine yazılır (bit-exact koruma kapısı).
    pub fn pop_output(&mut self, dst: &mut [u8]) -> Result<Option<OutputMeta>, CoreError> {
        let Some(item) = self.sched.peek_output() else {
            return Ok(None);
        };
        let bytes = self.geo.frame_bytes();
        if dst.len() < bytes {
            return Err(CoreError::BufferTooSmall);
        }
        let dst = &mut dst[..bytes];
        let lease = self.ring.read(item.seq)?;
        dst.copy_from_slice(lease.data());
        let mut modified = 0;
        if let OutputSource::Estimate {
            emit_index,
            fuse_index,
        } = item.source
        {
            let est = &self.est_emit[emit_index as usize];
            let cached = fuse_index
                .map(|i| &self.est_tail[i as usize])
                .filter(|c| !c.roi.is_empty());
            modified = compose(self.geo, dst, est, cached);
            if self.feather_px > 0 || self.grain {
                let roi = match cached {
                    Some(c) if !est.roi.is_empty() => est.roi.union(c.roi),
                    Some(c) => c.roi,
                    None => est.roi,
                };
                let region =
                    |i: usize| est.weight[i] != 0 || cached.is_some_and(|c| c.weight[i] != 0);
                finish_roi(
                    self.geo,
                    lease.data(),
                    dst,
                    &region,
                    roi,
                    self.feather_px,
                    self.grain,
                    item.seq,
                );
            }
            if self.integrity_check {
                self.stats.preservation_violations +=
                    verify_preservation(self.geo, lease.data(), dst, est, cached);
            }
        }
        let frame = lease.meta();
        drop(lease);
        self.sched.pop_output();

        if item.seq != self.stats.frames_out {
            self.stats.order_violations += 1;
        }
        self.stats.frames_out += 1;
        self.stats.modified_pixels += modified;
        let mut flags = item.flags;
        if frame.flags & FRAME_FLAG_SCENE_CUT != 0 {
            flags |= OUT_FLAG_SCENE_START;
        }
        Ok(Some(OutputMeta {
            seq: item.seq,
            pts_us: frame.pts_us,
            flags,
            modified_pixels: modified,
        }))
    }

    /// Hattı baştan sona sürer: `source` kare üretir (`None` = kaynak bitti),
    /// `restorer` pencereleri onarır, `sink` kesinleşen kareleri alır.
    pub fn run<S, R, K>(
        &mut self,
        mut source: S,
        restorer: &mut R,
        mut sink: K,
    ) -> Result<(), CoreError>
    where
        S: FnMut(&mut [u8]) -> Option<(i64, u32)>,
        R: Restorer + ?Sized,
        K: FnMut(&[u8], &OutputMeta),
    {
        let mut out = vec![0u8; self.geo.frame_bytes()];
        loop {
            match self.poll() {
                Step::NeedFrames(n) => {
                    for _ in 0..n {
                        if self.push_with(|dst| Ok(source(dst)))?.is_none() {
                            self.finish();
                            break;
                        }
                    }
                }
                Step::Window(_) => {
                    let mut io = self.window_io()?;
                    restorer.restore(&mut io)?;
                    self.complete_window()?;
                }
                Step::Output(_) => {
                    while let Some(meta) = self.pop_output(&mut out)? {
                        sink(&out, &meta);
                    }
                }
                Step::Finished => return Ok(()),
            }
        }
    }
}

// ============================================================================
// 9. Düzlemsel YUV 4:2:0 ↔ NV12 (MediaCodec Image düzlemleri için)
// ============================================================================

/// Satır ve piksel adımlı tek bir düzlem (ham işaretçi; U ve V düzlemleri aynı
/// belleğin iç içe görünümleri olabileceği için dilim kullanılmaz).
#[derive(Debug, Clone, Copy)]
pub struct RawPlane {
    /// Düzlemin ilk baytı.
    pub ptr: *mut u8,
    /// Erişilebilir bayt sayısı.
    pub len: usize,
    /// Satırlar arası bayt adımı.
    pub row_stride: usize,
    /// Aynı satırdaki örnekler arası bayt adımı (Y için 1; UV için 1 veya 2).
    pub pixel_stride: usize,
}

impl RawPlane {
    /// `cols × rows` örnek için gereken en küçük uzunluk sağlanıyor mu?
    fn fits(&self, cols: usize, rows: usize) -> bool {
        !self.ptr.is_null()
            && self.pixel_stride >= 1
            && self.row_stride > (cols - 1) * self.pixel_stride
            && self.len > (rows - 1) * self.row_stride + (cols - 1) * self.pixel_stride
    }
}

fn check_planes(
    geo: FrameGeometry,
    y: &RawPlane,
    u: &RawPlane,
    v: &RawPlane,
    packed_y: bool,
) -> Result<(), CoreError> {
    let (w, h) = (geo.width(), geo.height());
    let uv_ok = u.fits(w / 2, h / 2)
        && v.fits(w / 2, h / 2)
        && u.row_stride == v.row_stride
        && u.pixel_stride == v.pixel_stride;
    if (y.pixel_stride == 1 || !packed_y) && y.fits(w, h) && uv_ok {
        Ok(())
    } else {
        Err(CoreError::BufferTooSmall)
    }
}

/// Üç düzlemli YUV 4:2:0 görüntüyü (I420, NV12 veya NV21 yerleşimi) sıkı NV12'ye paketler.
///
/// # Safety
/// Her düzlemin `ptr..ptr+len` aralığı çağrı boyunca okunabilir olmalıdır.
pub unsafe fn pack_nv12(
    geo: FrameGeometry,
    y: RawPlane,
    u: RawPlane,
    v: RawPlane,
    dst: &mut [u8],
) -> Result<(), CoreError> {
    check_planes(geo, &y, &u, &v, false)?;
    if dst.len() < geo.frame_bytes() {
        return Err(CoreError::BufferTooSmall);
    }
    let (w, h) = (geo.width(), geo.height());
    let (dst_y, dst_uv) = dst[..geo.frame_bytes()].split_at_mut(geo.luma_bytes());
    // SAFETY: check_planes, aşağıda okunan her ofsetin düzlem uzunluğu içinde olduğunu doğruladı.
    unsafe {
        for row in 0..h {
            let src = y.ptr.add(row * y.row_stride);
            if y.pixel_stride == 1 {
                std::ptr::copy_nonoverlapping(src, dst_y.as_mut_ptr().add(row * w), w);
            } else {
                // Örnek başına birden çok bayt (ör. 10 bit P010'un üst baytı).
                for (x, out) in dst_y[row * w..(row + 1) * w].iter_mut().enumerate() {
                    *out = *src.add(x * y.pixel_stride);
                }
            }
        }
        let ps = u.pixel_stride;
        for cy in 0..h / 2 {
            let out = &mut dst_uv[cy * w..(cy + 1) * w];
            let (ur, vr) = (u.ptr.add(cy * u.row_stride), v.ptr.add(cy * v.row_stride));
            if ps == 2 && vr == ur.add(1) {
                // NV12: U düzlemi zaten "U V U V…"; son V baytı yalnızca V düzleminden okunabilir.
                std::ptr::copy_nonoverlapping(ur, out.as_mut_ptr(), w - 1);
                out[w - 1] = *vr.add(w - 2);
            } else {
                for cx in 0..w / 2 {
                    out[2 * cx] = *ur.add(cx * ps);
                    out[2 * cx + 1] = *vr.add(cx * ps);
                }
            }
        }
    }
    Ok(())
}

/// Sıkı NV12 kareyi üç düzlemli hedefe (kodlayıcı giriş görüntüsü) açar.
///
/// # Safety
/// Her düzlemin `ptr..ptr+len` aralığı çağrı boyunca yazılabilir olmalıdır.
pub unsafe fn unpack_nv12(
    geo: FrameGeometry,
    src: &[u8],
    y: RawPlane,
    u: RawPlane,
    v: RawPlane,
) -> Result<(), CoreError> {
    check_planes(geo, &y, &u, &v, true)?;
    if src.len() < geo.frame_bytes() {
        return Err(CoreError::BufferTooSmall);
    }
    let (w, h) = (geo.width(), geo.height());
    let (src_y, src_uv) = src[..geo.frame_bytes()].split_at(geo.luma_bytes());
    // SAFETY: check_planes, aşağıda yazılan her ofsetin düzlem uzunluğu içinde olduğunu doğruladı.
    unsafe {
        for row in 0..h {
            std::ptr::copy_nonoverlapping(
                src_y.as_ptr().add(row * w),
                y.ptr.add(row * y.row_stride),
                w,
            );
        }
        let ps = u.pixel_stride;
        for cy in 0..h / 2 {
            let row = &src_uv[cy * w..(cy + 1) * w];
            let (ur, vr) = (u.ptr.add(cy * u.row_stride), v.ptr.add(cy * v.row_stride));
            if ps == 2 && vr == ur.add(1) {
                std::ptr::copy_nonoverlapping(row.as_ptr(), ur, w - 1);
                *vr.add(w - 2) = row[w - 1];
            } else {
                for cx in 0..w / 2 {
                    *ur.add(cx * ps) = row[2 * cx];
                    *vr.add(cx * ps) = row[2 * cx + 1];
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// 9b. VC-Core motoru (planın JNI arayüzü: vcInitEngine / vcPush* / vcProcessWindow / vcRunTesterAudit)
// ============================================================================

/// Motorun pencere politikası.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// Mod B (11+4); zor kare ipucu gelirse Mod A (7+1+7). Varsayılan üretim modu.
    Auto,
    /// Yalnızca Mod B (zor kare ipuçları yok sayılır).
    B,
    /// Her kare Mod A ile (en yüksek hassasiyet, en yavaş).
    A,
}

impl EngineMode {
    /// JNI tamsayısından: 0 = otomatik, 1 = yalnız B, 2 = yalnız A.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(EngineMode::Auto),
            1 => Some(EngineMode::B),
            2 => Some(EngineMode::A),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            EngineMode::Auto => "auto",
            EngineMode::B => "B",
            EngineMode::A => "A",
        }
    }
}

/// VC-Tester çalışma zamanı sayaçları (motor düzeyi; hat sayaçlarına ek).
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditCounters {
    /// Donanım tamponundan (HardwareBuffer) itilen kareler.
    pub frames_hw: u64,
    /// CPU tamponundan itilen kareler.
    pub frames_cpu: u64,
    /// Zaman damgası artan sırada olmadığı için reddedilen kareler.
    pub pts_rejected: u64,
    /// İşlenen pencere sayısı ve süreleri (µs).
    pub windows_processed: u64,
    pub window_us_total: u64,
    pub window_us_max: u64,
    /// Giriş ve çıkış Y düzlemlerinin sırayla FNV-1a özetleri (kimlik onarıcıda eşit olmalı).
    pub input_digest: u64,
    pub output_digest: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Yüklü `.so` (veya test ikilisinin) PT_LOAD bölütlerinin en küçük hizası. 16 KB sayfalı cihazlar
/// için ≥ 16384 olmalıdır. Bulunamazsa 0.
pub fn elf_load_alignment() -> u64 {
    #[cfg(all(
        any(target_os = "linux", target_os = "android"),
        target_pointer_width = "64"
    ))]
    {
        use std::ffi::c_char;
        #[repr(C)]
        struct Phdr {
            p_type: u32,
            p_flags: u32,
            p_offset: u64,
            p_vaddr: u64,
            p_paddr: u64,
            p_filesz: u64,
            p_memsz: u64,
            p_align: u64,
        }
        #[repr(C)]
        struct DlPhdrInfo {
            addr: usize,
            name: *const c_char,
            phdr: *const Phdr,
            phnum: u16,
        }
        unsafe extern "C" {
            fn dl_iterate_phdr(
                cb: unsafe extern "C" fn(*mut DlPhdrInfo, usize, *mut c_void) -> i32,
                data: *mut c_void,
            ) -> i32;
        }
        struct Search {
            target: usize,
            align: u64,
        }
        unsafe extern "C" fn visit(info: *mut DlPhdrInfo, _size: usize, data: *mut c_void) -> i32 {
            // SAFETY: dinamik bağlayıcı geçerli bilgi verir; data bizim Search yapımızdır.
            unsafe {
                let info = &*info;
                let search = &mut *(data as *mut Search);
                let headers = std::slice::from_raw_parts(info.phdr, info.phnum as usize);
                let contains = headers.iter().any(|h| {
                    h.p_type == 1 && {
                        let start = info.addr + h.p_vaddr as usize;
                        search.target >= start && search.target < start + h.p_memsz as usize
                    }
                });
                if contains {
                    search.align = headers
                        .iter()
                        .filter(|h| h.p_type == 1)
                        .map(|h| h.p_align)
                        .min()
                        .unwrap_or(0);
                    return 1;
                }
                0
            }
        }
        let mut search = Search {
            target: elf_load_alignment as fn() -> u64 as usize,
            align: 0,
        };
        // SAFETY: geri çağırma yalnızca bu çağrı boyunca kullanılır.
        unsafe { dl_iterate_phdr(visit, &mut search as *mut Search as *mut c_void) };
        search.align
    }
    #[cfg(not(all(
        any(target_os = "linux", target_os = "android"),
        target_pointer_width = "64"
    )))]
    {
        0
    }
}

/// VC-Core motoru: hat (halka + zamanlayıcı) geometrisi ilk karede belirlenir.
pub struct Engine {
    mode: EngineMode,
    prefer_vulkan: bool,
    storage: StorageKind,
    pipeline: Option<Pipeline>,
    force_a: bool,
    audit: AuditCounters,
}

impl Engine {
    /// `storage`: halka belleği tercihi (Android'de donanım tamponu; yoksa yığın).
    pub fn new(mode: EngineMode, prefer_vulkan: bool, storage: StorageKind) -> Self {
        Self {
            mode,
            prefer_vulkan,
            storage,
            pipeline: None,
            force_a: mode == EngineMode::A,
            audit: AuditCounters::default(),
        }
    }

    /// Hat (ilk karede kurulur).
    pub fn pipeline(&self) -> Option<&Pipeline> {
        self.pipeline.as_ref()
    }

    /// Hattı ilk karede kurar; sonraki karelerde geometri değişemez.
    fn ensure(&mut self, width: usize, height: usize) -> Result<&mut Pipeline, CoreError> {
        let geo = FrameGeometry::new(width & !1, height & !1)?;
        if self.pipeline.is_none() {
            let mut pipeline = Pipeline::with_storage(geo, self.storage)?;
            pipeline.set_integrity_check(true);
            self.pipeline = Some(pipeline);
        }
        let pipeline = self.pipeline.as_mut().ok_or(CoreError::Internal)?;
        if pipeline.geometry() != geo {
            return Err(CoreError::InvalidGeometry);
        }
        Ok(pipeline)
    }

    fn after_push(
        &mut self,
        result: Result<FrameMeta, CoreError>,
        hardware: bool,
    ) -> Result<u64, CoreError> {
        match result {
            Ok(meta) => {
                if hardware {
                    self.audit.frames_hw += 1;
                } else {
                    self.audit.frames_cpu += 1;
                }
                let pipeline = self.pipeline.as_mut().ok_or(CoreError::Internal)?;
                if let Ok(frame) = pipeline.ring().read(meta.seq) {
                    self.audit.input_digest = fnv1a(
                        self.audit.input_digest ^ FNV_OFFSET,
                        &frame.data()[..pipeline.geometry().luma_bytes()],
                    );
                }
                if self.force_a && self.mode != EngineMode::B {
                    let _ = pipeline.set_hint(meta.seq, Hint::Precision);
                }
                Ok(meta.seq)
            }
            Err(CoreError::NonMonotonicPts) => {
                self.audit.pts_rejected += 1;
                Err(CoreError::NonMonotonicPts)
            }
            Err(e) => Err(e),
        }
    }

    /// Üç düzlemli YUV 4:2:0 kare (MediaCodec Image veya kilitlenmiş AHardwareBuffer düzlemleri).
    ///
    /// # Safety
    /// Düzlem işaretçileri çağrı boyunca okunabilir olmalıdır.
    pub unsafe fn push_planes(
        &mut self,
        width: usize,
        height: usize,
        y: RawPlane,
        u: RawPlane,
        v: RawPlane,
        pts_us: i64,
        flags: u32,
        hardware: bool,
    ) -> Result<u64, CoreError> {
        let pipeline = self.ensure(width, height)?;
        let geo = pipeline.geometry();
        let result = pipeline
            .push_with(|dst| {
                // SAFETY: çağıranın sözleşmesi.
                unsafe { pack_nv12(geo, y, u, v, dst)? };
                Ok(Some((pts_us, flags)))
            })
            .and_then(|m| m.ok_or(CoreError::Internal));
        self.after_push(result, hardware)
    }

    /// Sıkı NV12 kare (CPU yolu).
    pub fn push_nv12(
        &mut self,
        width: usize,
        height: usize,
        nv12: &[u8],
        pts_us: i64,
        flags: u32,
    ) -> Result<u64, CoreError> {
        let pipeline = self.ensure(width, height)?;
        let result = pipeline.push_frame(nv12, pts_us, flags);
        self.after_push(result, false)
    }

    /// Sıradaki adım (hat kurulmadıysa kare ister).
    pub fn poll(&mut self) -> Step {
        match self.pipeline.as_mut() {
            Some(p) => p.poll(),
            None => Step::NeedFrames(WINDOW_FRAMES),
        }
    }

    /// Hazır pencere varsa onarıcıyla işler. 0: pencere yok, 1: Mod B, 2: Mod A.
    /// `force_mode_a`: bundan sonra itilen karelere Mod A ipucu verilir (yalnız-B modunda yok sayılır).
    pub fn process_window(
        &mut self,
        force_mode_a: bool,
        restorer: &mut dyn Restorer,
    ) -> Result<i32, CoreError> {
        if force_mode_a {
            self.force_a = true;
        }
        let Some(pipeline) = self.pipeline.as_mut() else {
            return Ok(0);
        };
        let Step::Window(plan) = pipeline.poll() else {
            return Ok(0);
        };
        let started = std::time::Instant::now();
        {
            let mut io = pipeline.window_io()?;
            restorer.restore(&mut io)?;
        }
        pipeline.complete_window()?;
        let us = started.elapsed().as_micros() as u64;
        self.audit.windows_processed += 1;
        self.audit.window_us_total += us;
        self.audit.window_us_max = self.audit.window_us_max.max(us);
        Ok(if plan.mode == Mode::A { 2 } else { 1 })
    }

    /// Sıradaki çıkış karesi (NV12).
    pub fn pop_output(&mut self, dst: &mut [u8]) -> Result<Option<OutputMeta>, CoreError> {
        let Some(pipeline) = self.pipeline.as_mut() else {
            return Ok(None);
        };
        let out = pipeline.pop_output(dst)?;
        if out.is_some() {
            let luma = pipeline.geometry().luma_bytes();
            self.audit.output_digest = fnv1a(self.audit.output_digest ^ FNV_OFFSET, &dst[..luma]);
        }
        Ok(out)
    }

    /// Kaynak bitti.
    pub fn finish(&mut self) {
        if let Some(p) = self.pipeline.as_mut() {
            p.finish();
        }
    }

    /// Motor sayaçları.
    pub fn audit(&self) -> AuditCounters {
        self.audit
    }

    /// VC-Tester çalışma zamanı raporu (JSON, yalnız ASCII). Alanlar:
    /// mediaSync (kare/PTS/sıra), preservation (koruma kapısı, özet eşitliği), scheduler (Mod A/B),
    /// platform (sayfa boyutu, ELF LOAD hizası, bellek türü ve boyutu), runtime (model durumu).
    pub fn tester_audit_json(&self) -> String {
        let st = self
            .pipeline
            .as_ref()
            .map(|p| p.stats())
            .unwrap_or_default();
        let ring_bytes = self
            .pipeline
            .as_ref()
            .map(|p| p.ring().pool_bytes())
            .unwrap_or(0);
        let storage = match self
            .pipeline
            .as_ref()
            .map(|p| (p.ring().storage_kind(), p.ring().gpu_importable()))
        {
            Some((StorageKind::HardwareBuffer, true)) => "hardware_buffer_gpu",
            Some((StorageKind::HardwareBuffer, false)) => "hardware_buffer_cpu",
            Some((StorageKind::Heap, _)) => "heap",
            None => "none",
        };
        let (w, h) = self
            .pipeline
            .as_ref()
            .map(|p| (p.geometry().width(), p.geometry().height()))
            .unwrap_or((0, 0));
        let align = elf_load_alignment();
        let page = page_size();
        let a = self.audit;
        let frames_ok = st.frames_in == st.frames_out;
        let bit_exact = st.modified_pixels == 0 && a.input_digest == a.output_digest;
        let kernel = match kernels::backend_id() {
            1 => "avx2",
            2 => "neon",
            _ => "portable",
        };
        let avg = a
            .window_us_total
            .checked_div(a.windows_processed)
            .unwrap_or(0);
        format!(
            concat!(
                "{{\"abi\":{},\"version\":\"{}\",\"mode\":\"{}\",\"size\":\"{}x{}\",",
                "\"mediaSync\":{{\"frames_in\":{},\"frames_out\":{},\"frames_equal\":{},\"pts_rejected\":{},\"order_violations\":{},\"frames_hw\":{},\"frames_cpu\":{}}},",
                "\"preservation\":{{\"violations\":{},\"modified_pixels\":{},\"input_digest\":\"{:016x}\",\"output_digest\":\"{:016x}\",\"bit_exact\":{}}},",
                "\"scheduler\":{{\"windows_a\":{},\"windows_b\":{},\"bridges\":{},\"bypassed\":{},\"fused\":{},\"scene_cuts\":{},\"window_us_avg\":{},\"window_us_max\":{}}},",
                "\"platform\":{{\"page_size\":{},\"elf_load_align\":{},\"elf_16k_ok\":{},\"kernel\":\"{}\",\"storage\":\"{}\",\"ring_bytes\":{},\"max_occupancy\":{}}},",
                "\"runtime\":{{\"vulkan\":\"{}\",\"models\":\"none (identity restorer; weights not loaded)\"}}}}"
            ),
            ABI_VERSION,
            version(),
            self.mode.name(),
            w,
            h,
            st.frames_in,
            st.frames_out,
            frames_ok,
            a.pts_rejected,
            st.order_violations,
            a.frames_hw,
            a.frames_cpu,
            st.preservation_violations,
            st.modified_pixels,
            a.input_digest,
            a.output_digest,
            bit_exact,
            st.sched.windows_a,
            st.sched.windows_b,
            st.sched.bridges,
            st.sched.bypassed,
            st.sched.fused,
            st.scene_cuts,
            avg,
            a.window_us_max,
            page,
            align,
            align >= 16384,
            kernel,
            storage,
            ring_bytes,
            st.max_occupancy,
            if self.prefer_vulkan {
                "requested, not used (no compute shaders yet)"
            } else {
                "off"
            },
        )
    }
}

// ============================================================================
// 9b. Yeniden kullanım planlayıcısı — SAD ile algılayıcı atlama ve statik yama yeniden kullanımı
// ============================================================================

/// Kare başına hangi sinir ağı aşamalarının gerçekten çalışacağına karar verir (telefonda süre).
///
/// Gözlem: altyazı ortalama 2–3 sn (60–90 kare) aynı yerde sabit durur, filigran hiç kıpırdamaz. Şerit
/// (ROI) parlaklık düzlemi bir önceki kareyle Assembly `sad_u8` ile üç bölgede karşılaştırılır (piksel
/// başına ortalama mutlak fark):
///
/// * **Yazı pikselleri** (ham maske): altyazının kendisi değişti mi?
/// * **Çevre halkası** (maskenin `gate_passes` × 2 px genişletilmiş hâlinin, yani geri yazma bölgesinin,
///   dışındaki `ring_passes` × 2 px şerit): yazının ARKASINDAKİ sahne
///   hareket ediyor mu? Yazı pikselleri parlak ve baskın olduğundan tüm şerit ya da yalnızca yazı üzerinden
///   ölçülen fark düşük çıkabilir; hareketli arka planın önünde sabit duran altyazıda önceki yamayı
///   yapıştırmak "donmuş dikdörtgen" üretir. Halka bunu yakalar.
/// * **Sütun blokları** (32 px): şeridin herhangi bir yerinde YENİ yazı belirdi mi (en büyük blok farkı)?
///
/// Kararlar:
/// * **Algılayıcı**: yazı bölgesi ve bloklar değişmediyse önceki maske kullanılır (arka plan hareket etse
///   bile); en geç her `det_period` karede bir yine çalışır.
/// * **Yama yeniden kullanımı** (akış + onarım atlanır): yazı bölgesi VE çevre halkası neredeyse aynıysa
///   (arka plan gerçekten durağan, ör. sabit kamera) ve maske değişmediyse; en fazla `max_reuse` kare art
///   arda. Arka plan hareketliyse algılayıcı atlanır ama mevcut maskeyle akış + onarım çalışmaya devam eder.
///
/// Planlayıcı yalnızca karar verir; maske/yamayı saklamak çağıranın (halka tampon) işidir.
pub mod reuse {
    use crate::{dilate5x5, kernels};

    /// Eşikler (piksel başına ortalama mutlak fark, 0..255 ölçeğinde).
    #[derive(Clone, Copy, Debug)]
    pub struct Config {
        /// Yazı bölgesi bu farkı aşarsa algılayıcı çalışır.
        pub det_threshold: f32,
        /// Herhangi bir 32 px sütun bloğu bu farkı aşarsa (yeni yazı) algılayıcı çalışır.
        pub new_text_threshold: f32,
        pub det_period: u32,
        /// Yazı bölgesi ve çevre halkası bu farkın altındaysa yama yeniden kullanılır.
        pub reuse_threshold: f32,
        pub max_reuse: u32,
        /// Yazı bölgesi genişletmesi: geçiş başına 2 px (5×5).
        pub gate_passes: u32,
        /// Çevre halkası genişliği: geçiş başına 2 px (varsayılan 4 → 8 px).
        pub ring_passes: u32,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                det_threshold: 2.0,
                new_text_threshold: 20.0,
                det_period: 10,
                reuse_threshold: 0.5,
                max_reuse: 60,
                gate_passes: 1,
                ring_passes: 4,
            }
        }
    }

    /// Bir kare için karar ve ölçülen farklar (×100, tanılama).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Plan {
        /// Algılayıcı bu karede çalışsın mı (yoksa önceki maske).
        pub run_detector: bool,
        /// Önceki onarılmış yama aynen kullanılsın mı (akış + onarım atlanır).
        pub reuse_patch: bool,
        pub text_sad_x100: u32,
        pub ring_sad_x100: u32,
        pub block_max_x100: u32,
    }

    #[derive(Clone, Debug, Default)]
    pub struct Stats {
        pub frames: u64,
        pub detector_runs: u64,
        pub reused: u64,
    }

    /// Maskeden yazı bölgesi (genişletilmiş) ve çevre halkası: (yazı, halka) 0/255 baytları.
    pub fn regions(
        mask: &[u8],
        width: usize,
        height: usize,
        gate_passes: u32,
        ring_passes: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut gate = mask[..width * height].to_vec();
        let mut tmp = vec![0u8; width * height];
        for _ in 0..gate_passes {
            dilate5x5(&gate, width, height, &mut tmp);
            std::mem::swap(&mut gate, &mut tmp);
        }
        let mut outer = gate.clone();
        for _ in 0..ring_passes {
            dilate5x5(&outer, width, height, &mut tmp);
            std::mem::swap(&mut outer, &mut tmp);
        }
        let ring = outer
            .iter()
            .zip(&gate)
            .map(|(&o, &g)| if o != 0 && g == 0 { 255 } else { 0 })
            .collect();
        (gate, ring)
    }

    /// `region` (0 dışı) piksellerinde ortalama mutlak fark; bölge boşsa 0.
    fn masked_sad(
        a: &[u8],
        b: &[u8],
        region: &[u8],
        buf_a: &mut Vec<u8>,
        buf_b: &mut Vec<u8>,
    ) -> f32 {
        buf_a.clear();
        buf_b.clear();
        for ((&x, &y), &m) in a.iter().zip(b).zip(region) {
            if m != 0 {
                buf_a.push(x);
                buf_b.push(y);
            }
        }
        if buf_a.is_empty() {
            0.0
        } else {
            kernels::sad_u8(buf_a, buf_b) as f32 / buf_a.len() as f32
        }
    }

    /// 32 px sütun blokları içinde en büyük ortalama mutlak fark.
    fn block_max(a: &[u8], b: &[u8], width: usize, height: usize) -> f32 {
        let mut best = 0.0f32;
        let mut x = 0;
        while x < width {
            let w = 32.min(width - x);
            let mut sum = 0u64;
            for y in 0..height {
                let row = y * width + x;
                sum += kernels::sad_u8(&a[row..row + w], &b[row..row + w]);
            }
            best = best.max(sum as f32 / (w * height) as f32);
            x += w;
        }
        best
    }

    pub struct Planner {
        cfg: Config,
        prev: Vec<u8>,
        since_detector: u32,
        reuse_run: u32,
        scratch: (Vec<u8>, Vec<u8>),
        pub stats: Stats,
    }

    impl Planner {
        pub fn new(cfg: Config) -> Self {
            Self {
                cfg,
                prev: Vec::new(),
                since_detector: 0,
                reuse_run: 0,
                scratch: (Vec::new(), Vec::new()),
                stats: Stats::default(),
            }
        }

        /// Yeni sahne / sahne kesmesi: bir sonraki kare her şeyi baştan hesaplar.
        pub fn reset(&mut self) {
            self.prev.clear();
            self.since_detector = 0;
            self.reuse_run = 0;
        }

        /// `band`: şeridin parlaklık baytları (`width × height`, her karede aynı boyut). `mask`: bu kare
        /// için bilinen (önceki) yazı maskesi, 0/255. `mask_changed`: maske bir öncekinden farklı mı.
        pub fn plan(
            &mut self,
            band: &[u8],
            mask: &[u8],
            width: usize,
            height: usize,
            mask_changed: bool,
        ) -> Plan {
            assert!(band.len() == width * height && mask.len() >= width * height);
            self.stats.frames += 1;
            let fresh = self.prev.len() != band.len();
            let (text, ring, blocks) = if fresh || band.is_empty() {
                (f32::INFINITY, f32::INFINITY, f32::INFINITY)
            } else {
                // Yazı değişimi YALNIZCA ham maske piksellerinde ölçülür: genişletilmiş bölge yazının
                // kenarındaki hareketli arka planı da içerir ve algılayıcıyı gereksiz yere tetikler.
                let (_, ring_region) = regions(
                    mask,
                    width,
                    height,
                    self.cfg.gate_passes,
                    self.cfg.ring_passes,
                );
                let (sa, sb) = &mut self.scratch;
                let text = masked_sad(band, &self.prev, &mask[..width * height], sa, sb);
                let ring = masked_sad(band, &self.prev, &ring_region, sa, sb);
                (text, ring, block_max(band, &self.prev, width, height))
            };
            if fresh {
                self.prev = band.to_vec();
            } else {
                self.prev.copy_from_slice(band);
            }
            let cfg = self.cfg;
            let run_detector = fresh
                || text > cfg.det_threshold
                || blocks > cfg.new_text_threshold
                || self.since_detector + 1 >= cfg.det_period;
            self.since_detector = if run_detector {
                0
            } else {
                self.since_detector + 1
            };
            let reuse_patch = !fresh
                && !run_detector
                && !mask_changed
                && text <= cfg.reuse_threshold
                && ring <= cfg.reuse_threshold
                && self.reuse_run < cfg.max_reuse;
            self.reuse_run = if reuse_patch { self.reuse_run + 1 } else { 0 };
            self.stats.detector_runs += run_detector as u64;
            self.stats.reused += reuse_patch as u64;
            let x100 = |v: f32| {
                if v.is_finite() {
                    (v * 100.0) as u32
                } else {
                    u32::MAX
                }
            };
            Plan {
                run_detector,
                reuse_patch,
                text_sad_x100: x100(text),
                ring_sad_x100: x100(ring),
                block_max_x100: x100(blocks),
            }
        }

        /// Algılayıcı çalıştıktan sonra maske değiştiyse yeniden kullanım iptal edilir.
        pub fn confirm_mask(&mut self, plan: &mut Plan, mask_changed: bool) {
            if mask_changed && plan.reuse_patch {
                plan.reuse_patch = false;
                self.reuse_run = 0;
                self.stats.reused -= 1;
            }
        }
    }
}

// ============================================================================
// 9c. SPS/VUI yeniden yazıcı (H.264 + HEVC) — renk ve piksel en-boy oranı
// ============================================================================

/// Kodlayıcının bit akışına yazdığı SPS VUI'sini, pikselleri değiştirmeden düzeltir.
///
/// Neden: gerçek cihazlarda (Galaxy Z Flip8 / Qualcomm c2.qti) kodlayıcı istenen BT.601'i yok sayıp
/// VUI'ye BT.709 yazıyor; Android 11 OMX kodlayıcıları piksel en-boy oranını (SAR) hiç yazmıyor.
/// Bu modül SPS'i Exp-Golomb ile VUI'ye kadar ayrıştırır, yalnızca `aspect_ratio_info` ve
/// `video_signal_type` (renk) bölümlerini yeniden yazar; SPS'in geri kalan bütün bitleri (zamanlama,
/// HRD, bitstream_restriction, HEVC uzantıları) bit bit aynen kopyalanır. Öykünme önleme baytları
/// (0x000003) kaldırılıp yeniden eklenir. Hedef zaten yazılı değerlerle aynıysa çıktı girdiyle bayt
/// bayt aynıdır (birim testleriyle doğrulanır).
pub mod vui {
    use std::fmt;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Codec {
        Avc,
        Hevc,
    }

    /// ISO/IEC 23091-2 renk kodları (H.264/HEVC VUI ile aynı numaralar).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Colour {
        pub primaries: u8,
        pub transfer: u8,
        pub matrix: u8,
        pub full_range: bool,
    }

    /// Yazılacak hedef. `None` olan bölüm olduğu gibi bırakılır.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Target {
        pub sar: Option<(u16, u16)>,
        pub colour: Option<Colour>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum VuiError {
        Truncated,
        Unsupported(&'static str),
    }

    impl fmt::Display for VuiError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                VuiError::Truncated => write!(f, "SPS beklenenden kısa"),
                VuiError::Unsupported(what) => write!(f, "desteklenmeyen SPS: {what}"),
            }
        }
    }

    /// SPS'ten okunan VUI değerleri (testler ve tanılama için).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Parsed {
        pub vui_present: bool,
        pub sar: Option<(u16, u16)>,
        pub aspect_idc: Option<u8>,
        pub colour: Option<Colour>,
        pub video_format: Option<u8>,
        /// video_signal_type varsa tam aralık bayrağı (renk açıklaması olmasa da).
        pub full_range: Option<bool>,
    }

    // ---- öykünme önleme --------------------------------------------------------

    /// NAL yükünden 0x000003 öykünme önleme baytlarını kaldırır (RBSP).
    pub fn unescape(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        let mut zeros = 0usize;
        for &b in data {
            if zeros >= 2 && b == 3 {
                zeros = 0;
                continue;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            out.push(b);
        }
        out
    }

    /// RBSP'ye öykünme önleme baytlarını ekler (00 00 0x, x ≤ 3 → 00 00 03 0x).
    pub fn escape(rbsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 4);
        let mut zeros = 0usize;
        for &b in rbsp {
            if zeros >= 2 && b <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            out.push(b);
        }
        out
    }

    // ---- bit okuyucu / yazıcı ---------------------------------------------------

    struct Reader<'a> {
        data: &'a [u8],
        pos: usize,
        end: usize,
    }

    impl<'a> Reader<'a> {
        fn new(data: &'a [u8], end: usize) -> Self {
            Reader { data, pos: 0, end }
        }
        fn bit(&mut self) -> Result<u32, VuiError> {
            if self.pos >= self.end {
                return Err(VuiError::Truncated);
            }
            let b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
            self.pos += 1;
            Ok(b as u32)
        }
        fn u(&mut self, n: u32) -> Result<u32, VuiError> {
            let mut v = 0u32;
            for _ in 0..n {
                v = (v << 1) | self.bit()?;
            }
            Ok(v)
        }
        fn skip(&mut self, n: usize) -> Result<(), VuiError> {
            if self.pos + n > self.end {
                return Err(VuiError::Truncated);
            }
            self.pos += n;
            Ok(())
        }
        fn ue(&mut self) -> Result<u32, VuiError> {
            let mut zeros = 0u32;
            while self.bit()? == 0 {
                zeros += 1;
                if zeros > 31 {
                    return Err(VuiError::Unsupported("Exp-Golomb taşması"));
                }
            }
            if zeros == 0 {
                return Ok(0);
            }
            Ok(((1u64 << zeros) - 1 + self.u(zeros)? as u64) as u32)
        }
        fn se(&mut self) -> Result<i32, VuiError> {
            let k = self.ue()? as i64;
            Ok(if k & 1 == 1 {
                ((k + 1) / 2) as i32
            } else {
                -(k / 2) as i32
            })
        }
    }

    struct Writer {
        out: Vec<u8>,
        bits: usize,
    }

    impl Writer {
        fn new() -> Self {
            Writer {
                out: Vec::new(),
                bits: 0,
            }
        }
        fn put(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                if self.bits.is_multiple_of(8) {
                    self.out.push(0);
                }
                if (v >> i) & 1 == 1 {
                    let last = self.out.len() - 1;
                    self.out[last] |= 1 << (7 - (self.bits % 8));
                }
                self.bits += 1;
            }
        }
        /// `data` içindeki [from, to) bit aralığını aynen kopyalar.
        fn copy(&mut self, data: &[u8], from: usize, to: usize) {
            for p in from..to {
                self.put(((data[p >> 3] >> (7 - (p & 7))) & 1) as u32, 1);
            }
        }
        /// rbsp_trailing_bits: durak biti ve bayt hizası.
        fn finish(mut self) -> Vec<u8> {
            self.put(1, 1);
            while !self.bits.is_multiple_of(8) {
                self.put(0, 1);
            }
            self.out
        }
    }

    /// RBSP'deki rbsp_stop_one_bit'in bit konumu (veri bu konumdan önce biter).
    fn stop_bit(rbsp: &[u8]) -> Result<usize, VuiError> {
        let last = rbsp
            .iter()
            .rposition(|&b| b != 0)
            .ok_or(VuiError::Truncated)?;
        let tz = rbsp[last].trailing_zeros() as usize;
        Ok(last * 8 + 7 - tz)
    }

    // ---- SPS ayrıştırma: VUI'ye kadar ----------------------------------------

    /// VUI bayrağının ve (varsa) VUI içindeki bölümlerin bit konumları.
    struct Layout {
        vui_flag: usize,
        vui_present: bool,
        aspect_start: usize,
        aspect_end: usize,
        overscan_end: usize,
        signal_end: usize,
        parsed: Parsed,
    }

    fn avc_scaling_list(r: &mut Reader, size: usize) -> Result<(), VuiError> {
        let (mut last, mut next) = (8i32, 8i32);
        for _ in 0..size {
            if next != 0 {
                let delta = r.se()?;
                next = (last + delta + 256).rem_euclid(256);
            }
            if next != 0 {
                last = next;
            }
        }
        Ok(())
    }

    fn avc_to_vui(r: &mut Reader) -> Result<(), VuiError> {
        let profile = r.u(8)?;
        r.skip(16)?; // constraint bayrakları + level_idc
        r.ue()?; // seq_parameter_set_id
        if matches!(
            profile,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            let chroma = r.ue()?;
            if chroma == 3 {
                r.skip(1)?;
            }
            r.ue()?;
            r.ue()?;
            r.skip(1)?;
            if r.bit()? == 1 {
                let lists = if chroma == 3 { 12 } else { 8 };
                for i in 0..lists {
                    if r.bit()? == 1 {
                        avc_scaling_list(r, if i < 6 { 16 } else { 64 })?;
                    }
                }
            }
        }
        r.ue()?; // log2_max_frame_num_minus4
        match r.ue()? {
            0 => {
                r.ue()?;
            }
            1 => {
                r.skip(1)?;
                r.se()?;
                r.se()?;
                let n = r.ue()?;
                if n > 255 {
                    return Err(VuiError::Unsupported("POC döngüsü"));
                }
                for _ in 0..n {
                    r.se()?;
                }
            }
            _ => {}
        }
        r.ue()?; // max_num_ref_frames
        r.skip(1)?;
        r.ue()?;
        r.ue()?;
        if r.bit()? == 0 {
            r.skip(1)?; // mb_adaptive_frame_field_flag
        }
        r.skip(1)?; // direct_8x8_inference_flag
        if r.bit()? == 1 {
            for _ in 0..4 {
                r.ue()?;
            }
        }
        Ok(())
    }

    fn hevc_profile_tier_level(r: &mut Reader, max_sub_layers_minus1: u32) -> Result<(), VuiError> {
        r.skip(88)?; // general_profile_space … general_inbld/reserved
        r.skip(8)?; // general_level_idc
        let mut prof = [false; 8];
        let mut lvl = [false; 8];
        for i in 0..max_sub_layers_minus1 as usize {
            prof[i] = r.bit()? == 1;
            lvl[i] = r.bit()? == 1;
        }
        if max_sub_layers_minus1 > 0 {
            for _ in max_sub_layers_minus1..8 {
                r.skip(2)?;
            }
        }
        for i in 0..max_sub_layers_minus1 as usize {
            if prof[i] {
                r.skip(88)?;
            }
            if lvl[i] {
                r.skip(8)?;
            }
        }
        Ok(())
    }

    fn hevc_scaling_list_data(r: &mut Reader) -> Result<(), VuiError> {
        for size_id in 0..4u32 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut matrix = 0;
            while matrix < 6 {
                if r.bit()? == 0 {
                    r.ue()?; // scaling_list_pred_matrix_id_delta
                } else {
                    let coefs = 64.min(1 << (4 + (size_id << 1)));
                    if size_id > 1 {
                        r.se()?;
                    }
                    for _ in 0..coefs {
                        r.se()?;
                    }
                }
                matrix += step;
            }
        }
        Ok(())
    }

    fn hevc_to_vui(r: &mut Reader) -> Result<(), VuiError> {
        r.skip(4)?; // sps_video_parameter_set_id
        let max_sub = r.u(3)?;
        r.skip(1)?;
        hevc_profile_tier_level(r, max_sub)?;
        r.ue()?; // sps_seq_parameter_set_id
        if r.ue()? == 3 {
            r.skip(1)?;
        }
        r.ue()?;
        r.ue()?;
        if r.bit()? == 1 {
            for _ in 0..4 {
                r.ue()?;
            }
        }
        r.ue()?;
        r.ue()?;
        let log2_max_poc_lsb = r.ue()? + 4;
        let ordering_all = r.bit()? == 1;
        let first = if ordering_all { 0 } else { max_sub };
        for _ in first..=max_sub {
            r.ue()?;
            r.ue()?;
            r.ue()?;
        }
        for _ in 0..6 {
            r.ue()?;
        }
        if r.bit()? == 1 && r.bit()? == 1 {
            hevc_scaling_list_data(r)?;
        }
        r.skip(2)?; // amp_enabled, sample_adaptive_offset_enabled
        if r.bit()? == 1 {
            r.skip(8)?;
            r.ue()?;
            r.ue()?;
            r.skip(1)?;
        }
        let num_sets = r.ue()? as usize;
        if num_sets > 64 {
            return Err(VuiError::Unsupported("st_ref_pic_set sayısı"));
        }
        let mut deltas: Vec<u32> = Vec::with_capacity(num_sets);
        for idx in 0..num_sets {
            let inter = idx != 0 && r.bit()? == 1;
            if inter {
                r.skip(1)?; // delta_rps_sign
                r.ue()?; // abs_delta_rps_minus1
                let reference = deltas[idx - 1];
                let mut count = 0;
                for _ in 0..=reference {
                    let used = r.bit()? == 1;
                    let use_delta = if used { true } else { r.bit()? == 1 };
                    if use_delta {
                        count += 1;
                    }
                }
                deltas.push(count);
            } else {
                let neg = r.ue()?;
                let pos = r.ue()?;
                if neg + pos > 32 {
                    return Err(VuiError::Unsupported("st_ref_pic_set boyutu"));
                }
                for _ in 0..neg + pos {
                    r.ue()?;
                    r.skip(1)?;
                }
                deltas.push(neg + pos);
            }
        }
        if r.bit()? == 1 {
            let n = r.ue()?;
            if n > 32 {
                return Err(VuiError::Unsupported("uzun dönem referans"));
            }
            for _ in 0..n {
                r.skip(log2_max_poc_lsb as usize + 1)?;
            }
        }
        r.skip(2)?; // sps_temporal_mvp_enabled, strong_intra_smoothing_enabled
        Ok(())
    }

    fn layout(codec: Codec, rbsp: &[u8], end: usize) -> Result<Layout, VuiError> {
        let header_bits = match codec {
            Codec::Avc => 8,
            Codec::Hevc => 16,
        };
        let mut r = Reader::new(rbsp, end);
        r.skip(header_bits)?;
        match codec {
            Codec::Avc => avc_to_vui(&mut r)?,
            Codec::Hevc => hevc_to_vui(&mut r)?,
        }
        let vui_flag = r.pos;
        let vui_present = r.bit()? == 1;
        let mut parsed = Parsed {
            vui_present,
            ..Parsed::default()
        };
        let aspect_start = r.pos;
        if !vui_present {
            return Ok(Layout {
                vui_flag,
                vui_present,
                aspect_start,
                aspect_end: aspect_start,
                overscan_end: aspect_start,
                signal_end: aspect_start,
                parsed,
            });
        }
        if r.bit()? == 1 {
            let idc = r.u(8)? as u8;
            parsed.aspect_idc = Some(idc);
            if idc == 255 {
                parsed.sar = Some((r.u(16)? as u16, r.u(16)? as u16));
            } else {
                parsed.sar = sar_from_idc(idc);
            }
        }
        let aspect_end = r.pos;
        if r.bit()? == 1 {
            r.skip(1)?;
        }
        let overscan_end = r.pos;
        if r.bit()? == 1 {
            parsed.video_format = Some(r.u(3)? as u8);
            let full = r.bit()? == 1;
            parsed.full_range = Some(full);
            if r.bit()? == 1 {
                parsed.colour = Some(Colour {
                    primaries: r.u(8)? as u8,
                    transfer: r.u(8)? as u8,
                    matrix: r.u(8)? as u8,
                    full_range: full,
                });
            }
        }
        let signal_end = r.pos;
        Ok(Layout {
            vui_flag,
            vui_present,
            aspect_start,
            aspect_end,
            overscan_end,
            signal_end,
            parsed,
        })
    }

    /// Tablo 2-1 (H.264 Tablo E-1 / HEVC Tablo E.1) SAR değerleri.
    const SAR_TABLE: [(u16, u16); 17] = [
        (0, 0),
        (1, 1),
        (12, 11),
        (10, 11),
        (16, 11),
        (40, 33),
        (24, 11),
        (20, 11),
        (32, 11),
        (80, 33),
        (18, 11),
        (15, 11),
        (64, 33),
        (160, 99),
        (4, 3),
        (3, 2),
        (2, 1),
    ];

    fn sar_from_idc(idc: u8) -> Option<(u16, u16)> {
        SAR_TABLE.get(idc as usize).copied().filter(|s| s.0 != 0)
    }

    fn nal_type(codec: Codec, nal: &[u8]) -> Option<u8> {
        let b = *nal.first()?;
        Some(match codec {
            Codec::Avc => b & 0x1f,
            Codec::Hevc => (b >> 1) & 0x3f,
        })
    }

    fn is_sps(codec: Codec, nal: &[u8]) -> bool {
        match codec {
            Codec::Avc => nal_type(codec, nal) == Some(7),
            Codec::Hevc => nal_type(codec, nal) == Some(33) && nal.len() > 2,
        }
    }

    /// Başlık baytları dahil SPS NAL'inin VUI değerlerini okur.
    pub fn parse_sps(codec: Codec, nal: &[u8]) -> Result<Parsed, VuiError> {
        let rbsp = unescape(nal);
        let end = stop_bit(&rbsp)?;
        Ok(layout(codec, &rbsp, end)?.parsed)
    }

    /// Annex B verisindeki ilk SPS'in VUI değerleri; SPS yoksa `None`.
    pub fn probe_annexb(codec: Codec, data: &[u8]) -> Result<Option<Parsed>, VuiError> {
        let mut i = 0;
        while i + 3 <= data.len() {
            let payload = if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                i + 3
            } else if i + 4 <= data.len()
                && data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                i + 4
            } else {
                i += 1;
                continue;
            };
            // NAL sonu: bir sonraki başlangıç kodu (00 00 01) veya veri sonu.
            let mut end = payload;
            while end + 3 <= data.len()
                && !(data[end] == 0
                    && data[end + 1] == 0
                    && (data[end + 2] == 1
                        || (data[end + 2] == 0 && end + 3 < data.len() && data[end + 3] == 1)))
            {
                end += 1;
            }
            if end + 3 > data.len() {
                end = data.len();
            }
            let nal = &data[payload..end];
            if is_sps(codec, nal) {
                return parse_sps(codec, nal).map(Some);
            }
            i = end;
        }
        Ok(None)
    }

    /// Başlık baytları dahil tek bir SPS NAL'ini yeniden yazar. Değişiklik gerekmiyorsa `None`.
    pub fn rewrite_sps(
        codec: Codec,
        nal: &[u8],
        target: Target,
    ) -> Result<Option<Vec<u8>>, VuiError> {
        if target.sar.is_none() && target.colour.is_none() {
            return Ok(None);
        }
        let rbsp = unescape(nal);
        let end = stop_bit(&rbsp)?;
        let l = layout(codec, &rbsp, end)?;
        let sar_ok = target
            .sar
            .is_none_or(|s| l.parsed.sar == Some(s) || (s == (1, 1) && l.parsed.sar.is_none()));
        let colour_ok = target.colour.is_none_or(|c| l.parsed.colour == Some(c));
        if sar_ok && colour_ok {
            return Ok(None);
        }
        let mut w = Writer::new();
        w.copy(&rbsp, 0, l.vui_flag);
        w.put(1, 1); // vui_parameters_present_flag
        // aspect_ratio_info
        match target.sar {
            Some(sar) if !sar_ok => {
                w.put(1, 1);
                match SAR_TABLE.iter().position(|&t| t == sar && t.0 != 0) {
                    Some(idc) => w.put(idc as u32, 8),
                    None => {
                        w.put(255, 8);
                        w.put(sar.0 as u32, 16);
                        w.put(sar.1 as u32, 16);
                    }
                }
            }
            _ if l.vui_present => w.copy(&rbsp, l.aspect_start, l.aspect_end),
            _ => w.put(0, 1),
        }
        // overscan_info
        if l.vui_present {
            w.copy(&rbsp, l.aspect_end, l.overscan_end);
        } else {
            w.put(0, 1);
        }
        // video_signal_type
        match target.colour {
            Some(c) if !colour_ok => {
                w.put(1, 1);
                w.put(l.parsed.video_format.unwrap_or(5) as u32, 3);
                w.put(c.full_range as u32, 1);
                w.put(1, 1);
                w.put(c.primaries as u32, 8);
                w.put(c.transfer as u32, 8);
                w.put(c.matrix as u32, 8);
            }
            _ if l.vui_present => w.copy(&rbsp, l.overscan_end, l.signal_end),
            _ => w.put(0, 1),
        }
        if l.vui_present {
            // VUI'nin kalanı ve SPS'in VUI sonrası (HEVC uzantıları) aynen.
            w.copy(&rbsp, l.signal_end, end);
        } else {
            match codec {
                // chroma_loc, timing, nal_hrd, vcl_hrd, pic_struct, bitstream_restriction
                Codec::Avc => w.put(0, 6),
                // chroma_loc, neutral_chroma, field_seq, frame_field_info, default_display_window,
                // timing, bitstream_restriction
                Codec::Hevc => w.put(0, 7),
            }
            w.copy(&rbsp, l.aspect_start, end); // VUI'den sonraki SPS bitleri (HEVC uzantı bayrakları)
        }
        Ok(Some(escape(&w.finish())))
    }

    /// Annex B akışındaki (başlangıç kodlu) bütün SPS NAL'lerini yeniden yazar. Değişiklik yoksa `None`.
    pub fn rewrite_annexb(
        codec: Codec,
        data: &[u8],
        target: Target,
    ) -> Result<Option<Vec<u8>>, VuiError> {
        // Başlangıç kodu konumları: (kod başı, yük başı)
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 <= data.len() {
            if data[i] == 0 && data[i + 1] == 0 {
                if data[i + 2] == 1 {
                    starts.push((i, i + 3));
                    i += 3;
                    continue;
                }
                if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                    starts.push((i, i + 4));
                    i += 4;
                    continue;
                }
            }
            i += 1;
        }
        if starts.is_empty() {
            return Ok(None);
        }
        let mut out: Option<Vec<u8>> = None;
        let mut copied = 0;
        for (k, &(code, payload)) in starts.iter().enumerate() {
            let next = starts.get(k + 1).map(|s| s.0).unwrap_or(data.len());
            // NAL'in sonundaki sıfırlar bir sonraki başlangıç kodunun parçası sayılmaz; olduğu gibi kalır.
            let mut nal_end = next;
            while nal_end > payload && data[nal_end - 1] == 0 {
                nal_end -= 1;
            }
            let nal = &data[payload..nal_end];
            if !is_sps(codec, nal) {
                continue;
            }
            if let Some(new_nal) = rewrite_sps(codec, nal, target)? {
                let buf = out.get_or_insert_with(|| Vec::with_capacity(data.len() + 64));
                buf.extend_from_slice(&data[copied..payload]);
                buf.extend_from_slice(&new_nal);
                copied = nal_end;
            }
            let _ = code;
        }
        Ok(out.map(|mut buf| {
            buf.extend_from_slice(&data[copied..]);
            buf
        }))
    }
}

// ============================================================================
// 9b. half — FP32 ↔ FP16 (IEEE 754 binary16) dönüşümü
// ============================================================================

/// Ara özellik halkası FP16 tutulur (15 Kasım mimari kararı): ONNX Runtime giriş/çıkışı FP32, çözücüler
/// INT8 QDQ olduğu için atlama tensörleri çözücüye girerken zaten 8 bite yuvarlanır; FP16 saklama halkayı
/// yarıya indirir (14 Kasım Z Flip8: 264 MB). Dönüşüm burada, Kotlin'de değil:
///
/// * aarch64: `FCVTN`/`FCVTN2` ve `FCVTL`/`FCVTL2` — ARMv8.0 temel Advanced SIMD (FP16 aritmetik eklentisi
///   ARMv8.2 GEREKMEZ; Cortex-A53'te de vardır). 8 değer/yineleme, kuyruk skaler.
/// * x86_64: F16C (`VCVTPS2PH` en yakına yuvarlama / `VCVTPH2PS`), çalışma zamanında algılanır.
/// * Diğer / kuyruk: skaler başvuru — en yakına yuvarlama (eşitlikte çift), alt-normaller, ±sonsuz taşma,
///   NaN yükü korunur ve sessiz biti kurulur. Donanım yollarıyla bit düzeyinde aynıdır (testler denetler).
pub mod half {
    /// f32 → f16 bitleri (en yakına yuvarla, eşitlikte çift).
    pub fn f32_to_f16(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x007f_ffff;
        if exp == 0xff {
            // ±sonsuz veya NaN (sessiz bit kurulur, yükün üst bitleri korunur).
            return sign
                | 0x7c00
                | if man != 0 {
                    0x0200 | (man >> 13) as u16
                } else {
                    0
                };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = man | 0x0080_0000;
            let shift = (14 - e) as u32;
            let q = m >> shift;
            let rem = m & ((1 << shift) - 1);
            let mid = 1 << (shift - 1);
            return sign | (q + (rem > mid || (rem == mid && q & 1 == 1)) as u32) as u16;
        }
        let q = ((e as u32) << 10) | (man >> 13);
        let rem = man & 0x1fff;
        // Mantis taşması üsse geçer (en büyük sonlu değerden sonsuza doğru yuvarlama dahil).
        sign | (q + (rem > 0x1000 || (rem == 0x1000 && q & 1 == 1)) as u32) as u16
    }

    /// f16 bitleri → f32 (kayıpsız).
    pub fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = ((h >> 10) & 0x1f) as u32;
        let man = (h & 0x03ff) as u32;
        let bits = match (exp, man) {
            (0, 0) => sign,
            (0, _) => {
                let mut e = 127 - 14;
                let mut m = man;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x03ff) << 13)
            }
            (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
            _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
        };
        f32::from_bits(bits)
    }

    /// Hangi yol kullanılıyor (tanılama).
    pub fn backend() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "aarch64-fcvtn"
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("f16c") {
                "x86_64-f16c"
            } else {
                "scalar"
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "scalar"
        }
    }

    /// `dst[i] = f16(src[i])`; uzunluklar eşit olmalı.
    pub fn narrow(src: &[f32], dst: &mut [u16]) {
        assert_eq!(src.len(), dst.len());
        let done = simd_narrow(src, dst);
        for (d, &x) in dst[done..].iter_mut().zip(&src[done..]) {
            *d = f32_to_f16(x);
        }
    }

    /// `dst[i] = f32(src[i])`; uzunluklar eşit olmalı.
    pub fn widen(src: &[u16], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len());
        let done = simd_widen(src, dst);
        for (d, &h) in dst[done..].iter_mut().zip(&src[done..]) {
            *d = f16_to_f32(h);
        }
    }

    /// Skaler başvuru (testler donanım yoluyla karşılaştırır).
    pub fn narrow_scalar(src: &[f32], dst: &mut [u16]) {
        for (d, &x) in dst.iter_mut().zip(src) {
            *d = f32_to_f16(x);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_narrow(src: &[f32], dst: &mut [u16]) -> usize {
        let n = src.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            // SAFETY: i + 8 <= n <= len; ld1/st1 hizasız erişime izin verir. Yalnızca v0–v2 bozulur.
            unsafe {
                core::arch::asm!(
                    "ld1 {{v0.4s, v1.4s}}, [{s}]",
                    "fcvtn v2.4h, v0.4s",
                    "fcvtn2 v2.8h, v1.4s",
                    "st1 {{v2.8h}}, [{d}]",
                    s = in(reg) src.as_ptr().add(i),
                    d = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack, preserves_flags),
                );
            }
        }
        n
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_widen(src: &[u16], dst: &mut [f32]) -> usize {
        let n = src.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            // SAFETY: i + 8 <= n <= len; yalnızca v0–v2 bozulur.
            unsafe {
                core::arch::asm!(
                    "ld1 {{v0.8h}}, [{s}]",
                    "fcvtl v1.4s, v0.4h",
                    "fcvtl2 v2.4s, v0.8h",
                    "st1 {{v1.4s, v2.4s}}, [{d}]",
                    s = in(reg) src.as_ptr().add(i),
                    d = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack, preserves_flags),
                );
            }
        }
        n
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_narrow(src: &[f32], dst: &mut [u16]) -> usize {
        if !std::is_x86_feature_detected!("f16c") {
            return 0;
        }
        // SAFETY: F16C çalışma zamanında doğrulandı.
        unsafe { narrow_f16c(src, dst) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c,avx")]
    unsafe fn narrow_f16c(src: &[f32], dst: &mut [u16]) -> usize {
        use core::arch::x86_64::*;
        let n = src.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            // SAFETY: i + 8 <= n; hizasız yükleme/saklama.
            unsafe {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(v);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
            }
        }
        n
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_widen(src: &[u16], dst: &mut [f32]) -> usize {
        if !std::is_x86_feature_detected!("f16c") {
            return 0;
        }
        // SAFETY: F16C çalışma zamanında doğrulandı.
        unsafe { widen_f16c(src, dst) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c,avx")]
    unsafe fn widen_f16c(src: &[u16], dst: &mut [f32]) -> usize {
        use core::arch::x86_64::*;
        let n = src.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            // SAFETY: i + 8 <= n; hizasız yükleme/saklama.
            unsafe {
                let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
            }
        }
        n
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn simd_narrow(_: &[f32], _: &mut [u16]) -> usize {
        0
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn simd_widen(_: &[u16], _: &mut [f32]) -> usize {
        0
    }
}

// ============================================================================
// 10. JNI köprüsü (Kotlin: com.video.cleaner.Core)
// ============================================================================

/// Kotlin köprüsü. Harici JNI crate'i kullanılmaz: JNIEnv işlev tablosundan
/// yalnızca `GetDirectBufferAddress` (230) ve `GetDirectBufferCapacity` (231)
/// çağrılır. Tüm veri alışverişi doğrudan (direct) `ByteBuffer`'lar üzerindendir;
/// hatalar istisna yerine negatif [`CoreError`] kodu olarak döner.
pub mod jni {
    use super::*;

    /// `JNIEnv` = `const struct JNINativeInterface *` (işlev işaretçileri tablosu).
    pub type JniEnv = *const *const c_void;
    /// Herhangi bir Java nesne başvurusu.
    pub type JObject = *mut c_void;

    const JNI_GET_DIRECT_BUFFER_ADDRESS: usize = 230;
    const JNI_GET_DIRECT_BUFFER_CAPACITY: usize = 231;
    const JNI_NEW_DIRECT_BYTE_BUFFER: usize = 229;
    const HANDLE_MAGIC: u64 = 0x5643_4C45_414E_3230; // "VCLEAN20"

    /// `nativePoll` adım kodları.
    pub const STEP_NEED_FRAMES: i32 = 0;
    /// Pencere hazır: çıkarımı çalıştırıp `nativeCompleteWindow` çağırın.
    pub const STEP_WINDOW: i32 = 1;
    /// Çıkış kareleri hazır.
    pub const STEP_OUTPUT: i32 = 2;
    /// Akış tamamlandı.
    pub const STEP_FINISHED: i32 = 3;
    /// `nativePoll` bilgi tamponunun en küçük boyutu.
    pub const POLL_INFO_BYTES: usize = 64;
    /// `nativePopOutput*` bilgi tamponunun en küçük boyutu.
    pub const OUTPUT_META_BYTES: usize = 32;
    /// `nativeStats` tamponunun en küçük boyutu.
    pub const STATS_BYTES: usize = 112;

    struct Session {
        pipeline: Pipeline,
        scratch: Vec<u8>,
    }

    struct Handle {
        magic: u64,
        session: Mutex<Session>,
    }

    /// Doğrudan ByteBuffer'ın adresi ve kapasitesi.
    ///
    /// # Safety
    /// `env`, çağıran iş parçacığı için geçerli bir `JNIEnv*` olmalıdır.
    unsafe fn direct_buffer(env: *mut JniEnv, buf: JObject) -> Result<(*mut u8, usize), CoreError> {
        if env.is_null() || buf.is_null() {
            return Err(CoreError::BadArgument);
        }
        type GetAddress = unsafe extern "system" fn(*mut JniEnv, JObject) -> *mut c_void;
        type GetCapacity = unsafe extern "system" fn(*mut JniEnv, JObject) -> i64;
        // SAFETY: JNI belirtimi gereği *env, JNINativeInterface işlev tablosunu
        // gösterir; 230 ve 231 numaralı girişler bu iki işlevdir (JNI 1.4+).
        unsafe {
            let table: *const *const c_void = *env;
            if table.is_null() {
                return Err(CoreError::BadArgument);
            }
            let get_address: GetAddress =
                std::mem::transmute(*table.add(JNI_GET_DIRECT_BUFFER_ADDRESS));
            let get_capacity: GetCapacity =
                std::mem::transmute(*table.add(JNI_GET_DIRECT_BUFFER_CAPACITY));
            let address = get_address(env, buf);
            let capacity = get_capacity(env, buf);
            if address.is_null() || capacity <= 0 {
                // Doğrudan olmayan (heap) ByteBuffer verildi.
                return Err(CoreError::BadArgument);
            }
            Ok((address as *mut u8, capacity as usize))
        }
    }

    /// # Safety
    /// `env` geçerli bir `JNIEnv*` olmalıdır.
    unsafe fn plane(
        env: *mut JniEnv,
        buf: JObject,
        offset: i32,
        row_stride: i32,
        pixel_stride: i32,
    ) -> Result<RawPlane, CoreError> {
        // SAFETY: çağıranın sözleşmesi.
        let (ptr, cap) = unsafe { direct_buffer(env, buf)? };
        if offset < 0 || row_stride <= 0 || pixel_stride <= 0 || offset as usize >= cap {
            return Err(CoreError::BadArgument);
        }
        // SAFETY: offset < cap olduğundan sonuç tampon içindedir.
        let ptr = unsafe { ptr.add(offset as usize) };
        Ok(RawPlane {
            ptr,
            len: cap - offset as usize,
            row_stride: row_stride as usize,
            pixel_stride: pixel_stride as usize,
        })
    }

    /// Doğrudan tamponu yazılabilir dilim olarak verir.
    ///
    /// # Safety
    /// `env` geçerli olmalı; tampon çağrı boyunca yaşamalı ve başka yerden erişilmemelidir.
    unsafe fn buffer_mut<'a>(
        env: *mut JniEnv,
        buf: JObject,
        min_len: usize,
    ) -> Result<&'a mut [u8], CoreError> {
        // SAFETY: çağıranın sözleşmesi.
        let (ptr, cap) = unsafe { direct_buffer(env, buf)? };
        if cap < min_len {
            return Err(CoreError::BufferTooSmall);
        }
        // SAFETY: JVM, doğrudan tamponun [ptr, ptr+cap) aralığının geçerli olduğunu garanti eder.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, cap) })
    }

    fn put_i32(buf: &mut [u8], at: usize, v: i32) {
        buf[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    }

    fn put_i64(buf: &mut [u8], at: usize, v: i64) {
        buf[at..at + 8].copy_from_slice(&v.to_ne_bytes());
    }

    fn call<F>(handle: i64, f: F) -> i64
    where
        F: FnOnce(&mut Session) -> Result<i64, CoreError>,
    {
        let ptr = handle as usize as *const Handle;
        if ptr.is_null() || !ptr.is_aligned() {
            return CoreError::InvalidHandle.code() as i64;
        }
        // SAFETY: Tutamak nativeCreate tarafından Box::into_raw ile üretilir ve
        // nativeDestroy çağrılana kadar geçerlidir (Kotlin sarmalayıcının sözleşmesi).
        // Sihirli sayı, yanlış veya serbest bırakılmış tutamakların çoğunu yakalar.
        let handle = unsafe { &*ptr };
        if handle.magic != HANDLE_MAGIC {
            return CoreError::InvalidHandle.code() as i64;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut session = lock(&handle.session);
            f(&mut session)
        }));
        match result {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => error.code() as i64,
            Err(_) => CoreError::Internal.code() as i64,
        }
    }

    /// `external fun nativeAbiVersion(): Int`
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeAbiVersion(
        _env: *mut JniEnv,
        _class: JObject,
    ) -> i32 {
        ABI_VERSION
    }

    /// `external fun nativeKernelBackend(): Int` — 0 taşınabilir, 1 AVX2, 2 NEON.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeKernelBackend(
        _env: *mut JniEnv,
        _class: JObject,
    ) -> i32 {
        kernels::backend_id()
    }

    /// `external fun nativePageSize(): Int` — çalışma anındaki sayfa boyutu.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativePageSize(
        _env: *mut JniEnv,
        _class: JObject,
    ) -> i32 {
        page_size() as i32
    }

    /// `external fun nativeCreate(width: Int, height: Int): Long` — tutamak veya negatif hata kodu.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeCreate(
        _env: *mut JniEnv,
        _class: JObject,
        width: i32,
        height: i32,
    ) -> i64 {
        let created = std::panic::catch_unwind(|| -> Result<i64, CoreError> {
            if width <= 0 || height <= 0 {
                return Err(CoreError::InvalidGeometry);
            }
            let geo = FrameGeometry::new(width as usize, height as usize)?;
            let session = Session {
                pipeline: Pipeline::new(geo)?,
                scratch: vec![0u8; geo.frame_bytes()],
            };
            let handle = Box::new(Handle {
                magic: HANDLE_MAGIC,
                session: Mutex::new(session),
            });
            Ok(Box::into_raw(handle) as usize as i64)
        });
        match created {
            Ok(Ok(handle)) => handle,
            Ok(Err(error)) => error.code() as i64,
            Err(_) => CoreError::Internal.code() as i64,
        }
    }

    /// `external fun nativeDestroy(handle: Long)`
    ///
    /// # Safety
    /// `handle`, `nativeCreate`'ten dönen ve daha önce yok edilmemiş bir değer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeDestroy(
        _env: *mut JniEnv,
        _class: JObject,
        handle: i64,
    ) {
        let ptr = handle as usize as *mut Handle;
        if ptr.is_null() || !ptr.is_aligned() {
            return;
        }
        // SAFETY: çağıranın sözleşmesi; sihirli sayı çift serbest bırakmayı çoğu durumda yakalar.
        unsafe {
            if (*ptr).magic != HANDLE_MAGIC {
                return;
            }
            (*ptr).magic = 0;
            drop(Box::from_raw(ptr));
        }
    }

    /// `external fun nativeSetOption(handle: Long, key: Int, value: Int): Int`
    /// — 1: otomatik sahne geçişi, 2: bütünlük denetimi.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeSetOption(
        _env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        key: i32,
        value: i32,
    ) -> i32 {
        call(handle, |s| {
            match key {
                1 => s.pipeline.set_auto_scene_cut(value != 0),
                2 => s.pipeline.set_integrity_check(value != 0),
                3 => s.pipeline.set_feather(value.max(0) as u32)?,
                4 => s.pipeline.set_grain(value != 0),
                _ => return Err(CoreError::BadArgument),
            }
            Ok(0)
        }) as i32
    }

    /// `external fun nativePoll(handle: Long, info: ByteBuffer): Int` — adım kodu.
    ///
    /// `info` (yerel bayt sırası, ≥ 64 bayt): `i32 adım, i32 sayı, i64 planNo,
    /// i32 mod(0=A,1=B), i32 kayma, i32 geçerliSayısı, i32 emitBaş, i32 emitSon,
    /// i32 önbellekBaş, i32 önbellekSon, i32 bayraklar, i64 tabanSıra, i32 geçerlilikMaskesi`.
    ///
    /// # Safety
    /// `env` geçerli bir `JNIEnv*`, `info` doğrudan bir ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePoll(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        info: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let info = match unsafe { buffer_mut(env, info, POLL_INFO_BYTES) } {
            Ok(buf) => buf,
            Err(error) => return error.code(),
        };
        call(handle, |s| {
            info[..POLL_INFO_BYTES].fill(0);
            let code = match s.pipeline.poll() {
                Step::NeedFrames(n) => {
                    put_i32(info, 4, n as i32);
                    STEP_NEED_FRAMES
                }
                Step::Output(n) => {
                    put_i32(info, 4, n as i32);
                    STEP_OUTPUT
                }
                Step::Finished => STEP_FINISHED,
                Step::Window(plan) => {
                    put_i64(info, 8, plan.id as i64);
                    put_i32(info, 16, if plan.mode == Mode::A { 0 } else { 1 });
                    put_i32(info, 20, plan.slide as i32);
                    put_i32(info, 24, plan.valid_count() as i32);
                    put_i32(info, 28, plan.emit.0 as i32);
                    put_i32(info, 32, plan.emit.1 as i32);
                    put_i32(info, 36, plan.cache.0 as i32);
                    put_i32(info, 40, plan.cache.1 as i32);
                    put_i32(info, 44, plan.flags as i32);
                    put_i64(info, 48, plan.base_seq);
                    put_i32(info, 56, plan.valid as i32);
                    STEP_WINDOW
                }
            };
            put_i32(info, 0, code);
            Ok(code as i64)
        }) as i32
    }

    /// `external fun nativePushYuv420(...)`: MediaCodec `Image` düzlemlerinden kare iter.
    /// Örnek başına birden çok bayt taşıyan biçimlerde (P010) piksel adımları ve ofsetler üst
    /// baytı gösterecek şekilde verilir; çekirdek her zaman 8 bit NV12 saklar.
    /// Karenin sıra numarasını veya negatif hata kodunu döndürür.
    ///
    /// # Safety
    /// `env` geçerli olmalı; üç tampon da doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePushYuv420(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        y_buf: JObject,
        y_offset: i32,
        y_row_stride: i32,
        y_pixel_stride: i32,
        u_buf: JObject,
        u_offset: i32,
        v_buf: JObject,
        v_offset: i32,
        uv_row_stride: i32,
        uv_pixel_stride: i32,
        pts_us: i64,
        flags: i32,
    ) -> i64 {
        // SAFETY: çağıranın sözleşmesi.
        let planes = unsafe {
            (|| {
                Ok::<_, CoreError>((
                    plane(env, y_buf, y_offset, y_row_stride, y_pixel_stride)?,
                    plane(env, u_buf, u_offset, uv_row_stride, uv_pixel_stride)?,
                    plane(env, v_buf, v_offset, uv_row_stride, uv_pixel_stride)?,
                ))
            })()
        };
        let (y, u, v) = match planes {
            Ok(p) => p,
            Err(error) => return error.code() as i64,
        };
        call(handle, |s| {
            let geo = s.pipeline.geometry();
            let meta = s.pipeline.push_with(|dst| {
                // SAFETY: düzlemler JVM'in doğrudan tamponlarıdır ve çağrı boyunca geçerlidir.
                unsafe { pack_nv12(geo, y, u, v, dst)? };
                Ok(Some((pts_us, flags as u32)))
            })?;
            meta.map(|m| m.seq as i64).ok_or(CoreError::Internal)
        })
    }

    /// `external fun nativePushHardwareBuffer(handle: Long, buffer: HardwareBuffer, cropLeft: Int, cropTop: Int,
    /// ptsUs: Long, flags: Int): Long` — ImageReader görüntüsünün donanım tamponunu Java'da `Image.getPlanes()`
    /// çağırmadan, `AHardwareBuffer_lockPlanes` ile kilitleyip iter.
    ///
    /// Neden: bazı cihazlarda (Galaxy Z Flip8, Android 17, Qualcomm çözücü) Surface çıkışı CPU'ya açılamayan
    /// (sıkıştırılmış) tampon olur ve `getPlanes()` istisna yerine JNI abort ile süreci öldürür. Burada kilit
    /// başarısızsa veya düzlem adresi boşsa çökmeden `Unsupported` (-14) döner; çağıran bayt tamponu yoluna düşer.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `buffer` bir `android.hardware.HardwareBuffer` olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePushHardwareBuffer(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        buffer: JObject,
        crop_left: i32,
        crop_top: i32,
        pts_us: i64,
        flags: i32,
    ) -> i64 {
        #[cfg(target_os = "android")]
        {
            use crate::hwbuf::*;
            if buffer.is_null() || crop_left < 0 || crop_top < 0 {
                return CoreError::BadArgument.code() as i64;
            }
            // SAFETY: env ve buffer JVM'den gelir; AHardwareBuffer çağrı boyunca HardwareBuffer nesnesiyle yaşar.
            unsafe {
                let ahb = AHardwareBuffer_fromHardwareBuffer(env as *mut c_void, buffer);
                if ahb.is_null() {
                    return CoreError::BadArgument.code() as i64;
                }
                let mut desc = Desc::default();
                AHardwareBuffer_describe(ahb, &mut desc);
                let mut planes = Planes {
                    plane_count: 0,
                    planes: [Plane {
                        data: std::ptr::null_mut(),
                        pixel_stride: 0,
                        row_stride: 0,
                    }; 4],
                };
                if AHardwareBuffer_lockPlanes(
                    ahb,
                    USAGE_CPU_READ_OFTEN,
                    -1,
                    std::ptr::null(),
                    &mut planes,
                ) != 0
                {
                    return CoreError::Unsupported.code() as i64;
                }
                let usable = planes.plane_count >= 3
                    && planes.planes[..3]
                        .iter()
                        .all(|p| !p.data.is_null() && p.row_stride > 0 && p.pixel_stride > 0);
                let result = if !usable {
                    Err(CoreError::Unsupported)
                } else {
                    let rows = desc.height as usize;
                    let (left, top) = (crop_left as usize, crop_top as usize);
                    // Kırpma ofseti; 4:2:0 kroma yarım çözünürlüktedir.
                    let mk =
                        |p: Plane, plane_rows: usize, x: usize, y: usize| -> Option<RawPlane> {
                            let (rs, ps) = (p.row_stride as usize, p.pixel_stride as usize);
                            let offset = y * rs + x * ps;
                            let total = rs * plane_rows;
                            (offset < total).then(|| RawPlane {
                                ptr: (p.data as *mut u8).add(offset),
                                len: total - offset,
                                row_stride: rs,
                                pixel_stride: ps,
                            })
                        };
                    match (
                        mk(planes.planes[0], rows, left, top),
                        mk(planes.planes[1], rows / 2, left / 2, top / 2),
                        mk(planes.planes[2], rows / 2, left / 2, top / 2),
                    ) {
                        (Some(y), Some(u), Some(v)) => Ok((y, u, v)),
                        _ => Err(CoreError::BadArgument),
                    }
                };
                let code = match result {
                    Err(error) => error.code() as i64,
                    Ok((y, u, v)) => call(handle, |s| {
                        let geo = s.pipeline.geometry();
                        let meta = s.pipeline.push_with(|dst| {
                            // SAFETY: düzlemler kilitli donanım tamponundadır ve kilit açılana dek geçerlidir.
                            pack_nv12(geo, y, u, v, dst)?;
                            Ok(Some((pts_us, flags as u32)))
                        })?;
                        meta.map(|m| m.seq as i64).ok_or(CoreError::Internal)
                    }),
                };
                AHardwareBuffer_unlock(ahb, std::ptr::null_mut());
                code
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (env, handle, buffer, crop_left, crop_top, pts_us, flags);
            CoreError::BadState.code() as i64
        }
    }

    /// `external fun nativeAllocDirect(bytes: Long): ByteBuffer?`
    ///
    /// Java yığınına SAYILMAYAN, 64 bayt hizalı, sıfırlanmış yerel bellek üzerinde doğrudan ByteBuffer.
    /// Android'de `ByteBuffer.allocateDirect` Java yığınından ayrılır (taşınmaz dizi) ve 192–256 MB yığın
    /// sınırına takılır: 30 karelik ara özellik halkası (yüzlerce MB) orada tutulamaz. Bellek
    /// [`Java_com_video_cleaner_Core_nativeFreeDirect`] ile geri verilmelidir.
    ///
    /// # Safety
    /// `env` geçerli olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeAllocDirect(
        env: *mut JniEnv,
        _class: JObject,
        bytes: i64,
    ) -> JObject {
        if env.is_null() || bytes <= 0 {
            return std::ptr::null_mut();
        }
        let Some(ptr) = native_alloc(bytes as usize) else {
            return std::ptr::null_mut();
        };
        type NewDirect = unsafe extern "system" fn(*mut JniEnv, *mut c_void, i64) -> JObject;
        // SAFETY: JNI belirtimi: işlev tablosunun 229. girişi NewDirectByteBuffer'dır (JNI 1.4+).
        unsafe {
            let table: *const *const c_void = *env;
            let new_direct: NewDirect = std::mem::transmute(*table.add(JNI_NEW_DIRECT_BYTE_BUFFER));
            let buffer = new_direct(env, ptr as *mut c_void, bytes);
            if buffer.is_null() {
                native_free(ptr, bytes as usize);
            }
            buffer
        }
    }

    /// `external fun nativeFreeDirect(buffer: ByteBuffer): Int` — [`Java_com_video_cleaner_Core_nativeAllocDirect`]
    /// ile ayrılan belleği geri verir (0 başarı). Başka kaynaklı bir tampon VERİLMEMELİDİR.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `buffer` nativeAllocDirect'ten gelmeli ve bir kez serbest bırakılmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeFreeDirect(
        env: *mut JniEnv,
        _class: JObject,
        buffer: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        match unsafe { direct_buffer(env, buffer) } {
            Ok((ptr, capacity)) => {
                // SAFETY: ptr/capacity nativeAllocDirect'in ayırdığı blok.
                unsafe { native_free(ptr, capacity) };
                0
            }
            Err(e) => e.code(),
        }
    }

    /// Doğrudan tampondan `[offset, offset+count)` öğelik hizalı dilim (`T` = f32 veya u16).
    ///
    /// # Safety
    /// `env` geçerli olmalı; tampon çağrı boyunca yaşamalıdır.
    unsafe fn typed<'a, T>(
        env: *mut JniEnv,
        buf: JObject,
        offset: i32,
        count: i32,
    ) -> Result<&'a mut [T], CoreError> {
        // SAFETY: çağıranın sözleşmesi.
        let (ptr, cap) = unsafe { direct_buffer(env, buf)? };
        if offset < 0 || count < 0 {
            return Err(CoreError::BadArgument);
        }
        let size = std::mem::size_of::<T>();
        let end = (offset as usize + count as usize)
            .checked_mul(size)
            .ok_or(CoreError::BadArgument)?;
        if end > cap {
            return Err(CoreError::BufferTooSmall);
        }
        // SAFETY: aralık tamponun içinde.
        let start = unsafe { ptr.add(offset as usize * size) };
        if !(start as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(CoreError::BadArgument);
        }
        // SAFETY: hizalı, tampon içinde; çağıran aynı bölgeye eşzamanlı yazmaz.
        Ok(unsafe { std::slice::from_raw_parts_mut(start as *mut T, count as usize) })
    }

    /// `external fun nativeF32ToF16(src: ByteBuffer, srcOffset: Int, dst: ByteBuffer, dstOffset: Int, count: Int): Int`
    ///
    /// Ara özellik halkasına yazma: ORT'nin FP32 çıktısı → halkada FP16 ([`half::narrow`]). Ofset ve sayı
    /// ÖĞE cinsinden; dönüş `count` veya negatif hata kodu. Kaynak ve hedef çakışmamalıdır.
    ///
    /// # Safety
    /// `env` geçerli olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeF32ToF16(
        env: *mut JniEnv,
        _class: JObject,
        src: JObject,
        src_offset: i32,
        dst: JObject,
        dst_offset: i32,
        count: i32,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let run = || -> Result<i32, CoreError> {
            let s: &[f32] = unsafe { typed(env, src, src_offset, count)? };
            let d: &mut [u16] = unsafe { typed(env, dst, dst_offset, count)? };
            half::narrow(s, d);
            Ok(count)
        };
        run().unwrap_or_else(|e| e.code())
    }

    /// `external fun nativeF16ToF32(src: ByteBuffer, srcOffset: Int, dst: ByteBuffer, dstOffset: Int, count: Int): Int`
    ///
    /// Halkadan okuma: FP16 → ORT girişi için FP32 ([`half::widen`], kayıpsız).
    ///
    /// # Safety
    /// `env` geçerli olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeF16ToF32(
        env: *mut JniEnv,
        _class: JObject,
        src: JObject,
        src_offset: i32,
        dst: JObject,
        dst_offset: i32,
        count: i32,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let run = || -> Result<i32, CoreError> {
            let s: &[u16] = unsafe { typed(env, src, src_offset, count)? };
            let d: &mut [f32] = unsafe { typed(env, dst, dst_offset, count)? };
            half::widen(s, d);
            Ok(count)
        };
        run().unwrap_or_else(|e| e.code())
    }

    const NATIVE_ALIGN: usize = 64;

    /// Sıfırlanmış, 64 bayt hizalı blok; boyut taşarsa veya bellek yoksa `None`.
    pub(crate) fn native_alloc(bytes: usize) -> Option<*mut u8> {
        let layout = std::alloc::Layout::from_size_align(bytes, NATIVE_ALIGN).ok()?;
        // SAFETY: boyut > 0 (çağıran denetler).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        (!ptr.is_null()).then_some(ptr)
    }

    /// # Safety
    /// `ptr` aynı `bytes` ile [`native_alloc`]'tan gelmiş olmalıdır.
    pub(crate) unsafe fn native_free(ptr: *mut u8, bytes: usize) {
        if let Ok(layout) = std::alloc::Layout::from_size_align(bytes, NATIVE_ALIGN) {
            // SAFETY: çağıranın sözleşmesi.
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }

    /// `external fun nativeProbeVui(codec: Int, src: ByteBuffer, offset: Int, length: Int, out: ByteBuffer): Int`
    ///
    /// Kaynağın parametre kümelerindeki (csd-0, Annex B) ilk SPS'in VUI değerleri `out`'a (doğrudan tampon,
    /// yerel bayt sırasıyla 9 × i32) yazılır:
    /// [sarW, sarH, renkVar, primaries, transfer, matrix, aralıkVar, tamAralık, vuiVar]; SAR yoksa 0/0.
    /// Dönüş: 1 SPS bulundu, 0 SPS yok, negatif hata.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `src` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeProbeVui(
        env: *mut JniEnv,
        _class: JObject,
        codec: i32,
        src: JObject,
        offset: i32,
        length: i32,
        out: JObject,
    ) -> i32 {
        use crate::vui::{Codec, probe_annexb};
        let codec = match codec {
            0 => Codec::Avc,
            1 => Codec::Hevc,
            _ => return CoreError::BadArgument.code(),
        };
        // SAFETY: çağıranın sözleşmesi.
        let (src_ptr, src_cap) = match unsafe { direct_buffer(env, src) } {
            Ok(v) => v,
            Err(e) => return e.code(),
        };
        if offset < 0 || length <= 0 || offset as usize + length as usize > src_cap {
            return CoreError::BadArgument.code();
        }
        // SAFETY: aralık tampon içinde denetlendi.
        let data =
            unsafe { std::slice::from_raw_parts(src_ptr.add(offset as usize), length as usize) };
        let parsed = match probe_annexb(codec, data) {
            Ok(Some(p)) => p,
            Ok(None) => return 0,
            Err(_) => return CoreError::Unsupported.code(),
        };
        let (sw, sh) = parsed
            .sar
            .map(|(w, h)| (w as i32, h as i32))
            .unwrap_or((0, 0));
        let c = parsed.colour;
        let values = [
            sw,
            sh,
            c.is_some() as i32,
            c.map(|c| c.primaries as i32).unwrap_or(2),
            c.map(|c| c.transfer as i32).unwrap_or(2),
            c.map(|c| c.matrix as i32).unwrap_or(2),
            parsed.full_range.is_some() as i32,
            parsed.full_range.unwrap_or(false) as i32,
            parsed.vui_present as i32,
        ];
        // SAFETY: çağıranın sözleşmesi; `out` en az 36 baytlık doğrudan tampon.
        let dst = match unsafe { buffer_mut(env, out, values.len() * 4) } {
            Ok(b) => b,
            Err(e) => return e.code(),
        };
        for (k, v) in values.iter().enumerate() {
            put_i32(dst, k * 4, *v);
        }
        1
    }

    /// `external fun nativeRewriteVui(codec: Int, src: ByteBuffer, offset: Int, length: Int, dst: ByteBuffer,
    /// sarWidth: Int, sarHeight: Int, primaries: Int, transfer: Int, matrix: Int, fullRange: Int): Int`
    ///
    /// Annex B (başlangıç kodlu) H.264 (`codec` 0) veya HEVC (1) verisindeki SPS'lerin VUI'sini yeniden yazar
    /// ([`crate::vui`]). `sarWidth <= 0` → SAR'a dokunulmaz; `primaries < 0` → renge dokunulmaz.
    /// Dönüş: `dst`'ye yazılan yeni uzunluk (> 0), değişiklik yoksa 0, hata ise negatif kod.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `src` ve `dst` doğrudan ByteBuffer olmalı ve örtüşmemelidir.
    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeRewriteVui(
        env: *mut JniEnv,
        _class: JObject,
        codec: i32,
        src: JObject,
        offset: i32,
        length: i32,
        dst: JObject,
        sar_width: i32,
        sar_height: i32,
        primaries: i32,
        transfer: i32,
        matrix: i32,
        full_range: i32,
    ) -> i32 {
        use crate::vui::{Codec, Colour, Target, rewrite_annexb};
        let codec = match codec {
            0 => Codec::Avc,
            1 => Codec::Hevc,
            _ => return CoreError::BadArgument.code(),
        };
        // SAFETY: çağıranın sözleşmesi.
        let (src_ptr, src_cap) = match unsafe { direct_buffer(env, src) } {
            Ok(v) => v,
            Err(e) => return e.code(),
        };
        if offset < 0 || length <= 0 || offset as usize + length as usize > src_cap {
            return CoreError::BadArgument.code();
        }
        // SAFETY: aralık tampon içinde denetlendi; tampon çağrı boyunca yaşar.
        let data =
            unsafe { std::slice::from_raw_parts(src_ptr.add(offset as usize), length as usize) };
        let target = Target {
            sar: (sar_width > 0 && sar_height > 0 && sar_width <= 65535 && sar_height <= 65535)
                .then_some((sar_width as u16, sar_height as u16)),
            colour: (primaries >= 0).then_some(Colour {
                primaries: primaries.clamp(0, 255) as u8,
                transfer: transfer.clamp(0, 255) as u8,
                matrix: matrix.clamp(0, 255) as u8,
                full_range: full_range == 1,
            }),
        };
        match rewrite_annexb(codec, data, target) {
            Ok(None) => 0,
            Ok(Some(out)) => {
                // SAFETY: çağıranın sözleşmesi.
                let dst = match unsafe { buffer_mut(env, dst, out.len()) } {
                    Ok(b) => b,
                    Err(e) => return e.code(),
                };
                dst[..out.len()].copy_from_slice(&out);
                out.len() as i32
            }
            Err(_) => CoreError::Unsupported.code(),
        }
    }

    /// `external fun nativePushNv12(handle: Long, frame: ByteBuffer, ptsUs: Long, flags: Int): Long`
    ///
    /// # Safety
    /// `env` geçerli olmalı; `frame` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePushNv12(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        frame: JObject,
        pts_us: i64,
        flags: i32,
    ) -> i64 {
        // SAFETY: çağıranın sözleşmesi.
        let src = match unsafe { buffer_mut(env, frame, 1) } {
            Ok(buf) => buf,
            Err(error) => return error.code() as i64,
        };
        call(handle, |s| {
            s.pipeline
                .push_frame(src, pts_us, flags as u32)
                .map(|m| m.seq as i64)
        })
    }

    /// `external fun nativeFinish(handle: Long): Int` — kaynak bitti.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeFinish(
        _env: *mut JniEnv,
        _class: JObject,
        handle: i64,
    ) -> i32 {
        call(handle, |s| {
            s.pipeline.finish();
            Ok(0)
        }) as i32
    }

    /// `external fun nativeSetHint(handle: Long, seq: Long, hint: Int): Int`
    /// — 0 olağan, 1 hassas (Mod A), 2 bypass.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeSetHint(
        _env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        seq: i64,
        hint: i32,
    ) -> i32 {
        call(handle, |s| {
            let hint = Hint::from_i32(hint).ok_or(CoreError::BadArgument)?;
            if seq < 0 {
                return Err(CoreError::BadArgument);
            }
            s.pipeline.set_hint(seq as u64, hint).map(|_| 0)
        }) as i32
    }

    /// `external fun nativeReadWindowFrame(handle: Long, pos: Int, dst: ByteBuffer): Int`
    /// — pencere konumundaki kareyi NV12 olarak kopyalar. 1: kopyalandı, 0: dolgu konumu.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `dst` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeReadWindowFrame(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        pos: i32,
        dst: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let dst = match unsafe { buffer_mut(env, dst, 1) } {
            Ok(buf) => buf,
            Err(error) => return error.code(),
        };
        call(handle, |s| {
            if !(0..WINDOW_FRAMES as i32).contains(&pos) {
                return Err(CoreError::BadArgument);
            }
            let bytes = s.pipeline.geometry().frame_bytes();
            if dst.len() < bytes {
                return Err(CoreError::BufferTooSmall);
            }
            let io = s.pipeline.window_io()?;
            match io.frame(pos as usize) {
                Some(lease) => {
                    dst[..bytes].copy_from_slice(lease.data());
                    Ok(1)
                }
                None => Ok(0),
            }
        }) as i32
    }

    /// `external fun nativeSubmitEstimate(...)`: bir pencere konumunun model çıktısını verir.
    /// `y`/`weight`: ROI genişliği × ROI yüksekliği; `uv`: ROI genişliği × ROI yüksekliği / 2
    /// bayt (ROI, çift sınırlara hizalanmış hâliyle; hizalı ROI 0 dönerse reddedilmiştir).
    ///
    /// # Safety
    /// `env` geçerli olmalı; üç tampon da doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeSubmitEstimate(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        pos: i32,
        roi_x: i32,
        roi_y: i32,
        roi_w: i32,
        roi_h: i32,
        y_buf: JObject,
        uv_buf: JObject,
        weight_buf: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let buffers = unsafe {
            (|| {
                Ok::<_, CoreError>((
                    buffer_mut(env, y_buf, 1)?,
                    buffer_mut(env, uv_buf, 1)?,
                    buffer_mut(env, weight_buf, 1)?,
                ))
            })()
        };
        let (y_src, uv_src, w_src) = match buffers {
            Ok(b) => b,
            Err(error) => return error.code(),
        };
        call(handle, |s| {
            if !(0..WINDOW_FRAMES as i32).contains(&pos)
                || roi_x < 0
                || roi_y < 0
                || roi_w <= 0
                || roi_h <= 0
            {
                return Err(CoreError::BadArgument);
            }
            let geo = s.pipeline.geometry();
            let asked = Rect::new(
                roi_x as usize,
                roi_y as usize,
                roi_w as usize,
                roi_h as usize,
            );
            let roi = asked.aligned(geo).ok_or(CoreError::BadArgument)?;
            if roi != asked {
                // Tampon yerleşimi belirsiz olmasın: çağıran hizalı ROI vermelidir.
                return Err(CoreError::BadArgument);
            }
            if y_src.len() < roi.w * roi.h
                || w_src.len() < roi.w * roi.h
                || uv_src.len() < roi.w * roi.h / 2
            {
                return Err(CoreError::BufferTooSmall);
            }
            let mut io = s.pipeline.window_io()?;
            let est = io
                .estimate_mut(pos as usize)
                .ok_or(CoreError::BadArgument)?;
            est.begin(roi)?;
            for row in 0..roi.h {
                let (ys, ws) = est.luma_rows_mut(roi.y + row);
                ys.copy_from_slice(&y_src[row * roi.w..(row + 1) * roi.w]);
                ws.copy_from_slice(&w_src[row * roi.w..(row + 1) * roi.w]);
            }
            for row in 0..roi.h / 2 {
                est.uv_row_mut(roi.y / 2 + row)
                    .copy_from_slice(&uv_src[row * roi.w..(row + 1) * roi.w]);
            }
            Ok(0)
        }) as i32
    }

    /// `external fun nativeCompleteWindow(handle: Long): Int`
    /// — tahmin verilmeyen konumlar orijinal hâliyle çıkar.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_nativeCompleteWindow(
        _env: *mut JniEnv,
        _class: JObject,
        handle: i64,
    ) -> i32 {
        call(handle, |s| s.pipeline.complete_window().map(|_| 0)) as i32
    }

    fn write_output_meta(meta_out: &mut [u8], meta: &OutputMeta) {
        put_i64(meta_out, 0, meta.seq as i64);
        put_i64(meta_out, 8, meta.pts_us);
        put_i32(meta_out, 16, meta.flags as i32);
        put_i32(meta_out, 20, 0);
        put_i64(meta_out, 24, meta.modified_pixels as i64);
    }

    /// `external fun nativePopOutputYuv420(...)`: sıradaki kareyi kodlayıcının `Image`
    /// düzlemlerine yazar. 1: kare yazıldı, 0: hazır kare yok, negatif: hata.
    /// `meta` (≥ 32 bayt): `i64 sıra, i64 ptsUs, i32 bayraklar, i32 0, i64 değişenPiksel`.
    ///
    /// # Safety
    /// `env` geçerli olmalı; tüm tamponlar doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePopOutputYuv420(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        y_buf: JObject,
        y_offset: i32,
        y_row_stride: i32,
        u_buf: JObject,
        u_offset: i32,
        v_buf: JObject,
        v_offset: i32,
        uv_row_stride: i32,
        uv_pixel_stride: i32,
        meta_buf: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let prepared = unsafe {
            (|| {
                Ok::<_, CoreError>((
                    plane(env, y_buf, y_offset, y_row_stride, 1)?,
                    plane(env, u_buf, u_offset, uv_row_stride, uv_pixel_stride)?,
                    plane(env, v_buf, v_offset, uv_row_stride, uv_pixel_stride)?,
                    buffer_mut(env, meta_buf, OUTPUT_META_BYTES)?,
                ))
            })()
        };
        let (y, u, v, meta_out) = match prepared {
            Ok(p) => p,
            Err(error) => return error.code(),
        };
        call(handle, |s| {
            let geo = s.pipeline.geometry();
            // Kareyi kuyruktan çıkarmadan önce hedefi doğrula: hata olursa kare kaybolmasın.
            check_planes(geo, &y, &u, &v, true)?;
            let Session { pipeline, scratch } = s;
            let Some(meta) = pipeline.pop_output(scratch)? else {
                return Ok(0);
            };
            // SAFETY: düzlemler JVM'in doğrudan tamponlarıdır ve çağrı boyunca geçerlidir.
            unsafe { unpack_nv12(geo, scratch, y, u, v)? };
            write_output_meta(meta_out, &meta);
            Ok(1)
        }) as i32
    }

    /// `external fun nativePopOutputNv12(handle: Long, dst: ByteBuffer, meta: ByteBuffer): Int`
    ///
    /// # Safety
    /// `env` geçerli olmalı; tamponlar doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativePopOutputNv12(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        dst: JObject,
        meta_buf: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let prepared = unsafe {
            (|| {
                Ok::<_, CoreError>((
                    buffer_mut(env, dst, 1)?,
                    buffer_mut(env, meta_buf, OUTPUT_META_BYTES)?,
                ))
            })()
        };
        let (dst, meta_out) = match prepared {
            Ok(p) => p,
            Err(error) => return error.code(),
        };
        call(handle, |s| match s.pipeline.pop_output(dst)? {
            Some(meta) => {
                write_output_meta(meta_out, &meta);
                Ok(1)
            }
            None => Ok(0),
        }) as i32
    }

    /// `external fun nativeStats(handle: Long, out: ByteBuffer): Int` — 14 adet `i64`
    /// (≥ 112 bayt): girenKare, çıkanKare, sahneGeçişi, değişenPiksel, korumaİhlali,
    /// sıraİhlali, enYüksekDoluluk, pencereA, pencereB, köprü, bypass, harman, sahne,
    /// havuzBayt.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `out` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_nativeStats(
        env: *mut JniEnv,
        _class: JObject,
        handle: i64,
        out: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let out = match unsafe { buffer_mut(env, out, STATS_BYTES) } {
            Ok(buf) => buf,
            Err(error) => return error.code(),
        };
        call(handle, |s| {
            let st = s.pipeline.stats();
            let values = [
                st.frames_in,
                st.frames_out,
                st.scene_cuts,
                st.modified_pixels,
                st.preservation_violations,
                st.order_violations,
                st.max_occupancy as u64,
                st.sched.windows_a,
                st.sched.windows_b,
                st.sched.bridges,
                st.sched.bypassed,
                st.sched.fused,
                st.sched.segments,
                s.pipeline.ring().pool_bytes() as u64,
            ];
            for (i, v) in values.iter().enumerate() {
                put_i64(out, i * 8, *v as i64);
            }
            Ok(values.len() as i64)
        }) as i32
    }

    // ---- VC-Core motor arayüzü (planın sözleşmesi) ------------------------------------------

    const ENGINE_MAGIC: u64 = 0x5643_454E_4731_3030; // "VCENG100"
    const JNI_NEW_STRING_UTF: usize = 167;

    struct EngineHandle {
        magic: u64,
        engine: Mutex<Engine>,
    }

    fn engine_call<F>(ctx: i64, f: F) -> i64
    where
        F: FnOnce(&mut Engine) -> Result<i64, CoreError>,
    {
        let ptr = ctx as usize as *const EngineHandle;
        if ptr.is_null() || !ptr.is_aligned() {
            return CoreError::InvalidHandle.code() as i64;
        }
        // SAFETY: ctx vcInitEngine tarafından Box::into_raw ile üretilir, vcDestroy'a kadar geçerlidir.
        let handle = unsafe { &*ptr };
        if handle.magic != ENGINE_MAGIC {
            return CoreError::InvalidHandle.code() as i64;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f(&mut lock(&handle.engine))
        }));
        match result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => e.code() as i64,
            Err(_) => CoreError::Internal.code() as i64,
        }
    }

    /// `external fun vcInitEngine(mode: Int, preferVulkan: Boolean): Long` — 0 otomatik (B, zor karede A),
    /// 1 yalnız B, 2 yalnız A. Android'de halka belleği AHardwareBuffer'dır (ayrılamazsa yığın).
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_vcInitEngine(
        _env: *mut JniEnv,
        _class: JObject,
        mode: i32,
        prefer_vulkan: u8,
    ) -> i64 {
        let Some(mode) = EngineMode::from_i32(mode) else {
            return CoreError::BadArgument.code() as i64;
        };
        let storage = if cfg!(target_os = "android") {
            StorageKind::HardwareBuffer
        } else {
            StorageKind::Heap
        };
        let handle = Box::new(EngineHandle {
            magic: ENGINE_MAGIC,
            engine: Mutex::new(Engine::new(mode, prefer_vulkan != 0, storage)),
        });
        Box::into_raw(handle) as usize as i64
    }

    /// `external fun vcDestroy(ctx: Long)`
    ///
    /// # Safety
    /// `ctx`, vcInitEngine'den dönen ve daha önce yok edilmemiş bir değer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcDestroy(
        _env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
    ) {
        let ptr = ctx as usize as *mut EngineHandle;
        if ptr.is_null() || !ptr.is_aligned() {
            return;
        }
        // SAFETY: çağıranın sözleşmesi; sihirli sayı çift serbest bırakmayı yakalar.
        unsafe {
            if (*ptr).magic != ENGINE_MAGIC {
                return;
            }
            (*ptr).magic = 0;
            drop(Box::from_raw(ptr));
        }
    }

    /// `external fun vcPushHardwareFrame(ctx: Long, buffer: HardwareBuffer, ptsUs: Long, flags: Int): Int`
    /// — YUV 4:2:0 donanım tamponunu (ör. ImageReader `Image.hardwareBuffer`) Java yığınına kopyalamadan
    /// halkaya alır. Karenin sıra numarası (≥ 0) veya negatif hata kodu.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `buffer` bir `android.hardware.HardwareBuffer` olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcPushHardwareFrame(
        env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
        buffer: JObject,
        pts_us: i64,
        flags: i32,
    ) -> i32 {
        #[cfg(target_os = "android")]
        {
            engine_call(ctx, |e| {
                use crate::hwbuf::*;
                if buffer.is_null() {
                    return Err(CoreError::BadArgument);
                }
                // SAFETY: env ve buffer JVM'den gelir; AHardwareBuffer çağrı boyunca HardwareBuffer nesnesiyle yaşar.
                unsafe {
                    let ahb = AHardwareBuffer_fromHardwareBuffer(env as *mut c_void, buffer);
                    if ahb.is_null() {
                        return Err(CoreError::BadArgument);
                    }
                    let mut desc = Desc::default();
                    AHardwareBuffer_describe(ahb, &mut desc);
                    let mut planes = Planes {
                        plane_count: 0,
                        planes: [Plane {
                            data: std::ptr::null_mut(),
                            pixel_stride: 0,
                            row_stride: 0,
                        }; 4],
                    };
                    // Üreticiye özel (IMPLEMENTATION_DEFINED) biçimler CPU için düzlemlere ayrılamayabilir;
                    // çağıran bu durumda Image düzlemlerinden (CPU yolu) iter.
                    if AHardwareBuffer_lockPlanes(
                        ahb,
                        USAGE_CPU_READ_OFTEN,
                        -1,
                        std::ptr::null(),
                        &mut planes,
                    ) != 0
                    {
                        return Err(CoreError::Unsupported);
                    }
                    if planes.plane_count < 3 {
                        AHardwareBuffer_unlock(ahb, std::ptr::null_mut());
                        return Err(CoreError::Unsupported);
                    }
                    let (w, h) = (desc.width as usize, desc.height as usize);
                    let mk = |p: Plane, rows: usize| RawPlane {
                        ptr: p.data as *mut u8,
                        len: p.row_stride as usize * rows,
                        row_stride: p.row_stride as usize,
                        pixel_stride: p.pixel_stride.max(1) as usize,
                    };
                    let result = e.push_planes(
                        w,
                        h,
                        mk(planes.planes[0], h),
                        mk(planes.planes[1], h / 2),
                        mk(planes.planes[2], h / 2),
                        pts_us,
                        flags as u32,
                        true,
                    );
                    AHardwareBuffer_unlock(ahb, std::ptr::null_mut());
                    result.map(|seq| seq as i64)
                }
            }) as i32
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (env, buffer, pts_us, flags);
            engine_call(ctx, |_| Err(CoreError::BadState)) as i32
        }
    }

    /// `external fun vcPushNv12(ctx: Long, width: Int, height: Int, frame: ByteBuffer, ptsUs: Long, flags: Int): Int`
    /// — CPU yolu (donanım tamponu olmayan kaynaklar ve testler için).
    ///
    /// # Safety
    /// `env` geçerli olmalı; `frame` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcPushNv12(
        env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
        width: i32,
        height: i32,
        frame: JObject,
        pts_us: i64,
        flags: i32,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let data = match unsafe { buffer_mut(env, frame, 1) } {
            Ok(b) => b,
            Err(e) => return e.code(),
        };
        if width <= 0 || height <= 0 {
            return CoreError::InvalidGeometry.code();
        }
        engine_call(ctx, |e| {
            e.push_nv12(width as usize, height as usize, data, pts_us, flags as u32)
                .map(|s| s as i64)
        }) as i32
    }

    /// `external fun vcPoll(ctx: Long, info: ByteBuffer): Int` — `nativePoll` ile aynı adım kodları ve düzen.
    ///
    /// # Safety
    /// `env` geçerli olmalı; `info` doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcPoll(
        env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
        info: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let info = match unsafe { buffer_mut(env, info, POLL_INFO_BYTES) } {
            Ok(b) => b,
            Err(e) => return e.code(),
        };
        engine_call(ctx, |e| {
            info[..POLL_INFO_BYTES].fill(0);
            let code = match e.poll() {
                Step::NeedFrames(n) => {
                    put_i32(info, 4, n as i32);
                    STEP_NEED_FRAMES
                }
                Step::Output(n) => {
                    put_i32(info, 4, n as i32);
                    STEP_OUTPUT
                }
                Step::Finished => STEP_FINISHED,
                Step::Window(plan) => {
                    put_i64(info, 8, plan.id as i64);
                    put_i32(info, 16, if plan.mode == Mode::A { 0 } else { 1 });
                    put_i32(info, 20, plan.slide as i32);
                    put_i64(info, 48, plan.base_seq);
                    put_i32(info, 56, plan.valid as i32);
                    STEP_WINDOW
                }
            };
            put_i32(info, 0, code);
            Ok(code as i64)
        }) as i32
    }

    /// `external fun vcProcessWindow(ctx: Long, forceModeA: Boolean): Int` — hazır pencereyi işler.
    /// 0: pencere yok, 1: Mod B, 2: Mod A, negatif: hata. Model ağırlıkları yüklenmediğinde onarıcı
    /// kimliktir (hiçbir piksel değişmez, çıkış girişle bit-bit aynı).
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_vcProcessWindow(
        _env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
        force_mode_a: u8,
    ) -> i32 {
        engine_call(ctx, |e| {
            e.process_window(force_mode_a != 0, &mut IdentityRestorer)
                .map(|v| v as i64)
        }) as i32
    }

    /// `external fun vcPopOutputNv12(ctx: Long, dst: ByteBuffer, meta: ByteBuffer): Int` — 1 kare, 0 yok.
    ///
    /// # Safety
    /// `env` geçerli olmalı; tamponlar doğrudan ByteBuffer olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcPopOutputNv12(
        env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
        dst: JObject,
        meta: JObject,
    ) -> i32 {
        // SAFETY: çağıranın sözleşmesi.
        let prepared = unsafe {
            (|| {
                Ok::<_, CoreError>((
                    buffer_mut(env, dst, 1)?,
                    buffer_mut(env, meta, OUTPUT_META_BYTES)?,
                ))
            })()
        };
        let (dst, meta_out) = match prepared {
            Ok(p) => p,
            Err(e) => return e.code(),
        };
        engine_call(ctx, |e| match e.pop_output(dst)? {
            Some(m) => {
                write_output_meta(meta_out, &m);
                Ok(1)
            }
            None => Ok(0),
        }) as i32
    }

    /// `external fun vcFinish(ctx: Long): Int`
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_video_cleaner_Core_vcFinish(
        _env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
    ) -> i32 {
        engine_call(ctx, |e| {
            e.finish();
            Ok(0)
        }) as i32
    }

    /// `external fun vcRunTesterAudit(ctx: Long): String` — VC-Tester çalışma zamanı raporu (JSON).
    ///
    /// # Safety
    /// `env` geçerli bir `JNIEnv*` olmalıdır.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn Java_com_video_cleaner_Core_vcRunTesterAudit(
        env: *mut JniEnv,
        _class: JObject,
        ctx: i64,
    ) -> JObject {
        let mut json = String::new();
        let code = engine_call(ctx, |e| {
            json = e.tester_audit_json();
            Ok(0)
        });
        if code != 0 {
            json = format!("{{\"error\":{code}}}");
        }
        let mut bytes = json.into_bytes();
        bytes.retain(|b| *b != 0 && b.is_ascii());
        bytes.push(0);
        type NewStringUtf =
            unsafe extern "system" fn(*mut JniEnv, *const std::ffi::c_char) -> JObject;
        // SAFETY: JNI belirtimi: işlev tablosunun 167. girişi NewStringUTF'tir; dize NUL ile biter ve ASCII'dir.
        unsafe {
            if env.is_null() || (*env).is_null() {
                return std::ptr::null_mut();
            }
            let f: NewStringUtf = std::mem::transmute(*(*env).add(JNI_NEW_STRING_UTF));
            f(env, bytes.as_ptr() as *const std::ffi::c_char)
        }
    }
}

// ============================================================================
// 11. Testler
// ============================================================================

#[cfg(test)]
mod tests {
    use super::jni::*;
    use super::*;
    use std::collections::{BTreeSet, HashMap};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Küçük, bağımlılıksız sözde rastgele üreteç (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf {
                *b = (self.next() >> 32) as u8;
            }
        }
    }

    fn small() -> FrameGeometry {
        FrameGeometry::new(64, 48).unwrap()
    }

    /// Sıra numarasına özgü, her baytı farklı kare içeriği.
    fn frame_content(geo: FrameGeometry, seq: u64) -> Vec<u8> {
        let mut buf = vec![0u8; geo.frame_bytes()];
        Rng::new(seq + 1).fill(&mut buf);
        buf
    }

    // ---- geometri ve bellek --------------------------------------------------

    #[test]
    fn geometry_and_pool_match_spec() {
        let geo = FrameGeometry::HD720;
        assert_eq!(geo.frame_bytes(), 1_382_400);
        let ring = FrameRing::new(geo).unwrap();
        assert_eq!(ring.pool_bytes(), 1_382_400 * RING_FRAMES); // 22 × 720p NV12 = 30,4 MB
        assert_eq!(ring.base_addr() % page_size(), 0);
        assert!(page_size() == 4096 || page_size() == 16384 || page_size() == 65536);
        assert_eq!(
            FrameGeometry::new(1281, 720),
            Err(CoreError::InvalidGeometry)
        );
        assert_eq!(FrameGeometry::new(1280, 0), Err(CoreError::InvalidGeometry));
        assert_eq!(
            FrameGeometry::new(16384, 720),
            Err(CoreError::InvalidGeometry)
        );
    }

    // ---- halka ve sahiplik ---------------------------------------------------

    #[test]
    fn ring_write_read_retire() {
        let geo = small();
        let ring = FrameRing::new(geo).unwrap();
        for seq in 0..RING_FRAMES as u64 {
            let mut w = ring.begin_write().unwrap();
            w.data().copy_from_slice(&frame_content(geo, seq));
            let meta = w.commit(seq as i64 * 16_667, 0).unwrap();
            assert_eq!(meta.seq, seq);
        }
        assert_eq!(ring.len(), 22);
        assert_eq!(ring.begin_write().err(), Some(CoreError::RingFull));
        for seq in [0u64, 7, 21] {
            let lease = ring.read(seq).unwrap();
            assert_eq!(lease.data(), &frame_content(geo, seq)[..]);
            assert_eq!(lease.meta().pts_us, seq as i64 * 16_667);
            assert_eq!(lease.luma().len() + lease.chroma().len(), geo.frame_bytes());
        }
        assert_eq!(ring.retire_before(11), 11);
        assert_eq!(ring.bounds(), (11, 22));
        assert_eq!(ring.read(10).err(), Some(CoreError::UnknownFrame));
        assert!(ring.meta(10).is_none() && ring.meta(11).is_some());
        // Boşalan yuvalar yeniden kullanılır; eski kareler bozulmaz.
        for seq in 22..33u64 {
            let mut w = ring.begin_write().unwrap();
            w.data().copy_from_slice(&frame_content(geo, seq));
            w.commit(seq as i64 * 16_667, 0).unwrap();
        }
        for seq in 11..33u64 {
            assert_eq!(ring.read(seq).unwrap().data(), &frame_content(geo, seq)[..]);
        }
    }

    #[test]
    fn ring_rejects_bad_usage() {
        let ring = FrameRing::new(small()).unwrap();
        let w = ring.begin_write().unwrap();
        assert_eq!(ring.begin_write().err(), Some(CoreError::WriteInProgress));
        drop(w); // commit edilmeyen kira kare eklemez
        assert!(ring.is_empty());

        ring.begin_write().unwrap().commit(100, 0).unwrap();
        assert_eq!(
            ring.begin_write().unwrap().commit(100, 0).err(),
            Some(CoreError::NonMonotonicPts)
        );
        assert_eq!(
            ring.begin_write().unwrap().commit(50, 0).err(),
            Some(CoreError::NonMonotonicPts)
        );
        assert_eq!(ring.len(), 1);
        for i in 1..RING_FRAMES as i64 {
            ring.begin_write().unwrap().commit(100 + i, 0).unwrap();
        }

        // Okuyucusu olan yuva halkadan çıkarılsa bile üzerine yazılamaz.
        let reader = ring.read(0).unwrap();
        ring.retire_before(1);
        assert_eq!(ring.begin_write().err(), Some(CoreError::SlotBusy));
        assert_eq!(reader.meta().seq, 0);
        drop(reader);
        ring.begin_write().unwrap().commit(1000, 0).unwrap();
        assert_eq!(ring.bounds(), (1, 23));
    }

    #[test]
    fn ring_concurrent_readers_never_see_torn_frames() {
        let ring = FrameRing::new(small()).unwrap();
        let stop = AtomicBool::new(false);
        let verified = AtomicU64::new(0);
        let pattern = |seq: u64| (seq.wrapping_mul(31).wrapping_add(7)) as u8;
        std::thread::scope(|scope| {
            for t in 0..3u64 {
                let (ring, stop, verified) = (&ring, &stop, &verified);
                scope.spawn(move || {
                    let mut rng = Rng::new(1000 + t);
                    while !stop.load(Ordering::Relaxed) {
                        let (head, tail) = ring.bounds();
                        if tail == head {
                            continue;
                        }
                        if let Ok(lease) = ring.read(head + rng.below(tail - head)) {
                            let expect = pattern(lease.meta().seq);
                            assert!(
                                lease.data().iter().all(|&b| b == expect),
                                "yırtık kare okundu"
                            );
                            verified.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
            // Yazıcı, okuyucular yeterince doğrulama yapana kadar sürer: test, iş
            // parçacıklarının ne zaman zamanlandığına dair bir varsayıma dayanmaz.
            let mut seq = 0u64;
            while seq < 4000 || verified.load(Ordering::Relaxed) < 3000 {
                assert!(seq < 50_000_000, "okuyucular ilerlemiyor");
                match ring.begin_write() {
                    Ok(mut w) => {
                        let value = pattern(seq);
                        let (first, second) = w.data().split_at_mut(1000);
                        first.fill(value);
                        std::thread::yield_now();
                        second.fill(value);
                        w.commit(seq as i64, 0).unwrap();
                        seq += 1;
                    }
                    Err(CoreError::RingFull) => {
                        let (head, _) = ring.bounds();
                        ring.retire_before(head + 7);
                    }
                    Err(CoreError::SlotBusy) => std::thread::yield_now(),
                    Err(other) => panic!("beklenmeyen hata: {other}"),
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert!(verified.load(Ordering::Relaxed) >= 3000);
    }

    // ---- SIMD çekirdekleri ---------------------------------------------------

    #[test]
    fn kernels_match_portable_reference() {
        println!("çekirdek arka ucu: {}", kernels::backend());
        let mut rng = Rng::new(42);
        for trial in 0..400 {
            let n = match trial {
                0 => 0,
                1 => 31,
                2 => 32,
                3 => 33,
                _ => rng.below(700) as usize,
            };
            let (mut a, mut b, mut m) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
            rng.fill(&mut a);
            rng.fill(&mut b);
            rng.fill(&mut m);
            for v in m.iter_mut() {
                // Maskede hem 0 hem sıfırdan farklı (yalnızca 255 değil) değerler bulunsun.
                *v = if *v < 150 { 0 } else { *v & 0x81 | 1 };
            }
            assert_eq!(
                kernels::sad_u8(&a, &b),
                kernels::portable::sad_u8(&a, &b),
                "sad n={n}"
            );

            let (mut got, mut want) = (vec![0u8; n], vec![0u8; n]);
            kernels::select_u8(&mut got, &a, &b, &m);
            kernels::portable::select_u8(&mut want, &a, &b, &m);
            assert_eq!(got, want, "select n={n}");

            let mut in_place = b.clone();
            kernels::select_in_place(&mut in_place, &a, &m);
            assert_eq!(in_place, want, "select_in_place n={n}");
        }
        // En kötü durum: 0 ile 255 arasındaki fark taşmadan toplanmalı.
        let (zeros, full) = (vec![0u8; 1 << 16], vec![255u8; 1 << 16]);
        assert_eq!(kernels::sad_u8(&zeros, &full), 255 << 16);
    }

    #[test]
    fn max5_and_unblend_match_portable_reference() {
        let mut rng = Rng::new(99);
        for trial in 0..300 {
            let n = match trial {
                0 => 0,
                1 => 15,
                2 => 16,
                3 => 31,
                4 => 32,
                5 => 33,
                _ => rng.below(900) as usize,
            };
            let mut rows: Vec<Vec<u8>> = (0..5).map(|_| vec![0u8; n]).collect();
            for r in rows.iter_mut() {
                rng.fill(r);
            }
            let refs = [
                &rows[0][..],
                &rows[1][..],
                &rows[2][..],
                &rows[3][..],
                &rows[4][..],
            ];
            let (mut got, mut want) = (vec![0u8; n], vec![0u8; n]);
            kernels::max5_u8(&mut got, refs);
            kernels::portable::max5_u8(&mut want, refs);
            assert_eq!(got, want, "max5 n={n}");

            let mut raw = vec![0u8; 4 * n];
            rng.fill(&mut raw);
            let img: Vec<u16> = (0..n)
                .map(|i| u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) & 0x0fff)
                .collect();
            let fg: Vec<u16> = (0..n)
                .map(|i| u16::from_le_bytes([raw[2 * n + 2 * i], raw[2 * n + 2 * i + 1]]) & 0x0fff)
                .collect();
            let mut alpha = vec![0u8; n];
            rng.fill(&mut alpha);
            for a in alpha.iter_mut() {
                *a = (*a).min(254);
            }
            let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
            kernels::unblend_u16(&mut got, &img, &fg, &alpha);
            kernels::portable::unblend_u16(&mut want, &img, &fg, &alpha);
            assert_eq!(got, want, "unblend n={n}");
        }
        // Uç değerler: α = 0 → B = I; α = 254 ve F = 0 → kırpma 65535'e dayanır.
        let img = vec![65535u16; 32];
        let fg = vec![0u16; 32];
        let (mut got, mut want) = (vec![0u16; 32], vec![0u16; 32]);
        for a in [0u8, 1, 128, 254] {
            let alpha = vec![a; 32];
            kernels::unblend_u16(&mut got, &img, &fg, &alpha);
            kernels::portable::unblend_u16(&mut want, &img, &fg, &alpha);
            assert_eq!(got, want, "uç α={a}");
        }
        assert!(got.iter().all(|&v| v == 65535));
    }

    #[test]
    fn dilate5x5_matches_naive_window_max() {
        let mut rng = Rng::new(5);
        for &(w, h) in &[(1usize, 1usize), (7, 3), (33, 17), (64, 40), (130, 9)] {
            let mut mask = vec![0u8; w * h];
            rng.fill(&mut mask);
            for v in mask.iter_mut() {
                *v = if *v > 240 { *v } else { 0 };
            }
            let mut got = vec![0u8; w * h];
            dilate5x5(&mask, w, h, &mut got);
            for y in 0..h {
                for x in 0..w {
                    let mut m = 0u8;
                    for yy in y.saturating_sub(2)..(y + 3).min(h) {
                        for xx in x.saturating_sub(2)..(x + 3).min(w) {
                            m = m.max(mask[yy * w + xx]);
                        }
                    }
                    assert_eq!(got[y * w + x], m, "({x},{y}) {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn alpha_unblend_recovers_background() {
        let u = AlphaUnblend::new();
        let n = 4096;
        let lin = |v: u8| (v as f64 / 255.0).powf(2.4);
        let gam = |l: f64| (l.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        let background: Vec<u8> = (0..n).map(|i| (16 + (i * 7) % 220) as u8).collect();
        let overlay = 235u8;
        let alpha: Vec<u8> = (0..n).map(|i| (i % 240) as u8).collect();
        // Yarı saydam altyazı: doğrusal ışıkta karışım, sonra 8 bit gama.
        let mut plane: Vec<u8> = (0..n)
            .map(|i| {
                let a = alpha[i] as f64 / 255.0;
                gam(a * lin(overlay) + (1.0 - a) * lin(background[i]))
            })
            .collect();
        let (resolved, deferred) =
            u.unblend_plane(&mut plane, &alpha, overlay, true, AlphaUnblend::MAX_ALPHA);
        assert_eq!(
            deferred,
            alpha
                .iter()
                .filter(|&&a| a > AlphaUnblend::MAX_ALPHA)
                .count()
        );
        assert_eq!(
            resolved,
            alpha
                .iter()
                .filter(|&&a| a > 0 && a <= AlphaUnblend::MAX_ALPHA)
                .count()
        );
        // Hata doğrusal ışıkta ölçülür: 8 bit gama nicemleme adımı (q) 1/(1−α) ile büyür; koyu arka
        // planda gama değerindeki büyük fark fiziksel bilgi kaybıdır, yöntemin hatası değildir.
        let step = |v: u8| lin(v.saturating_add(1)) - lin(v);
        let mut total = 0f64;
        let mut count = 0f64;
        for i in 0..n {
            let a = alpha[i];
            if a == 0 || a > AlphaUnblend::MAX_ALPHA {
                continue;
            }
            let observed = {
                let af = a as f64 / 255.0;
                gam(af * lin(overlay) + (1.0 - af) * lin(background[i]))
            };
            let bound =
                (step(observed) / (1.0 - a as f64 / 255.0) + step(plane[i]) + step(background[i]))
                    * 1.01
                    + 2.0 / 65535.0;
            let err = (lin(plane[i]) - lin(background[i])).abs();
            assert!(
                err <= bound,
                "i={i} α={a} B={} bulunan={} hata={err:.5} sınır={bound:.5}",
                background[i],
                plane[i]
            );
            total += err;
            count += 1.0;
        }
        assert!(
            total / count < 0.01,
            "ortalama doğrusal hata {}",
            total / count
        );
        // α > eşik pikselleri değişmemeli (modele bırakılır).
        for i in 0..n {
            if alpha[i] > AlphaUnblend::MAX_ALPHA {
                let a = alpha[i] as f64 / 255.0;
                assert_eq!(
                    plane[i],
                    gam(a * lin(overlay) + (1.0 - a) * lin(background[i]))
                );
            }
        }
    }

    #[test]
    fn haar_matches_reference_and_roundtrips_losslessly() {
        let mut rng = Rng::new(7);
        for trial in 0..200 {
            let n = if trial < 4 {
                [0usize, 15, 16, 17][trial]
            } else {
                rng.below(200) as usize
            };
            let (mut r0, mut r1) = (vec![0u8; 2 * n], vec![0u8; 2 * n]);
            rng.fill(&mut r0);
            rng.fill(&mut r1);
            if trial % 5 == 0 {
                // Uç değerler: katsayılar ±510 ve 1020 sınırlarına dayanır.
                for (i, (a, b)) in r0.iter_mut().zip(r1.iter_mut()).enumerate() {
                    *a = if i % 2 == 0 { 255 } else { 0 };
                    *b = if (i / 2) % 2 == 0 { 255 } else { 0 };
                }
            }
            let mut got = [vec![0i16; n], vec![0i16; n], vec![0i16; n], vec![0i16; n]];
            let mut want = got.clone();
            {
                let [ll, hl, lh, hh] = &mut got;
                kernels::haar_fwd_rows(&r0, &r1, ll, hl, lh, hh);
                let [ll, hl, lh, hh] = &mut want;
                kernels::portable::haar_fwd_rows(&r0, &r1, ll, hl, lh, hh);
            }
            assert_eq!(got, want, "haar_fwd n={n}");

            let (mut b0, mut b1) = (vec![0u8; 2 * n], vec![0u8; 2 * n]);
            kernels::haar_inv_rows(&got[0], &got[1], &got[2], &got[3], &mut b0, &mut b1);
            assert_eq!((b0, b1), (r0, r1), "haar gidiş-dönüş kayıpsız olmalı n={n}");
        }
    }

    #[test]
    fn haar_inverse_matches_reference_on_arbitrary_coefficients() {
        let mut rng = Rng::new(99);
        for trial in 0..200 {
            let n = 16 + rng.below(100) as usize;
            let mut bands = [vec![0i16; n], vec![0i16; n], vec![0i16; n], vec![0i16; n]];
            for band in bands.iter_mut() {
                for v in band.iter_mut() {
                    *v = match (trial + rng.below(4) as usize) % 4 {
                        0 => rng.next() as i16, // tüm i16 aralığı (doygunluk)
                        1 => (rng.below(4096) as i16) - 2048,
                        2 => [i16::MIN, i16::MAX, -1, 0, 1, 2, 3][rng.below(7) as usize],
                        _ => (rng.below(1021)) as i16,
                    };
                }
            }
            let (mut g0, mut g1) = (vec![0u8; 2 * n], vec![0u8; 2 * n]);
            let (mut w0, mut w1) = (vec![0u8; 2 * n], vec![0u8; 2 * n]);
            kernels::haar_inv_rows(&bands[0], &bands[1], &bands[2], &bands[3], &mut g0, &mut g1);
            kernels::portable::haar_inv_rows(
                &bands[0], &bands[1], &bands[2], &bands[3], &mut w0, &mut w1,
            );
            assert_eq!((g0, g1), (w0, w1), "haar_inv n={n}");
        }
    }

    #[test]
    fn haar_bands_plane_roundtrip_and_detail_energy() {
        let (w, h, stride) = (96usize, 40usize, 100usize);
        let mut plane = vec![0u8; stride * h];
        Rng::new(5).fill(&mut plane);
        let mut bands = HaarBands::new(w, h).unwrap();
        bands.forward(&plane, stride);
        assert!(
            bands.detail_energy() > 50.0,
            "gürültü yüksek frekans enerjisi taşımalı"
        );
        let mut back = vec![0u8; stride * h];
        bands.inverse(&mut back, stride);
        for y in 0..h {
            assert_eq!(
                &back[y * stride..y * stride + w],
                &plane[y * stride..y * stride + w]
            );
        }
        let flat = vec![128u8; stride * h];
        bands.forward(&flat, stride);
        assert_eq!(bands.detail_energy(), 0.0);
        assert!(bands.ll.iter().all(|&v| v == 512));
        assert!(HaarBands::new(95, 40).is_err());
    }

    // ---- sahne geçişi --------------------------------------------------------

    /// Yumuşak bir doku: sahne kimliğine göre farklı parlaklık ve desen.
    fn scene_luma(geo: FrameGeometry, scene: u64, shift: usize, noise_seed: u64) -> Vec<u8> {
        let (w, h) = (geo.width(), geo.height());
        let mut rng = Rng::new(noise_seed);
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let xx = x + shift;
                let base = match scene {
                    0 => 60 + ((xx / 40 + y / 40) % 2) as i32 * 30,
                    1 => 170 + ((xx / 16) % 3) as i32 * 20 - (y as i32 * 40 / h as i32),
                    _ => 20 + (y as i32 * 200 / h as i32),
                };
                let noise = (rng.next() % 5) as i32 - 2;
                luma[y * w + x] = (base + noise).clamp(0, 255) as u8;
            }
        }
        luma
    }

    #[test]
    fn scene_detector_flags_hard_cuts_only() {
        let geo = FrameGeometry::new(320, 180).unwrap();
        let mut det = SceneDetector::new(geo, SceneCutConfig::default());
        let mut cuts = Vec::new();
        // 0..19: sahne 0 (yavaş kaydırma), 20..39: sahne 1, 40..59: sahne 2.
        for i in 0..60u64 {
            let scene = i / 20;
            let verdict = det.feed(&scene_luma(geo, scene, (i % 20) as usize, i));
            if verdict.cut {
                cuts.push(i);
            }
        }
        assert_eq!(cuts, vec![20, 40]);

        // Aynı sahnede kademeli parlaklık değişimi (fade) sert geçiş sayılmaz.
        let mut det = SceneDetector::new(geo, SceneCutConfig::default());
        let base = scene_luma(geo, 0, 0, 1);
        for step in 0..30u32 {
            let frame: Vec<u8> = base
                .iter()
                .map(|&v| (v as u32 + step * 3).min(255) as u8)
                .collect();
            assert!(
                !det.feed(&frame).cut,
                "kademeli değişim kesme sayıldı (adım {step})"
            );
        }
    }

    // ---- zamanlayıcı ---------------------------------------------------------

    /// Zamanlayıcıyı kare içeriği olmadan sürer; verilen planları ve çıkışları toplar.
    fn drive_scheduler(
        total: u64,
        cuts: &BTreeSet<u64>,
        hint_of: &dyn Fn(u64) -> Hint,
    ) -> (Vec<WindowPlan>, Vec<OutputItem>, Scheduler) {
        let mut s = Scheduler::new();
        let (mut plans, mut outputs) = (Vec::new(), Vec::new());
        loop {
            match s.poll() {
                Step::NeedFrames(n) => {
                    assert!(n >= 1);
                    if s.pushed() == total {
                        s.on_eos();
                        continue;
                    }
                    for _ in 0..n {
                        if s.pushed() == total {
                            break;
                        }
                        let seq = s.on_push(cuts.contains(&s.pushed())).unwrap();
                        s.set_hint(seq, hint_of(seq)).unwrap();
                    }
                }
                Step::Window(plan) => {
                    plans.push(plan);
                    s.complete(plan.id).unwrap();
                }
                Step::Output(n) => {
                    for _ in 0..n {
                        outputs.push(s.pop_output().unwrap());
                    }
                }
                Step::Finished => return (plans, outputs, s),
            }
        }
    }

    #[test]
    fn scheduler_steady_state_is_window_15_with_stride_11() {
        let (plans, outputs, s) = drive_scheduler(75, &BTreeSet::new(), &|_| Hint::Normal);
        let bases: Vec<i64> = plans.iter().map(|p| p.base_seq).collect();
        assert_eq!(bases, vec![0, 11, 22, 33, 44, 55, 64]);
        for plan in &plans[..6] {
            assert_eq!(plan.mode, Mode::B);
            assert_eq!(plan.slide, 11);
            assert_eq!(plan.emit, (0, 11));
            assert_eq!(plan.valid, (1 << 15) - 1, "tam 15 karelik pencere");
            assert_eq!(plan.cache, (11, 15));
        }
        assert_ne!(plans[0].flags & PLAN_FLAG_SEGMENT_START, 0);
        // Son pencere: 66..74 kesinleşir; [64, 75) geçerli, gelecek tarafı dolgu.
        let last = plans[6];
        assert_eq!((last.slide, last.emit), (9, (2, 11)));
        assert_eq!(last.valid, (1 << 11) - 1);
        assert_eq!(last.cache, (11, 11));
        assert_ne!(last.flags & PLAN_FLAG_SEGMENT_END, 0);

        assert_eq!(outputs.len(), 75);
        for (i, item) in outputs.iter().enumerate() {
            assert_eq!(item.seq, i as u64);
            let fused = item.flags & OUT_FLAG_FUSED != 0;
            assert_eq!(
                fused,
                i >= 11 && (i - 11) % 11 < 4,
                "yalnızca örtüşen 4 kare iki pencereden harmanlanır (kare {i})"
            );
        }
        assert_eq!(s.stats().windows_b, 7);
        assert_eq!(s.stats().fused, 24);
        // Doğrusal geçiş: örtüşen iki tahminin öncelikleri (4-k) : (k+1).
        for k in 0..4 {
            assert_eq!(plans[1].prior_at(k), (k + 1) as u8);
            assert_eq!(plans[0].prior_at(11 + k), (4 - k) as u8);
        }
    }

    #[test]
    fn scheduler_pads_short_and_tail_windows() {
        // 20 kare: [0,15) penceresi + son 9 kare için [9,24) dolgulu pencere.
        let (plans, outputs, _) = drive_scheduler(20, &BTreeSet::new(), &|_| Hint::Normal);
        assert_eq!(plans.len(), 2);
        assert_eq!(
            (
                plans[0].base_seq,
                plans[0].valid_count(),
                plans[0].emit,
                plans[0].cache
            ),
            (0, 15, (0, 11), (11, 15))
        );
        assert_eq!(
            (plans[1].base_seq, plans[1].slide, plans[1].emit),
            (9, 9, (2, 11))
        );
        assert_eq!(plans[1].valid, (1 << 11) - 1);
        assert_eq!(outputs.len(), 20);

        // 4 kare: tek pencere, düşük bağlam işaretli.
        let (plans, outputs, _) = drive_scheduler(4, &BTreeSet::new(), &|_| Hint::Normal);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].base_seq, -7);
        assert_eq!(plans[0].valid, 0b1111 << 7);
        assert_eq!(plans[0].emit, (7, 11));
        assert_ne!(plans[0].flags & PLAN_FLAG_LOW_CONTEXT, 0);
        assert!(outputs.iter().all(|o| o.flags & OUT_FLAG_LOW_CONTEXT != 0));

        // Boş akış.
        let (plans, outputs, _) = drive_scheduler(0, &BTreeSet::new(), &|_| Hint::Normal);
        assert!(plans.is_empty() && outputs.is_empty());
    }

    #[test]
    fn scheduler_windows_never_cross_scene_cuts() {
        let cuts: BTreeSet<u64> = [37u64, 38, 90].into_iter().collect();
        let (plans, outputs, s) = drive_scheduler(120, &cuts, &|_| Hint::Normal);
        let segment = |seq: u64| cuts.iter().filter(|&&c| c <= seq).count();
        for plan in &plans {
            let seqs: Vec<u64> = (0..WINDOW_FRAMES).filter_map(|p| plan.seq_at(p)).collect();
            assert!(!seqs.is_empty());
            assert!(
                seqs.iter().all(|&q| segment(q) == segment(seqs[0])),
                "pencere sahne geçişini aştı: {plan:?}"
            );
        }
        assert_eq!(outputs.len(), 120);
        assert_eq!(s.stats().segments, 3);
    }

    #[test]
    fn scheduler_precision_and_bypass_hints() {
        // 20..22 hassas: önce B(9) ile 11..19, sonra üç Mod A adımı, sonra Mod A'nın kırptığı
        // önbellekte tek kare (23) kaldığı için B(8): tek tahminli kareler de en az 4 geçmiş görür.
        let hint = |seq: u64| {
            if (20..23).contains(&seq) {
                Hint::Precision
            } else {
                Hint::Normal
            }
        };
        let (plans, outputs, s) = drive_scheduler(60, &BTreeSet::new(), &hint);
        let summary: Vec<(Mode, u8, i64)> = plans
            .iter()
            .take(6)
            .map(|p| (p.mode, p.slide, p.base_seq))
            .collect();
        assert_eq!(
            summary,
            vec![
                (Mode::B, 11, 0),
                (Mode::B, 9, 9),
                (Mode::A, 1, 13),
                (Mode::A, 1, 14),
                (Mode::A, 1, 15),
                (Mode::B, 8, 20)
            ]
        );
        for plan in plans.iter().filter(|p| p.mode == Mode::A) {
            assert_eq!(plan.emit, (7, 8));
            assert_eq!(plan.past_context(7), 7);
            assert_eq!(plan.future_context(7), 7);
        }
        assert_eq!(plans[5].emit, (3, 11));
        assert_eq!(s.stats().windows_a, 3);
        assert_eq!(
            s.stats().bridges,
            0,
            "kırpılmış önbellek köprüye gerek bırakmaz"
        );
        for item in &outputs {
            assert_eq!(
                item.flags & OUT_FLAG_MODE_A != 0,
                (20..23).contains(&item.seq)
            );
        }

        // 0..39 bypass, 40'tan itibaren altyazı: 40. kare için önce köprü geçişi
        // çalışır ki altyazıdan önceki temiz kareler bağlam olarak kullanılsın.
        let hint = |seq: u64| if seq < 40 { Hint::Bypass } else { Hint::Normal };
        let (plans, outputs, s) = drive_scheduler(100, &BTreeSet::new(), &hint);
        assert_eq!(plans[0].flags & PLAN_FLAG_BRIDGE, PLAN_FLAG_BRIDGE);
        assert_eq!(
            (
                plans[0].base_seq,
                plans[0].slide,
                plans[0].emit,
                plans[0].cache
            ),
            (29, 0, (11, 11), (11, 15))
        );
        assert_eq!((plans[1].base_seq, plans[1].slide), (40, 11));
        assert_eq!(s.stats().bypassed, 40);
        assert_eq!(s.stats().bridges, 1);
        for item in &outputs {
            if item.seq < 40 {
                assert_eq!(item.source, OutputSource::Original);
            } else {
                assert_eq!(
                    item.flags & OUT_FLAG_FUSED != 0,
                    (item.seq - 40) % 11 < 4,
                    "kare {}: köprüden sonra her pencerenin ilk 4 karesi harmanlanır",
                    item.seq
                );
            }
        }
    }

    #[test]
    fn scheduler_rejects_late_or_unknown_hints() {
        let mut s = Scheduler::new();
        assert_eq!(s.set_hint(0, Hint::Bypass), Err(CoreError::UnknownFrame));
        for _ in 0..11 {
            s.on_push(false).unwrap();
        }
        assert_eq!(s.poll(), Step::NeedFrames(4)); // B(11) kararı verildi: 0..10 artık değişmez
        assert_eq!(s.set_hint(3, Hint::Precision), Err(CoreError::BadState));
        assert_eq!(s.complete(0), Err(CoreError::BadState));
        s.on_eos();
        assert_eq!(s.on_push(false), Err(CoreError::StreamFinished));
    }

    /// Rastgele akışlar: uzunluk, sahne geçişleri, ipuçları ve itme düzeni rastgele.
    /// Denetlenen değişmezler:
    ///  1. Her kare tam bir kez ve sırayla çıkar.
    ///  2. Halka doluluğu 22'yi aşmaz; planlar yalnızca halkadaki kareleri kullanır.
    ///  3. Pencereler sahne geçişini aşmaz.
    ///  4. Bağlam güvencesi: onarılan her kare en az min(4, sahne başına uzaklık) geçmiş ve
    ///     min(4, sahne sonuna uzaklık) gelecek kareye erişmiştir (Mod A: min(7, …)).
    #[test]
    fn scheduler_random_streams_keep_invariants() {
        let mut rng = Rng::new(2026);
        for case in 0..600 {
            let total = match case % 6 {
                0 => rng.below(40),
                1 => 10 + rng.below(14),
                _ => rng.below(260),
            };
            let cut_rate = [0u64, 0, 4, 15, 40][rng.below(5) as usize];
            let cuts: BTreeSet<u64> = (1..total)
                .filter(|_| cut_rate != 0 && rng.below(100) < cut_rate / 2 + 1)
                .collect();
            let style = rng.below(5);
            let hints: Vec<Hint> = {
                let mut v = Vec::with_capacity(total as usize);
                let mut current = Hint::Normal;
                for _ in 0..total {
                    if rng.below(100) < [0u64, 8, 25, 60, 100][style as usize] {
                        current =
                            [Hint::Normal, Hint::Precision, Hint::Bypass][rng.below(3) as usize];
                    }
                    v.push(current);
                }
                v
            };
            let seg_start = |seq: u64| cuts.range(..=seq).next_back().copied().unwrap_or(0);
            let seg_end = |seq: u64| cuts.range(seq + 1..).next().copied().unwrap_or(total);

            let mut s = Scheduler::new();
            let mut next_out = 0u64;
            let mut past = vec![0u32; total as usize];
            let mut future = vec![0u32; total as usize];
            let mut emit_ctx: HashMap<u64, (u32, u32)> = HashMap::new();
            let mut cache_ctx: HashMap<u64, (u32, u32)> = HashMap::new();
            let mut old_cache_ctx: HashMap<u64, (u32, u32)> = HashMap::new();
            let mut steps = 0u64;
            loop {
                steps += 1;
                assert!(steps < 100_000, "zamanlayıcı ilerlemiyor (durum {case})");
                match s.poll() {
                    Step::NeedFrames(n) => {
                        assert!(n >= 1 && n <= s.capacity_left(), "durum {case}: n={n}");
                        if s.pushed() == total {
                            s.on_eos();
                            continue;
                        }
                        for _ in 0..1 + rng.below(n as u64) {
                            if s.pushed() == total {
                                break;
                            }
                            let seq = s.on_push(cuts.contains(&s.pushed())).unwrap();
                            s.set_hint(seq, hints[seq as usize]).unwrap();
                        }
                    }
                    Step::Window(plan) => {
                        let seqs: Vec<u64> =
                            (0..WINDOW_FRAMES).filter_map(|p| plan.seq_at(p)).collect();
                        assert!(!seqs.is_empty());
                        // Geçerli konumlar kesintisiz bir aralıktır.
                        assert_eq!(seqs.len() as u64, seqs[seqs.len() - 1] - seqs[0] + 1);
                        for &q in &seqs {
                            assert!(
                                q >= s.retired() && q < s.pushed(),
                                "durum {case}: plan halkada olmayan kareyi istedi: q={q} retired={} pushed={} plan={plan:?}",
                                s.retired(),
                                s.pushed()
                            );
                            assert_eq!(
                                seg_start(q),
                                seg_start(next_out),
                                "durum {case}: pencere sahne geçişini aştı"
                            );
                        }
                        emit_ctx.clear();
                        for p in plan.emit.0..plan.emit.1 {
                            let seq = plan
                                .seq_at(p as usize)
                                .expect("kesinleşen konum geçerli olmalı");
                            assert_eq!(seq, next_out + (p - plan.emit.0) as u64);
                            let want = if plan.mode == Mode::A {
                                Hint::Precision
                            } else {
                                Hint::Normal
                            };
                            assert_eq!(
                                hints[seq as usize], want,
                                "durum {case}: kare {seq} yanlış modda"
                            );
                            emit_ctx.insert(
                                seq,
                                (
                                    plan.past_context(p as usize),
                                    plan.future_context(p as usize),
                                ),
                            );
                        }
                        let mut new_cache = HashMap::new();
                        for p in plan.cache.0..plan.cache.1 {
                            let seq = plan
                                .seq_at(p as usize)
                                .expect("önbellek konumu geçerli olmalı");
                            new_cache.insert(
                                seq,
                                (
                                    plan.past_context(p as usize),
                                    plan.future_context(p as usize),
                                ),
                            );
                        }
                        if s.complete(plan.id).unwrap() {
                            old_cache_ctx = std::mem::replace(&mut cache_ctx, new_cache);
                        }
                    }
                    Step::Output(n) => {
                        for _ in 0..n {
                            let item = s.pop_output().unwrap();
                            assert_eq!(item.seq, next_out, "durum {case}: sıra bozuldu");
                            let i = item.seq as usize;
                            match item.source {
                                OutputSource::Original => assert_eq!(hints[i], Hint::Bypass),
                                OutputSource::Estimate { fuse_index, .. } => {
                                    let (p, f) = emit_ctx[&item.seq];
                                    past[i] = p;
                                    future[i] = f;
                                    if fuse_index.is_some() {
                                        let (cp, cf) = old_cache_ctx[&item.seq];
                                        past[i] = past[i].max(cp);
                                        future[i] = future[i].max(cf);
                                    }
                                }
                            }
                            next_out += 1;
                        }
                    }
                    Step::Finished => break,
                }
                assert!(
                    s.pushed() - s.retired() <= RING_FRAMES as u64,
                    "durum {case}: halka taştı"
                );
            }
            assert_eq!(next_out, total, "durum {case}: her kare çıkmalı");
            for seq in 0..total {
                let i = seq as usize;
                if hints[i] == Hint::Bypass {
                    continue;
                }
                let reach = if hints[i] == Hint::Precision {
                    MODE_A_TARGET as u64
                } else {
                    OVERLAP as u64
                };
                let need_past = (seq - seg_start(seq)).min(reach) as u32;
                let need_future = (seg_end(seq) - 1 - seq).min(reach) as u32;
                assert!(
                    past[i] >= need_past,
                    "durum {case}: kare {seq} geçmiş {} < {need_past}",
                    past[i]
                );
                assert!(
                    future[i] >= need_future,
                    "durum {case}: kare {seq} gelecek {} < {need_future}",
                    future[i]
                );
            }
        }
    }

    // ---- hat -----------------------------------------------------------------

    fn run_pipeline<R: Restorer>(
        geo: FrameGeometry,
        total: u64,
        restorer: &mut R,
        setup: impl FnOnce(&mut Pipeline),
    ) -> (Vec<(OutputMeta, Vec<u8>)>, PipelineStats) {
        let mut pipe = Pipeline::new(geo).unwrap();
        pipe.set_auto_scene_cut(false);
        pipe.set_integrity_check(true);
        setup(&mut pipe);
        let mut produced = 0u64;
        let mut outputs = Vec::new();
        pipe.run(
            |dst| {
                if produced == total {
                    return None;
                }
                dst.copy_from_slice(&frame_content(geo, produced));
                produced += 1;
                Some(((produced - 1) as i64 * 16_667 + 5, 0))
            },
            restorer,
            |frame, meta| outputs.push((*meta, frame.to_vec())),
        )
        .unwrap();
        (outputs, pipe.stats())
    }

    #[test]
    fn pipeline_identity_is_bit_exact_ordered_and_keeps_pts() {
        let geo = small();
        for total in [0u64, 1, 14, 15, 16, 29, 30, 31, 44, 45, 46, 100, 181] {
            let (outputs, stats) = run_pipeline(geo, total, &mut IdentityRestorer, |_| {});
            assert_eq!(outputs.len() as u64, total);
            for (i, (meta, frame)) in outputs.iter().enumerate() {
                assert_eq!(meta.seq, i as u64);
                assert_eq!(meta.pts_us, i as i64 * 16_667 + 5);
                assert_eq!(meta.modified_pixels, 0);
                assert_eq!(
                    frame,
                    &frame_content(geo, i as u64),
                    "kare {i} (toplam {total}) bit düzeyinde aynı olmalı"
                );
            }
            assert_eq!((stats.frames_in, stats.frames_out), (total, total));
            assert_eq!(stats.order_violations + stats.preservation_violations, 0);
            assert!(stats.max_occupancy <= RING_FRAMES as u32);
        }
    }

    /// Her konum için sabit bir bölgeyi konuma bağlı renkle boyar.
    struct PaintRestorer {
        roi: Rect,
        weight: fn(usize) -> u8,
    }

    impl PaintRestorer {
        fn value(pos: usize) -> (u8, u8, u8) {
            (100 + pos as u8, 40 + pos as u8, 220 - pos as u8)
        }
    }

    impl Restorer for PaintRestorer {
        fn restore(&mut self, io: &mut WindowIo<'_>) -> Result<(), CoreError> {
            for pos in 0..WINDOW_FRAMES {
                let valid = io.plan().is_valid(pos);
                if let Some(est) = io.estimate_mut(pos) {
                    assert!(valid, "dolgu konumu için tahmin istenmemeli");
                    let (y, u, v) = Self::value(pos);
                    est.paint(self.roi, y, u, v, (self.weight)(pos))?;
                }
            }
            Ok(())
        }
    }

    fn blend(a: u8, wa: u32, b: u8, wb: u32) -> u8 {
        ((a as u32 * wa + b as u32 * wb + (wa + wb) / 2) / (wa + wb)) as u8
    }

    #[test]
    fn pipeline_fuses_overlap_with_triangular_weights_and_preserves_unmasked() {
        let geo = small();
        // 48 piksel genişlik: her satırda 32 baytlık bir assembly bloğu + 16 baytlık kuyruk.
        let roi = Rect::new(8, 4, 48, 8);
        let total = 75u64;
        for weight in [(|_| 255u8) as fn(usize) -> u8, |pos| 40 + 7 * pos as u8] {
            let mut restorer = PaintRestorer { roi, weight };
            let (outputs, stats) = run_pipeline(geo, total, &mut restorer, |_| {});
            assert_eq!(outputs.len() as u64, total);
            assert_eq!(stats.preservation_violations, 0);
            assert_eq!(stats.modified_pixels, total * (roi.w * roi.h) as u64);
            for (meta, frame) in &outputs {
                let original = frame_content(geo, meta.seq);
                // 75 kare: pencereler [0,15) → [11,26) → … → [55,70) (j = 11), son pencere [64,79)
                // (dolgulu, 66..74'ü kesinleştirir). Kare, onu kesinleştiren pencerenin `pos`
                // konumundan; örtüşen 4 karede ayrıca önceki pencerenin `pos+11` konumundan gelir.
                let seq = meta.seq;
                let (base, k) = if seq < 66 {
                    (seq / 11 * 11, (seq % 11) as usize)
                } else {
                    (64, (seq - 64) as usize)
                };
                let cached = (seq >= 11 && (seq - 11) % 11 < 4).then(|| {
                    let prev = if seq < 66 { base - 11 } else { 55 };
                    (prev, (seq - prev) as usize)
                });
                // Öncelik = 1 + min(geçerli geçmiş, geçerli gelecek): örtüşmede (k+1) : (4-k) doğrusal
                // geçiş; akışın sonundaki dolgulu pencerede gelecek kısaldığı için öncelik de küçülür.
                let prior = |window_base: u64, pos: usize| -> u32 {
                    let last = (total - 1 - window_base).min(14) as usize;
                    1 + pos.min(last - pos) as u32
                };
                let expect = |plane: usize| -> u8 {
                    let v = PaintRestorer::value(k);
                    let cur = [v.0, v.1, v.2][plane];
                    let Some((prev, old_pos)) = cached else {
                        return cur;
                    };
                    let o = PaintRestorer::value(old_pos);
                    let old = [o.0, o.1, o.2][plane];
                    // Y pikseli başına ağırlık w; UV bloğu başına 4w — oran aynıdır.
                    blend(
                        cur,
                        prior(base, k) * weight(k) as u32,
                        old,
                        prior(prev, old_pos) * weight(old_pos) as u32,
                    )
                };
                for y in 0..geo.height() {
                    for x in 0..geo.width() {
                        let i = y * geo.width() + x;
                        let inside =
                            x >= roi.x && x < roi.x + roi.w && y >= roi.y && y < roi.y + roi.h;
                        let want = if inside { expect(0) } else { original[i] };
                        assert_eq!(frame[i], want, "Y kare {} ({x},{y})", meta.seq);
                    }
                }
                for cy in 0..geo.height() / 2 {
                    for cx in 0..geo.width() / 2 {
                        let i = geo.luma_bytes() + cy * geo.width() + 2 * cx;
                        let inside = 2 * cx >= roi.x
                            && 2 * cx < roi.x + roi.w
                            && 2 * cy >= roi.y
                            && 2 * cy < roi.y + roi.h;
                        let want = if inside {
                            (expect(1), expect(2))
                        } else {
                            (original[i], original[i + 1])
                        };
                        assert_eq!(
                            (frame[i], frame[i + 1]),
                            want,
                            "UV kare {} ({cx},{cy})",
                            meta.seq
                        );
                    }
                }
                assert_eq!(meta.flags & OUT_FLAG_FUSED != 0, cached.is_some());
            }
        }
    }

    #[test]
    fn pipeline_mode_a_bypass_and_odd_masks() {
        let geo = small();
        // Tek sayılı sınırlı ROI çift sınırlara genişler: (9,5,3,3) → (8,4,4,4).
        struct OddMask;
        impl Restorer for OddMask {
            fn restore(&mut self, io: &mut WindowIo<'_>) -> Result<(), CoreError> {
                for pos in 0..WINDOW_FRAMES {
                    if let Some(est) = io.estimate_mut(pos) {
                        let r = est.begin(Rect::new(9, 5, 3, 3))?;
                        assert_eq!(r, Rect::new(8, 4, 4, 4));
                        // Yalnızca tek bir piksel maskeli: (9,5).
                        for row in r.y..r.y + r.h {
                            let (ys, ws) = est.luma_rows_mut(row);
                            ys.fill(200);
                            ws.fill(0);
                            if row == 5 {
                                ws[1] = 255;
                            }
                        }
                        for cy in r.y / 2..(r.y + r.h) / 2 {
                            est.uv_row_mut(cy).fill(77);
                        }
                    }
                }
                Ok(())
            }
        }
        let hints = |seq: u64| match seq {
            20..=24 => Hint::Precision,
            50..=69 => Hint::Bypass,
            _ => Hint::Normal,
        };
        let total = 90u64;
        let mut pipe = Pipeline::new(geo).unwrap();
        pipe.set_auto_scene_cut(false);
        pipe.set_integrity_check(true);
        let mut outputs = Vec::new();
        let mut out = vec![0u8; geo.frame_bytes()];
        let mut produced = 0u64;
        loop {
            match pipe.poll() {
                Step::NeedFrames(n) => {
                    for _ in 0..n {
                        if produced == total {
                            pipe.finish();
                            break;
                        }
                        let meta = pipe
                            .push_frame(&frame_content(geo, produced), produced as i64 * 1000, 0)
                            .unwrap();
                        pipe.set_hint(meta.seq, hints(meta.seq)).unwrap();
                        produced += 1;
                    }
                }
                Step::Window(plan) => {
                    let mut io = pipe.window_io().unwrap();
                    if plan.mode == Mode::A {
                        assert!(io.frame(MODE_A_TARGET).is_some());
                        assert!(io.estimate_mut(0).is_none() && io.estimate_mut(15).is_none());
                    }
                    OddMask.restore(&mut io).unwrap();
                    pipe.complete_window().unwrap();
                }
                Step::Output(_) => {
                    while let Some(meta) = pipe.pop_output(&mut out).unwrap() {
                        outputs.push((meta, out.clone()));
                    }
                }
                Step::Finished => break,
            }
        }
        let stats = pipe.stats();
        assert_eq!(outputs.len() as u64, total);
        assert_eq!(stats.preservation_violations, 0);
        assert_eq!((stats.sched.windows_a, stats.sched.bypassed), (5, 20));
        for (meta, frame) in &outputs {
            let original = frame_content(geo, meta.seq);
            let w = geo.width();
            let bypass = (50..70).contains(&meta.seq);
            assert_eq!(meta.flags & OUT_FLAG_BYPASS != 0, bypass);
            assert_eq!(
                meta.flags & OUT_FLAG_MODE_A != 0,
                (20..25).contains(&meta.seq)
            );
            assert_eq!(meta.modified_pixels, if bypass { 0 } else { 1 });
            let mut expected = original.clone();
            if !bypass {
                expected[5 * w + 9] = 200; // tek maskeli Y pikseli
                let uv = geo.luma_bytes() + 2 * w + 8; // onu içeren 2×2 bloğun rengi
                expected[uv] = 77;
                expected[uv + 1] = 77;
            }
            assert_eq!(frame, &expected, "kare {}", meta.seq);
        }
    }

    #[test]
    fn pipeline_detects_scene_cut_and_reports_errors() {
        let geo = FrameGeometry::new(320, 180).unwrap();
        let mut pipe = Pipeline::new(geo).unwrap();
        let mut frame = vec![128u8; geo.frame_bytes()];
        let mut out = vec![0u8; geo.frame_bytes()];
        assert_eq!(pipe.pop_output(&mut out), Ok(None));
        assert_eq!(pipe.window_io().err(), Some(CoreError::BadState));
        assert_eq!(pipe.complete_window(), Err(CoreError::BadState));
        assert_eq!(
            pipe.push_frame(&frame[..10], 0, 0).err(),
            Some(CoreError::BufferTooSmall)
        );

        let mut starts = Vec::new();
        let mut pushed = 0u64;
        loop {
            match pipe.poll() {
                Step::NeedFrames(n) => {
                    for _ in 0..n {
                        if pushed == 50 {
                            pipe.finish();
                            break;
                        }
                        frame[..geo.luma_bytes()].copy_from_slice(&scene_luma(
                            geo,
                            pushed / 25,
                            (pushed % 25) as usize,
                            pushed,
                        ));
                        pipe.push_frame(&frame, pushed as i64, 0).unwrap();
                        pushed += 1;
                    }
                }
                Step::Window(plan) => {
                    let first = plan.seq_at(plan.valid.trailing_zeros() as usize).unwrap();
                    let last = plan
                        .seq_at(31 - plan.valid.leading_zeros() as usize)
                        .unwrap();
                    assert_eq!(
                        first / 25,
                        last / 25,
                        "pencere algılanan sahne geçişini aşmamalı"
                    );
                    pipe.complete_window().unwrap();
                }
                Step::Output(_) => {
                    while let Some(meta) = pipe.pop_output(&mut out).unwrap() {
                        if meta.flags & OUT_FLAG_SCENE_START != 0 {
                            starts.push(meta.seq);
                        }
                    }
                }
                Step::Finished => break,
            }
        }
        assert_eq!(starts, vec![0, 25]);
        assert_eq!(pipe.stats().scene_cuts, 1);
        assert_eq!(
            pipe.push_frame(&frame, 1_000_000, 0).err(),
            Some(CoreError::StreamFinished)
        );
    }

    #[test]
    fn pipeline_preserves_variable_frame_timing() {
        // Değişken kare hızı: düzensiz aralıklar, negatif başlangıç ve çok büyük damgalar.
        // Çekirdek damgaları yorumlamaz; yalnızca kesin artan olmalarını ister ve aynen taşır.
        let geo = small();
        let mut rng = Rng::new(77);
        let mut pts = Vec::new();
        let mut t = -5_000_000i64;
        for i in 0..97u64 {
            t += match i % 7 {
                0 => 1,             // art arda en küçük adım
                1 => 41_708,        // 23,976 kare/sn
                2 => 16_667,        // 60 kare/sn
                3 => 3_600_000_000, // bir saatlik boşluk
                _ => 1 + rng.below(100_000) as i64,
            };
            pts.push(t);
        }
        let mut pipe = Pipeline::new(geo).unwrap();
        pipe.set_auto_scene_cut(false);
        let mut produced = 0usize;
        let mut seen = Vec::new();
        pipe.run(
            |dst| {
                let i = produced;
                if i == pts.len() {
                    return None;
                }
                dst.copy_from_slice(&frame_content(geo, i as u64));
                produced += 1;
                Some((pts[i], 0))
            },
            &mut IdentityRestorer,
            |frame, meta| {
                assert_eq!(frame, &frame_content(geo, meta.seq)[..]);
                seen.push(meta.pts_us);
            },
        )
        .unwrap();
        assert_eq!(seen, pts, "zaman damgaları sırasıyla ve değişmeden çıkmalı");

        // Geri giden veya yinelenen damga kareyi reddeder, akışı bozmaz.
        let mut pipe = Pipeline::new(geo).unwrap();
        let frame = frame_content(geo, 0);
        pipe.push_frame(&frame, 1000, 0).unwrap();
        assert_eq!(
            pipe.push_frame(&frame, 1000, 0).err(),
            Some(CoreError::NonMonotonicPts)
        );
        assert_eq!(
            pipe.push_frame(&frame, 999, 0).err(),
            Some(CoreError::NonMonotonicPts)
        );
        assert_eq!(pipe.push_frame(&frame, 1001, 0).unwrap().seq, 1);
        assert_eq!(pipe.stats().frames_in, 2);
    }

    #[test]
    fn estimate_roi_is_clipped_validated_and_cleared() {
        let geo = small(); // 64×48
        let mut est = Estimate::new(geo);
        // Kare dışı veya boş bölge reddedilir.
        assert_eq!(
            est.begin(Rect::new(64, 0, 4, 4)).err(),
            Some(CoreError::BadArgument)
        );
        assert_eq!(
            est.begin(Rect::new(0, 48, 4, 4)).err(),
            Some(CoreError::BadArgument)
        );
        assert_eq!(
            est.begin(Rect::new(10, 10, 0, 4)).err(),
            Some(CoreError::BadArgument)
        );
        assert!(est.roi().is_empty());
        // Kareden taşan bölge kırpılır; tek sınırlar çifte genişler.
        assert_eq!(
            est.begin(Rect::new(59, 43, 100, 100)).unwrap(),
            Rect::new(58, 42, 6, 6)
        );
        assert_eq!(
            est.begin(Rect::new(0, 0, usize::MAX, usize::MAX)).unwrap(),
            Rect::new(0, 0, 64, 48)
        );
        // Yeni bölgeye geçince eski bölgenin ağırlıkları sıfırlanır (bayat maske kalmaz).
        est.paint(Rect::new(0, 0, 64, 48), 9, 9, 9, 255).unwrap();
        est.paint(Rect::new(8, 8, 4, 4), 1, 2, 3, 200).unwrap();
        let mut frame = frame_content(geo, 5);
        let original = frame.clone();
        assert_eq!(compose(geo, &mut frame, &est, None), 16);
        assert_eq!(verify_preservation(geo, &original, &frame, &est, None), 0);
        let changed = (0..geo.luma_bytes())
            .filter(|&i| frame[i] != original[i])
            .count();
        assert!(
            changed <= 16,
            "yalnızca 4×4 bölge değişebilir, değişen: {changed}"
        );
        // Kapı denetimi bozulmayı gerçekten yakalıyor: maske dışı tek bir bayt değişirse sayılır.
        frame[0] ^= 0xFF;
        *frame.last_mut().unwrap() ^= 0xFF;
        assert_eq!(verify_preservation(geo, &original, &frame, &est, None), 2);
        est.clear();
        let mut untouched = original.clone();
        assert_eq!(compose(geo, &mut untouched, &est, None), 0);
        assert_eq!(untouched, original);
    }

    #[test]
    #[should_panic(expected = "satır ROI dışında")]
    fn estimate_rejects_writes_outside_roi() {
        let mut est = Estimate::new(small());
        est.begin(Rect::new(8, 8, 4, 4)).unwrap();
        let _ = est.luma_rows_mut(12);
    }

    // ---- düzlem dönüşümleri --------------------------------------------------

    #[test]
    fn yuv_planes_pack_and_unpack_for_i420_nv12_nv21() {
        let geo = FrameGeometry::new(32, 16).unwrap();
        let (w, h) = (geo.width(), geo.height());
        let nv12 = frame_content(geo, 3);
        let (y_stride, uv_stride) = (40usize, 44usize);
        let mut y_buf = vec![0xEEu8; y_stride * h];
        for row in 0..h {
            y_buf[row * y_stride..row * y_stride + w]
                .copy_from_slice(&nv12[row * w..(row + 1) * w]);
        }
        let y_plane = |buf: &mut Vec<u8>| RawPlane {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
            row_stride: y_stride,
            pixel_stride: 1,
        };
        let uv_src = &nv12[geo.luma_bytes()..];

        // I420: ayrı U ve V düzlemleri, piksel adımı 1.
        let (mut u_buf, mut v_buf) = (vec![0u8; uv_stride * h / 2], vec![0u8; uv_stride * h / 2]);
        for cy in 0..h / 2 {
            for cx in 0..w / 2 {
                u_buf[cy * uv_stride + cx] = uv_src[cy * w + 2 * cx];
                v_buf[cy * uv_stride + cx] = uv_src[cy * w + 2 * cx + 1];
            }
        }
        let planar = |b: &mut Vec<u8>| RawPlane {
            ptr: b.as_mut_ptr(),
            len: b.len(),
            row_stride: uv_stride,
            pixel_stride: 1,
        };
        let mut packed = vec![0u8; geo.frame_bytes()];
        unsafe {
            pack_nv12(
                geo,
                y_plane(&mut y_buf),
                planar(&mut u_buf),
                planar(&mut v_buf),
                &mut packed,
            )
            .unwrap()
        };
        assert_eq!(packed, nv12);
        let (mut y2, mut u2, mut v2) = (
            vec![0u8; y_stride * h],
            vec![0u8; uv_stride * h / 2],
            vec![0u8; uv_stride * h / 2],
        );
        unsafe {
            unpack_nv12(
                geo,
                &nv12,
                y_plane(&mut y2),
                planar(&mut u2),
                planar(&mut v2),
            )
            .unwrap()
        };
        let mut again = vec![0u8; geo.frame_bytes()];
        unsafe {
            pack_nv12(
                geo,
                y_plane(&mut y2),
                planar(&mut u2),
                planar(&mut v2),
                &mut again,
            )
            .unwrap()
        };
        assert_eq!(again, nv12);

        // Yarı düzlemsel (piksel adımı 2): U ve V aynı belleğin 1 bayt kaydırılmış görünümleri.
        // MediaCodec'te olduğu gibi her düzlem son örneğinde biter (uzunluk = adım*(satır-1) + w - 1).
        for swap in [false, true] {
            let len = uv_stride * (h / 2 - 1) + w;
            let mut inter = vec![0u8; len];
            for cy in 0..h / 2 {
                for cx in 0..w / 2 {
                    let (u, v) = (uv_src[cy * w + 2 * cx], uv_src[cy * w + 2 * cx + 1]);
                    inter[cy * uv_stride + 2 * cx] = if swap { v } else { u };
                    inter[cy * uv_stride + 2 * cx + 1] = if swap { u } else { v };
                }
            }
            let views = |b: &mut Vec<u8>| {
                let first = RawPlane {
                    ptr: b.as_mut_ptr(),
                    len: len - 1,
                    row_stride: uv_stride,
                    pixel_stride: 2,
                };
                let second = RawPlane {
                    ptr: unsafe { b.as_mut_ptr().add(1) },
                    len: len - 1,
                    row_stride: uv_stride,
                    pixel_stride: 2,
                };
                if swap {
                    (second, first)
                } else {
                    (first, second)
                }
            };
            let (u, v) = views(&mut inter);
            let mut packed = vec![0u8; geo.frame_bytes()];
            unsafe { pack_nv12(geo, y_plane(&mut y_buf), u, v, &mut packed).unwrap() };
            assert_eq!(packed, nv12, "paketleme (swap={swap})");
            let mut target = vec![0u8; len];
            let (u, v) = views(&mut target);
            unsafe { unpack_nv12(geo, &nv12, y_plane(&mut y2), u, v).unwrap() };
            assert_eq!(target, inter, "açma (swap={swap})");
        }

        // 10 bit (P010 yerleşimi): örnekler 16 bit küçük-endian, değer üst 10 bitte. Ofset +1 ve
        // piksel adımı 2 (Y) / 4 (UV) ile üst baytlar okunur → 8 bit NV12.
        let mut y16 = vec![0u8; 2 * y_stride * h];
        for row in 0..h {
            for x in 0..w {
                y16[2 * (row * y_stride + x)] = 0xC0; // alt 2 bit: atılacak
                y16[2 * (row * y_stride + x) + 1] = nv12[row * w + x];
            }
        }
        let mut uv16 = vec![0u8; 2 * w * (h / 2)];
        for i in 0..w * (h / 2) {
            uv16[2 * i] = 0x40;
            uv16[2 * i + 1] = uv_src[i];
        }
        let y_hi = RawPlane {
            ptr: unsafe { y16.as_mut_ptr().add(1) },
            len: y16.len() - 1,
            row_stride: 2 * y_stride,
            pixel_stride: 2,
        };
        let u_hi = RawPlane {
            ptr: unsafe { uv16.as_mut_ptr().add(1) },
            len: uv16.len() - 1,
            row_stride: 2 * w,
            pixel_stride: 4,
        };
        let v_hi = RawPlane {
            ptr: unsafe { uv16.as_mut_ptr().add(3) },
            len: uv16.len() - 3,
            row_stride: 2 * w,
            pixel_stride: 4,
        };
        let mut packed = vec![0u8; geo.frame_bytes()];
        unsafe { pack_nv12(geo, y_hi, u_hi, v_hi, &mut packed).unwrap() };
        assert_eq!(packed, nv12, "P010 üst bayt paketleme");
        // Çıkış (kodlayıcı) tarafında Y adımı 1 olmak zorundadır.
        let result = unsafe { unpack_nv12(geo, &nv12, y_hi, u_hi, v_hi) };
        assert_eq!(result, Err(CoreError::BufferTooSmall));

        // Çok kısa düzlem reddedilir.
        let short = RawPlane {
            ptr: u_buf.as_mut_ptr(),
            len: 10,
            row_stride: uv_stride,
            pixel_stride: 1,
        };
        let result = unsafe {
            pack_nv12(
                geo,
                y_plane(&mut y_buf),
                short,
                planar(&mut v_buf),
                &mut packed,
            )
        };
        assert_eq!(result, Err(CoreError::BufferTooSmall));
    }

    // ---- JNI yüzeyi (sahte JNIEnv ile) ---------------------------------------

    /// Testte "doğrudan ByteBuffer" yerine geçen nesne.
    #[repr(C)]
    struct FakeBuffer {
        ptr: *mut u8,
        cap: i64,
    }

    unsafe extern "system" fn fake_address(_env: *mut JniEnv, buf: JObject) -> *mut c_void {
        unsafe { (*(buf as *const FakeBuffer)).ptr as *mut c_void }
    }

    unsafe extern "system" fn fake_capacity(_env: *mut JniEnv, buf: JObject) -> i64 {
        unsafe { (*(buf as *const FakeBuffer)).cap }
    }

    thread_local! {
        static LAST_STRING: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    }

    unsafe extern "system" fn fake_new_string_utf(
        _env: *mut JniEnv,
        s: *const std::ffi::c_char,
    ) -> JObject {
        let text = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        LAST_STRING.with(|cell| *cell.borrow_mut() = text);
        1usize as JObject
    }

    struct FakeEnv {
        _table: Box<[*const c_void; 232]>,
        env: Box<JniEnv>,
    }

    impl FakeEnv {
        fn new() -> Self {
            let mut table = Box::new([std::ptr::null::<c_void>(); 232]);
            table[230] = fake_address as *const c_void;
            table[231] = fake_capacity as *const c_void;
            table[167] = fake_new_string_utf as *const c_void;
            let env = Box::new(table.as_ptr() as JniEnv);
            Self { _table: table, env }
        }
        fn ptr(&mut self) -> *mut JniEnv {
            &mut *self.env
        }
    }

    fn fake(buf: &mut [u8]) -> FakeBuffer {
        FakeBuffer {
            ptr: buf.as_mut_ptr(),
            cap: buf.len() as i64,
        }
    }

    fn obj(b: &mut FakeBuffer) -> JObject {
        b as *mut FakeBuffer as JObject
    }

    fn read_i32(buf: &[u8], at: usize) -> i32 {
        i32::from_ne_bytes(buf[at..at + 4].try_into().unwrap())
    }

    fn read_i64(buf: &[u8], at: usize) -> i64 {
        i64::from_ne_bytes(buf[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn jni_surface_roundtrip_with_fake_env() {
        let mut fake_env = FakeEnv::new();
        let env = fake_env.ptr();
        let null = std::ptr::null_mut();
        let geo = small();
        let (w, h) = (geo.width(), geo.height());
        let total = 47u64;

        assert_eq!(
            Java_com_video_cleaner_Core_nativeAbiVersion(env, null),
            ABI_VERSION
        );
        assert_eq!(
            Java_com_video_cleaner_Core_nativePageSize(env, null) as usize,
            page_size()
        );
        assert_eq!(
            Java_com_video_cleaner_Core_nativeCreate(env, null, 63, 48),
            CoreError::InvalidGeometry.code() as i64
        );
        let handle = Java_com_video_cleaner_Core_nativeCreate(env, null, w as i32, h as i32);
        assert!(handle > 0);
        assert_eq!(
            Java_com_video_cleaner_Core_nativeSetOption(env, null, handle, 1, 0),
            0
        );
        assert_eq!(
            Java_com_video_cleaner_Core_nativeSetOption(env, null, handle, 2, 1),
            0
        );
        assert_eq!(
            Java_com_video_cleaner_Core_nativeSetOption(env, null, handle, 9, 1),
            CoreError::BadArgument.code()
        );
        assert_eq!(
            Java_com_video_cleaner_Core_nativeFinish(env, null, 0),
            CoreError::InvalidHandle.code()
        );

        let mut info = vec![0u8; POLL_INFO_BYTES];
        let mut meta = vec![0u8; OUTPUT_META_BYTES];
        let mut small_info = vec![0u8; 8];
        let code = unsafe {
            Java_com_video_cleaner_Core_nativePoll(
                env,
                null,
                handle,
                obj(&mut fake(&mut small_info)),
            )
        };
        assert_eq!(code, CoreError::BufferTooSmall.code());
        // Null tampon (Kotlin tarafında yanlış tür / null aktarımı) çökme yerine hata kodu döndürür.
        let code = unsafe { Java_com_video_cleaner_Core_nativePoll(env, null, handle, null) };
        assert_eq!(code, CoreError::BadArgument.code());
        let mut scratch = vec![0u8; geo.frame_bytes()];
        let code = unsafe {
            Java_com_video_cleaner_Core_nativePushYuv420(
                env,
                null,
                handle,
                obj(&mut fake(&mut scratch)),
                0,
                w as i32,
                1,
                null,
                0,
                null,
                0,
                (w / 2) as i32,
                1,
                0,
                0,
            )
        };
        assert_eq!(code, CoreError::BadArgument.code() as i64);
        // Negatif ofset, sıfır adım ve tampondan büyük ofset de reddedilir.
        for (offset, stride) in [(-1, w as i32), (0, 0), (i32::MAX, w as i32)] {
            let code = unsafe {
                Java_com_video_cleaner_Core_nativePushYuv420(
                    env,
                    null,
                    handle,
                    obj(&mut fake(&mut scratch)),
                    offset,
                    stride,
                    1,
                    obj(&mut fake(&mut scratch)),
                    0,
                    obj(&mut fake(&mut scratch)),
                    0,
                    (w / 2) as i32,
                    1,
                    0,
                    0,
                )
            };
            assert_eq!(
                code,
                CoreError::BadArgument.code() as i64,
                "ofset {offset}, adım {stride}"
            );
        }
        // Çok küçük düzlem: kare eklenmez.
        let mut tiny = vec![0u8; 16];
        let code = unsafe {
            Java_com_video_cleaner_Core_nativePushYuv420(
                env,
                null,
                handle,
                obj(&mut fake(&mut scratch)),
                0,
                w as i32,
                1,
                obj(&mut fake(&mut tiny)),
                0,
                obj(&mut fake(&mut tiny)),
                0,
                (w / 2) as i32,
                1,
                0,
                0,
            )
        };
        assert_eq!(code, CoreError::BufferTooSmall.code() as i64);

        // Giriş: I420 düzlemleri (satır adımı genişlikten büyük). Çıkış: NV12 görünümlü düzlemler.
        let (ys, cs) = (w + 8, w / 2 + 4);
        let (mut y_in, mut u_in, mut v_in) = (
            vec![0u8; ys * h],
            vec![0u8; cs * h / 2],
            vec![0u8; cs * h / 2],
        );
        let mut y_out = vec![0u8; ys * h];
        let mut uv_out = vec![0u8; w * h / 2];
        let mut roi_y = vec![150u8; 8 * 4];
        let mut roi_uv = vec![90u8; 8 * 2];
        let mut roi_w = vec![255u8; 8 * 4];
        let mut window_frame = vec![0u8; geo.frame_bytes()];
        let (mut pushed, mut popped) = (0u64, 0u64);
        let mut windows = 0;
        loop {
            let step = unsafe {
                Java_com_video_cleaner_Core_nativePoll(env, null, handle, obj(&mut fake(&mut info)))
            };
            assert_eq!(step, read_i32(&info, 0));
            match step {
                STEP_NEED_FRAMES => {
                    for _ in 0..read_i32(&info, 4) {
                        if pushed == total {
                            assert_eq!(
                                Java_com_video_cleaner_Core_nativeFinish(env, null, handle),
                                0
                            );
                            break;
                        }
                        let nv12 = frame_content(geo, pushed);
                        for row in 0..h {
                            y_in[row * ys..row * ys + w]
                                .copy_from_slice(&nv12[row * w..(row + 1) * w]);
                        }
                        for cy in 0..h / 2 {
                            for cx in 0..w / 2 {
                                u_in[cy * cs + cx] = nv12[geo.luma_bytes() + cy * w + 2 * cx];
                                v_in[cy * cs + cx] = nv12[geo.luma_bytes() + cy * w + 2 * cx + 1];
                            }
                        }
                        let seq = unsafe {
                            Java_com_video_cleaner_Core_nativePushYuv420(
                                env,
                                null,
                                handle,
                                obj(&mut fake(&mut y_in)),
                                0,
                                ys as i32,
                                1,
                                obj(&mut fake(&mut u_in)),
                                0,
                                obj(&mut fake(&mut v_in)),
                                0,
                                cs as i32,
                                1,
                                pushed as i64 * 33_333,
                                0,
                            )
                        };
                        assert_eq!(seq, pushed as i64);
                        pushed += 1;
                    }
                }
                STEP_WINDOW => {
                    windows += 1;
                    let (emit0, emit1) = (read_i32(&info, 28), read_i32(&info, 32));
                    let valid = read_i32(&info, 56) as u32;
                    let base = read_i64(&info, 48);
                    // Pencere karesi okunabilir ve doğru kareyi verir; dolgu konumu 0 döner.
                    for pos in [0, 7, 14] {
                        let got = unsafe {
                            Java_com_video_cleaner_Core_nativeReadWindowFrame(
                                env,
                                null,
                                handle,
                                pos,
                                obj(&mut fake(&mut window_frame)),
                            )
                        };
                        if valid & (1 << pos) != 0 {
                            assert_eq!(got, 1);
                            assert_eq!(
                                window_frame,
                                frame_content(geo, (base + pos as i64) as u64)
                            );
                        } else {
                            assert_eq!(got, 0);
                        }
                    }
                    // Kesinleşen her konuma (4,2)-(12,6) bölgesi için tahmin ver.
                    for pos in emit0..emit1 {
                        let rc = unsafe {
                            Java_com_video_cleaner_Core_nativeSubmitEstimate(
                                env,
                                null,
                                handle,
                                pos,
                                4,
                                2,
                                8,
                                4,
                                obj(&mut fake(&mut roi_y)),
                                obj(&mut fake(&mut roi_uv)),
                                obj(&mut fake(&mut roi_w)),
                            )
                        };
                        assert_eq!(rc, 0);
                    }
                    // Hizasız ROI ve çıktı istenmeyen konum reddedilir.
                    let bad = unsafe {
                        Java_com_video_cleaner_Core_nativeSubmitEstimate(
                            env,
                            null,
                            handle,
                            emit0.max(0),
                            5,
                            2,
                            8,
                            4,
                            obj(&mut fake(&mut roi_y)),
                            obj(&mut fake(&mut roi_uv)),
                            obj(&mut fake(&mut roi_w)),
                        )
                    };
                    assert_eq!(bad, CoreError::BadArgument.code());
                    assert_eq!(
                        Java_com_video_cleaner_Core_nativeCompleteWindow(env, null, handle),
                        0
                    );
                }
                STEP_OUTPUT => loop {
                    let got = unsafe {
                        Java_com_video_cleaner_Core_nativePopOutputYuv420(
                            env,
                            null,
                            handle,
                            obj(&mut fake(&mut y_out)),
                            0,
                            ys as i32,
                            obj(&mut fake(&mut uv_out[..w * h / 2 - 1])),
                            0,
                            obj(&mut fake(&mut uv_out[1..])),
                            0,
                            w as i32,
                            2,
                            obj(&mut fake(&mut meta)),
                        )
                    };
                    if got == 0 {
                        break;
                    }
                    assert_eq!(got, 1);
                    assert_eq!(read_i64(&meta, 0), popped as i64);
                    assert_eq!(read_i64(&meta, 8), popped as i64 * 33_333);
                    assert_eq!(read_i64(&meta, 24), 32);
                    let original = frame_content(geo, popped);
                    for y in 0..h {
                        for x in 0..w {
                            let inside = (4..12).contains(&x) && (2..6).contains(&y);
                            assert_eq!(
                                y_out[y * ys + x],
                                if inside { 150 } else { original[y * w + x] }
                            );
                        }
                    }
                    for cy in 0..h / 2 {
                        for x in 0..w {
                            let inside = (4..12).contains(&x) && (1..3).contains(&cy);
                            assert_eq!(
                                uv_out[cy * w + x],
                                if inside {
                                    90
                                } else {
                                    original[geo.luma_bytes() + cy * w + x]
                                }
                            );
                        }
                    }
                    popped += 1;
                },
                STEP_FINISHED => break,
                other => panic!("beklenmeyen adım kodu {other}"),
            }
        }
        assert_eq!((pushed, popped), (total, total));
        assert!(windows >= 3);

        let mut stats = vec![0u8; STATS_BYTES];
        let n = unsafe {
            Java_com_video_cleaner_Core_nativeStats(env, null, handle, obj(&mut fake(&mut stats)))
        };
        assert_eq!(n, 14);
        assert_eq!(read_i64(&stats, 0), total as i64); // giren
        assert_eq!(read_i64(&stats, 8), total as i64); // çıkan
        assert_eq!(read_i64(&stats, 32), 0); // koruma ihlali
        assert_eq!(read_i64(&stats, 40), 0); // sıra ihlali

        unsafe { Java_com_video_cleaner_Core_nativeDestroy(env, null, handle) };
    }

    // ---- hız ölçümü (cargo test --release -- --ignored --nocapture bench) -----

    #[test]
    #[ignore = "hız ölçümü; --ignored ile çalıştırın"]
    fn bench_720p_core_overhead() {
        use std::time::Instant;
        let geo = FrameGeometry::HD720;
        println!(
            "arka uç: {}, sayfa: {} bayt",
            kernels::backend(),
            page_size()
        );

        let mut a = vec![0u8; geo.luma_bytes()];
        let mut b = vec![0u8; geo.luma_bytes()];
        Rng::new(1).fill(&mut a);
        Rng::new(2).fill(&mut b);
        let mask: Vec<u8> = a.iter().map(|&v| if v > 200 { 255 } else { 0 }).collect();
        let mut dst = vec![0u8; geo.luma_bytes()];
        let mut bands = HaarBands::new(geo.width(), geo.height()).unwrap();
        let time = |name: &str, f: &mut dyn FnMut()| {
            let reps = 200;
            let start = Instant::now();
            for _ in 0..reps {
                f();
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            println!("  {name:<28} {us:>9.1} µs / 720p Y düzlemi");
        };
        for portable in [false, true] {
            kernels::force_portable(portable);
            println!("[{}]", kernels::backend());
            time("sad_u8", &mut || {
                std::hint::black_box(kernels::sad_u8(&a, &b));
            });
            time("select_u8", &mut || {
                kernels::select_u8(&mut dst, &a, &b, &mask)
            });
            time("haar ileri", &mut || bands.forward(&a, geo.width()));
            time("haar ters", &mut || bands.inverse(&mut dst, geo.width()));
        }
        kernels::force_portable(false);

        let frames = 600u64;
        let source_frame = frame_content(geo, 1);
        let mut pipe = Pipeline::new(geo).unwrap();
        let mut produced = 0u64;
        let start = Instant::now();
        pipe.run(
            |dst| {
                if produced == frames {
                    return None;
                }
                dst.copy_from_slice(&source_frame);
                produced += 1;
                Some((produced as i64, 0))
            },
            &mut IdentityRestorer,
            |frame, _| {
                std::hint::black_box(frame[0]);
            },
        )
        .unwrap();
        let secs = start.elapsed().as_secs_f64();
        let stats = pipe.stats();
        assert_eq!(stats.frames_out, frames);
        println!(
            "hat (model yok): {frames} kare {:.3} s → {:.0} kare/sn, {:.3} ms/kare; pencere: {}, halka: {:.2} MiB",
            secs,
            frames as f64 / secs,
            secs * 1e3 / frames as f64,
            stats.sched.windows_b,
            pipe.ring().pool_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    // ---- VC-Core motoru (vc* arayüzü) ------------------------------------------------

    /// Motoru sonuna kadar sürer: her adımda poll → pencere/çıktı; kimlik onarıcıyla.
    fn drive_engine(
        engine: &mut Engine,
        frames: &[(Vec<u8>, i64)],
        geo: FrameGeometry,
    ) -> Vec<Vec<u8>> {
        let mut next = 0usize;
        let mut finished = false;
        let mut out = Vec::new();
        let mut dst = vec![0u8; geo.frame_bytes()];
        loop {
            match engine.poll() {
                Step::NeedFrames(n) => {
                    if next == frames.len() {
                        assert!(!finished, "bitişten sonra kare istendi");
                        engine.finish();
                        finished = true;
                        continue;
                    }
                    for _ in 0..n.min(frames.len() - next) {
                        let (f, pts) = &frames[next];
                        engine
                            .push_nv12(geo.width(), geo.height(), f, *pts, 0)
                            .unwrap();
                        next += 1;
                    }
                }
                Step::Window(_) => {
                    assert!(engine.process_window(false, &mut IdentityRestorer).unwrap() > 0);
                }
                Step::Output(_) => {
                    while engine.pop_output(&mut dst).unwrap().is_some() {
                        out.push(dst.clone());
                    }
                }
                Step::Finished => break,
            }
        }
        out
    }

    fn engine_frames(geo: FrameGeometry, count: usize, cut_at: usize) -> Vec<(Vec<u8>, i64)> {
        (0..count)
            .map(|i| {
                let content = frame_content(
                    geo,
                    if i >= cut_at {
                        1000 + i as u64
                    } else {
                        (i / 7) as u64
                    },
                );
                (content, i as i64 * 33_333)
            })
            .collect()
    }

    #[test]
    fn engine_identity_is_bit_exact_and_complete() {
        let geo = small();
        let frames = engine_frames(geo, 71, 40);
        let mut engine = Engine::new(EngineMode::Auto, false, StorageKind::Heap);
        let out = drive_engine(&mut engine, &frames, geo);
        assert_eq!(out.len(), frames.len());
        for (o, (f, _)) in out.iter().zip(&frames) {
            assert_eq!(
                &o[..],
                &f[..geo.frame_bytes()],
                "kimlik onarıcıda çıkış girişle bit-bit aynı olmalı"
            );
        }
        let audit = engine.audit();
        assert_eq!(audit.input_digest, audit.output_digest);
        assert_eq!(audit.frames_cpu, frames.len() as u64);
        let json = engine.tester_audit_json();
        assert!(json.contains("\"bit_exact\":true"), "{json}");
        assert!(json.contains("\"frames_equal\":true"), "{json}");
        assert!(json.contains("\"violations\":0"), "{json}");
        assert!(json.is_ascii());
        let st = engine.pipeline().unwrap().stats();
        assert!(st.sched.windows_b > 0);
        assert_eq!(st.sched.windows_a, 0);
    }

    #[test]
    fn engine_mode_a_finalises_every_frame_alone() {
        let geo = small();
        let frames = engine_frames(geo, 33, usize::MAX);
        let mut engine = Engine::new(EngineMode::A, false, StorageKind::Heap);
        let out = drive_engine(&mut engine, &frames, geo);
        assert_eq!(out.len(), frames.len());
        let st = engine.pipeline().unwrap().stats();
        assert_eq!(
            st.sched.windows_a,
            frames.len() as u64,
            "Mod A'da her kare tek hedef pencereyle kesinleşir"
        );
        assert_eq!(st.sched.windows_b, 0);
        assert!(engine.tester_audit_json().contains("\"mode\":\"A\""));
    }

    #[test]
    fn engine_rejects_non_monotonic_pts_and_geometry_change() {
        let geo = small();
        let frame = frame_content(geo, 3);
        let mut engine = Engine::new(EngineMode::Auto, true, StorageKind::Heap);
        engine
            .push_nv12(geo.width(), geo.height(), &frame, 1000, 0)
            .unwrap();
        assert_eq!(
            engine.push_nv12(geo.width(), geo.height(), &frame, 1000, 0),
            Err(CoreError::NonMonotonicPts)
        );
        assert_eq!(engine.audit().pts_rejected, 1);
        assert_eq!(
            engine.push_nv12(geo.width() + 2, geo.height(), &frame, 2000, 0),
            Err(CoreError::InvalidGeometry)
        );
        let json = engine.tester_audit_json();
        assert!(json.contains("\"pts_rejected\":1"), "{json}");
        assert!(json.contains("requested, not used"), "{json}");
    }

    #[test]
    fn elf_alignment_is_reported() {
        // Test ikilisi için de PT_LOAD hizası okunabilmeli (Linux x86_64'te ≥ 4096).
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        assert!(elf_load_alignment() >= 4096, "{}", elf_load_alignment());
    }

    #[test]
    fn jni_engine_surface_with_fake_env() {
        let mut fake_env = FakeEnv::new();
        let env = fake_env.ptr();
        let null = std::ptr::null_mut();
        let geo = small();
        assert_eq!(
            Java_com_video_cleaner_Core_vcInitEngine(env, null, 9, 0),
            CoreError::BadArgument.code() as i64
        );
        let ctx = Java_com_video_cleaner_Core_vcInitEngine(env, null, 0, 1);
        assert!(ctx > 0);
        let mut info = vec![0u8; POLL_INFO_BYTES];
        let mut info_buf = fake(&mut info);
        let mut dst = vec![0u8; geo.frame_bytes()];
        let mut meta = vec![0u8; OUTPUT_META_BYTES];
        let frames = engine_frames(geo, 20, usize::MAX);
        let mut pushed = 0;
        let mut popped = 0;
        let mut finished = false;
        loop {
            let step =
                unsafe { Java_com_video_cleaner_Core_vcPoll(env, null, ctx, obj(&mut info_buf)) };
            match step {
                STEP_NEED_FRAMES if pushed < frames.len() => {
                    let mut f = frames[pushed].0.clone();
                    let mut fb = fake(&mut f);
                    let seq = unsafe {
                        Java_com_video_cleaner_Core_vcPushNv12(
                            env,
                            null,
                            ctx,
                            geo.width() as i32,
                            geo.height() as i32,
                            obj(&mut fb),
                            frames[pushed].1,
                            0,
                        )
                    };
                    assert_eq!(seq, pushed as i32);
                    pushed += 1;
                }
                STEP_NEED_FRAMES => {
                    assert!(!finished);
                    assert_eq!(Java_com_video_cleaner_Core_vcFinish(env, null, ctx), 0);
                    finished = true;
                }
                STEP_WINDOW => {
                    assert!(Java_com_video_cleaner_Core_vcProcessWindow(env, null, ctx, 0) > 0)
                }
                STEP_OUTPUT => {
                    let (mut db, mut mb) = (fake(&mut dst), fake(&mut meta));
                    while unsafe {
                        Java_com_video_cleaner_Core_vcPopOutputNv12(
                            env,
                            null,
                            ctx,
                            obj(&mut db),
                            obj(&mut mb),
                        )
                    } == 1
                    {
                        assert_eq!(&dst[..], &frames[popped].0[..geo.frame_bytes()]);
                        assert_eq!(read_i64(&meta, 8), frames[popped].1);
                        popped += 1;
                        (db, mb) = (fake(&mut dst), fake(&mut meta));
                    }
                }
                STEP_FINISHED => break,
                other => panic!("beklenmeyen adım {other}"),
            }
        }
        assert_eq!(popped, frames.len());
        // Donanım tamponu masaüstünde yok: açık hata kodu döner.
        assert_eq!(
            unsafe { Java_com_video_cleaner_Core_vcPushHardwareFrame(env, null, ctx, null, 0, 0) },
            CoreError::BadState.code()
        );
        let jstr = unsafe { Java_com_video_cleaner_Core_vcRunTesterAudit(env, null, ctx) };
        assert!(!jstr.is_null());
        let json = LAST_STRING.with(|c| c.borrow().clone());
        assert!(
            json.starts_with("{\"abi\":")
                && json.contains("\"bit_exact\":true")
                && json.contains("\"frames_out\":20"),
            "{json}"
        );
        unsafe { Java_com_video_cleaner_Core_vcDestroy(env, null, ctx) };
    }
}

#[cfg(test)]
mod vui_tests {
    use super::vui::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // x264 / x265 ile üretilmiş gerçek SPS'ler (başlık baytları dahil, öykünme önleme baytlarıyla).
    const AVC: &[&str] = &[
        "6742c00dd90141fb0110000003001000000303c0f142a480", // Constrained Baseline, VUI var
        "6764000dacd94141fb0110000003001000000303c0f1429960", // High
        "6764000dacd94141fbff0020001b6a02020280000003008000001e078a14cb", // High, SAR 32:27, BT.709
        "6764000dacd94141fb0110000003001000000303c604000927c00124fa4c30078a14cb", // High + NAL HRD
        "67f4000d919b28283f6022000003000200000300781e28532c", // High 4:4:4
    ];
    const HEVC: &[&str] = &[
        "42010101600000030090000003000003003ca00a080f165959a4932bc05a020000030002000003003c10", // Main
        "42010102200000030090000003000003003ca00a080f1365959a4932bc05a848804820000003002000000303cc0292cdc00124f800249f10", // Main10 HDR + HRD
        "42010101600000030090000003000003003ca00a080f165959a49395e02d010000030001000003001e08", // ölçekleme listesi, B-piramit
    ];

    fn all() -> Vec<(Codec, Vec<u8>)> {
        AVC.iter()
            .map(|h| (Codec::Avc, hex(h)))
            .chain(HEVC.iter().map(|h| (Codec::Hevc, hex(h))))
            .collect()
    }

    const BT601: Colour = Colour {
        primaries: 6,
        transfer: 6,
        matrix: 6,
        full_range: false,
    };

    #[test]
    fn parses_every_fixture() {
        for (codec, sps) in all() {
            let p =
                parse_sps(codec, &sps).unwrap_or_else(|e| panic!("{codec:?} {}: {e}", sps.len()));
            assert!(p.vui_present || codec == Codec::Avc, "{p:?}");
        }
        let p = parse_sps(Codec::Avc, &hex(AVC[2])).unwrap();
        assert_eq!(p.sar, Some((32, 27)));
        assert_eq!(
            p.colour,
            Some(Colour {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: false
            })
        );
        let p = parse_sps(Codec::Hevc, &hex(HEVC[1])).unwrap();
        assert_eq!(
            p.colour,
            Some(Colour {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: false
            })
        );
    }

    #[test]
    fn same_values_leave_sps_untouched() {
        for (codec, sps) in all() {
            let p = parse_sps(codec, &sps).unwrap();
            let t = Target {
                sar: p.sar,
                colour: p.colour,
            };
            assert_eq!(rewrite_sps(codec, &sps, t).unwrap(), None, "{codec:?}");
        }
    }

    #[test]
    fn rewrite_sets_values_and_round_trips_bit_exactly() {
        for (codec, sps) in all() {
            let before = parse_sps(codec, &sps).unwrap();
            let t = Target {
                sar: Some((40, 33)),
                colour: Some(BT601),
            };
            let new = rewrite_sps(codec, &sps, t).unwrap().expect("değişmeli");
            let after = parse_sps(codec, &new).unwrap();
            assert_eq!(after.sar, Some((40, 33)), "{codec:?}");
            assert_eq!(after.colour, Some(BT601), "{codec:?}");
            assert!(after.vui_present);
            // Tablo dışı SAR: 255 + açık değerler
            let odd = rewrite_sps(
                codec,
                &sps,
                Target {
                    sar: Some((853, 854)),
                    colour: None,
                },
            )
            .unwrap()
            .unwrap();
            assert_eq!(parse_sps(codec, &odd).unwrap().sar, Some((853, 854)));
            assert_eq!(
                parse_sps(codec, &odd).unwrap().colour,
                before.colour,
                "renk korunmalı"
            );
            // Asıl SPS'te iki bölüm de varsa eski değerlere dönünce bayt bayt aynı SPS çıkmalı.
            if before.colour.is_some() && before.aspect_idc.is_some() {
                let back = rewrite_sps(
                    codec,
                    &new,
                    Target {
                        sar: before.sar,
                        colour: before.colour,
                    },
                )
                .unwrap()
                .unwrap();
                if before.aspect_idc == Some(255) || before.sar.is_none() {
                    // tablo dışı ya da belirsiz SAR yeniden üretilemeyebilir: değerleri karşılaştır
                    assert_eq!(parse_sps(codec, &back).unwrap().colour, before.colour);
                } else {
                    assert_eq!(back, sps, "{codec:?}: A→B→A bit bit aynı olmalı");
                }
            }
        }
    }

    #[test]
    fn escape_round_trip() {
        let mut seed = 0x1234_5678u32;
        for len in [0usize, 1, 2, 3, 4, 17, 300] {
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // bol sıfır ve küçük değer: öykünme koşullarını zorla
                    [0u8, 0, 0, 1, 2, 3, 0xff][(seed >> 24) as usize % 7]
                })
                .collect();
            let esc = escape(&data);
            assert_eq!(unescape(&esc), data);
            for w in esc.windows(3) {
                assert!(
                    !(w[0] == 0 && w[1] == 0 && w[2] <= 2),
                    "kaçışsız 00 00 0x kalmamalı: {esc:02x?}"
                );
            }
        }
    }

    #[test]
    fn probe_finds_first_sps_in_parameter_sets() {
        // csd-0 biçimi: VPS/SPS/PPS başlangıç kodlarıyla; HEVC'de SPS ikinci NAL'dir.
        for (codec, sps) in all() {
            let mut stream = vec![0, 0, 0, 1];
            if codec == Codec::Hevc {
                stream.extend(hex("40010c01ffff016000000300900000030000030078959809"));
                stream.extend([0, 0, 0, 1]);
            }
            stream.extend(&sps);
            stream.extend([0, 0, 0, 1]);
            stream.extend(hex(if codec == Codec::Avc {
                "68ebe3cb22c0"
            } else {
                "4401c172b46240"
            }));
            let probed = probe_annexb(codec, &stream).unwrap().unwrap();
            assert_eq!(probed, parse_sps(codec, &sps).unwrap());
        }
        // SAR 32:27 + BT.709 örneği değerleriyle okunur.
        let mut stream = vec![0, 0, 1];
        stream.extend(hex(AVC[2]));
        let p = probe_annexb(Codec::Avc, &stream).unwrap().unwrap();
        assert_eq!(p.sar, Some((32, 27)));
        assert_eq!(p.colour.map(|c| (c.primaries, c.matrix)), Some((1, 1)));
        assert_eq!(p.full_range, Some(false));
        // SPS yok
        assert_eq!(
            probe_annexb(Codec::Avc, &[0, 0, 1, 0x68, 0xeb]).unwrap(),
            None
        );
    }

    #[test]
    fn annexb_only_sps_changes() {
        let sps = hex(AVC[1]);
        let pps = hex("68ebe3cb22c0");
        let idr = hex("6588840021ffd6bf");
        let mut stream = vec![0, 0, 0, 1];
        stream.extend(&sps);
        stream.extend([0, 0, 1]);
        stream.extend(&pps);
        stream.extend([0, 0, 0, 1]);
        stream.extend(&idr);
        let t = Target {
            sar: Some((32, 27)),
            colour: Some(BT601),
        };
        let out = rewrite_annexb(Codec::Avc, &stream, t).unwrap().unwrap();
        let expected_sps = rewrite_sps(Codec::Avc, &sps, t).unwrap().unwrap();
        let mut expected = vec![0, 0, 0, 1];
        expected.extend(&expected_sps);
        expected.extend([0, 0, 1]);
        expected.extend(&pps);
        expected.extend([0, 0, 0, 1]);
        expected.extend(&idr);
        assert_eq!(out, expected);
        // SPS'siz akış (P karesi) değişmez
        assert_eq!(
            rewrite_annexb(Codec::Avc, &[0, 0, 1, 0x41, 0x9a, 0x02], t).unwrap(),
            None
        );
        // Zaten doğruysa değişmez
        assert_eq!(rewrite_annexb(Codec::Avc, &out, t).unwrap(), None);
    }

    /// Yerel uçtan uca denetim: VUI_IN (Annex B) → VUI_OUT, ardından ffprobe ve framemd5 ile karşılaştırılır.
    #[test]
    #[ignore = "yerel dosya gerekir: VUI_IN, VUI_OUT, VUI_CODEC=avc|hevc"]
    fn rewrite_file() {
        let input = std::env::var("VUI_IN").unwrap();
        let output = std::env::var("VUI_OUT").unwrap();
        let codec = if std::env::var("VUI_CODEC").unwrap() == "hevc" {
            Codec::Hevc
        } else {
            Codec::Avc
        };
        let data = std::fs::read(input).unwrap();
        let t = Target {
            sar: Some((40, 33)),
            colour: Some(BT601),
        };
        let out = rewrite_annexb(codec, &data, t).unwrap().unwrap_or(data);
        std::fs::write(output, out).unwrap();
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::reuse::*;

    const W: usize = 256;
    const H: usize = 64;

    /// Dokulu arka plan (kaydırılabilir) + sabit "altyazı" bloğu.
    fn frame(shift: usize, text: bool) -> (Vec<u8>, Vec<u8>) {
        let mut band = vec![0u8; W * H];
        let mut mask = vec![0u8; W * H];
        for y in 0..H {
            for x in 0..W {
                let v = (((x + shift) * 7 + y * 13) % 61) as u8 + 60;
                band[y * W + x] = v;
                if text && (24..40).contains(&y) && (64..192).contains(&x) {
                    band[y * W + x] = 250;
                    mask[y * W + x] = 255;
                }
            }
        }
        (band, mask)
    }

    #[test]
    fn regions_split_gate_and_ring() {
        let (_, mask) = frame(0, true);
        let (gate, ring) = regions(&mask, W, H, 1, 4);
        let gate_px = gate.iter().filter(|&&v| v != 0).count();
        let ring_px = ring.iter().filter(|&&v| v != 0).count();
        assert_eq!(gate_px, (16 + 4) * (128 + 4)); // 2 px genişletme
        assert_eq!(ring_px, (20 + 16) * (132 + 16) - gate_px); // +8 px halka
        assert!(gate.iter().zip(&ring).all(|(&g, &r)| !(g != 0 && r != 0)));
    }

    #[test]
    fn static_scene_runs_detector_periodically_and_reuses_patch() {
        let mut p = Planner::new(Config::default());
        let (band, mask) = frame(0, true);
        let plans: Vec<Plan> = (0..30).map(|_| p.plan(&band, &mask, W, H, false)).collect();
        assert!(plans[0].run_detector && !plans[0].reuse_patch);
        let runs: Vec<usize> = plans
            .iter()
            .enumerate()
            .filter(|(_, x)| x.run_detector)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(runs, vec![0, 10, 20]);
        assert!(
            plans
                .iter()
                .filter(|x| !x.run_detector)
                .all(|x| x.reuse_patch)
        );
        assert_eq!((p.stats.detector_runs, p.stats.reused), (3, 27));
    }

    #[test]
    fn moving_background_behind_static_subtitle_never_freezes_patch() {
        // "Donmuş yama" tuzağı: altyazı sabit, arkadaki sahne kayıyor.
        let mut p = Planner::new(Config::default());
        let mut plans = Vec::new();
        for i in 0..30 {
            let (band, mask) = frame(i, true);
            plans.push(p.plan(&band, &mask, W, H, false));
        }
        // algılayıcı yalnızca periyotla çalışır (yazı sabit), ama yama hiç yeniden kullanılmaz
        let runs: Vec<usize> = plans
            .iter()
            .enumerate()
            .filter(|(_, x)| x.run_detector)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(runs, vec![0, 10, 20], "{plans:?}");
        assert_eq!(
            p.stats.reused, 0,
            "hareketli arka planda yama dondurulmamalı: {plans:?}"
        );
        assert!(
            plans[5].text_sad_x100 <= 200 && plans[5].ring_sad_x100 > 50,
            "{:?}",
            plans[5]
        );
    }

    #[test]
    fn new_subtitle_anywhere_in_band_triggers_detector() {
        let mut p = Planner::new(Config::default());
        let (empty, no_mask) = frame(0, false);
        p.plan(&empty, &no_mask, W, H, false);
        assert!(p.plan(&empty, &no_mask, W, H, false).reuse_patch);
        let (with_text, _) = frame(0, true);
        // maske henüz bilinmiyor (boş): yazı bölgesi farkı 0, ama sütun bloğu büyük fark görür
        let plan = p.plan(&with_text, &no_mask, W, H, false);
        assert!(plan.run_detector && !plan.reuse_patch, "{plan:?}");
        // algılayıcı maskeyi değiştirdiğini bildirince yeniden kullanım iptal
        let mut next = p.plan(&with_text, &no_mask, W, H, false);
        p.confirm_mask(&mut next, true);
        assert!(!next.reuse_patch);
    }

    #[test]
    fn reuse_is_capped_and_reset_restarts() {
        let cfg = Config {
            det_period: 1000,
            max_reuse: 5,
            ..Config::default()
        };
        let mut p = Planner::new(cfg);
        let (band, mask) = frame(3, true);
        let reuse: Vec<bool> = (0..13)
            .map(|_| p.plan(&band, &mask, W, H, false).reuse_patch)
            .collect();
        assert_eq!(
            reuse,
            vec![
                false, true, true, true, true, true, false, true, true, true, true, true, false
            ]
        );
        p.reset();
        let first = p.plan(&band, &mask, W, H, false);
        assert!(first.run_detector && !first.reuse_patch);
    }
}

#[cfg(test)]
mod finish_tests {
    use super::*;

    #[test]
    fn ramp_and_distance_follow_cosine_profile() {
        assert_eq!(finish::ramp(0.0, 4), 0);
        assert_eq!(finish::ramp(10.0, 4), 255);
        assert_eq!(finish::ramp(1.0, 0), 255);
        let r: Vec<u8> = (1..=5).map(|d| finish::ramp(d as f32, 4)).collect();
        assert!(r.windows(2).all(|p| p[0] < p[1]), "{r:?}");
        assert_eq!(r[4], 255);
        // 6×6 bölgenin mesafesi: kenar 1, merkeze doğru artar
        let region: Vec<bool> = (0..100)
            .map(|k| (2..8).contains(&(k % 10)) && (2..8).contains(&(k / 10)))
            .collect();
        let d = finish::inner_distance(&region, 10, 10, [false; 4]);
        // kare kenarına dayanan bölge: kenar tarafında yumuşatma yok
        let corner: Vec<bool> = (0..100).map(|k| k % 10 < 5 && k / 10 < 5).collect();
        let dc = finish::inner_distance(&corner, 10, 10, [true, true, false, false]);
        assert_eq!(dc[0], 5.0);
        assert_eq!(dc[4 * 10 + 4], 1.0);
        assert_eq!(d[2 * 10 + 2], 1.0);
        assert_eq!(d[4 * 10 + 4], 3.0);
        assert_eq!(d[0], 0.0);
    }

    #[test]
    fn feather_blends_only_inside_region_and_keeps_outside_bit_exact() {
        let geo = FrameGeometry::new(64, 48).unwrap();
        let mut orig = vec![0u8; geo.frame_bytes()];
        for (i, v) in orig.iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        let mut est = Estimate::new(geo);
        let roi = Rect::new(8, 8, 32, 16);
        est.paint(roi, 200, 90, 160, 255).unwrap();
        let mut dst = orig.clone();
        compose(geo, &mut dst, &est, None);
        finish_roi(
            geo,
            &orig,
            &mut dst,
            &|i| est.weight[i] != 0,
            roi,
            4,
            false,
            0,
        );
        let w = geo.width();
        for y in 0..geo.height() {
            for x in 0..w {
                let i = y * w + x;
                let inside = (8..40).contains(&x) && (8..24).contains(&y);
                if !inside {
                    assert_eq!(dst[i], orig[i], "bölge dışı değişmemeli ({x},{y})");
                    continue;
                }
                let depth = (x - 8 + 1).min(40 - x).min(y - 8 + 1).min(24 - y);
                if depth > 4 {
                    assert_eq!(dst[i], 200, "derin bölge tam onarım ({x},{y})");
                } else {
                    let a = finish::ramp(depth as f32, 4) as u32;
                    let expect = ((200 * a + orig[i] as u32 * (255 - a) + 127) / 255) as u8;
                    assert!(
                        (dst[i] as i32 - expect as i32).abs() <= 1,
                        "({x},{y}) {} ≠ {expect}",
                        dst[i]
                    );
                }
            }
        }
    }

    #[test]
    fn grain_restores_measured_camera_noise_deterministically() {
        let geo = FrameGeometry::new(128, 64).unwrap();
        let mut rng = finish::Grain::new(99);
        let mut orig = vec![128u8; geo.frame_bytes()];
        for v in orig[..geo.luma_bytes()].iter_mut() {
            *v = (128.0 + 8.0 * rng.normal()).round().clamp(0.0, 255.0) as u8;
        }
        let mut est = Estimate::new(geo);
        let region_rect = Rect::new(32, 16, 64, 32);
        est.paint(region_rect, 128, 128, 128, 255).unwrap();
        let roi = Rect::new(0, 0, 128, 64);
        let run = |seed: u64| {
            let mut dst = orig.clone();
            compose(geo, &mut dst, &est, None);
            let added = finish_roi(
                geo,
                &orig,
                &mut dst,
                &|i| est.weight[i] != 0,
                roi,
                0,
                true,
                seed,
            );
            (dst, added)
        };
        let (a, added) = run(5);
        let (b, _) = run(5);
        assert_eq!(a, b, "aynı kare sırası aynı gürültü");
        let w = geo.width();
        let plane = &a[..geo.luma_bytes()];
        let inside = |i: usize| (32..96).contains(&(i % w)) && (16..48).contains(&(i / w));
        let sigma_in = finish::noise_sigma(plane, w, geo.height(), &inside).unwrap();
        let sigma_out = finish::noise_sigma(plane, w, geo.height(), &|i| !inside(i)).unwrap();
        assert!(added > 6.0 && added < 10.0, "eklenen σ {added}");
        assert!(
            (sigma_in - sigma_out).abs() / sigma_out < 0.25,
            "içeride σ {sigma_in}, dışarıda σ {sigma_out}"
        );
    }
}

#[cfg(test)]
mod half_tests {
    use super::half::*;

    #[test]
    fn known_values_and_edges() {
        let cases: [(f32, u16); 14] = [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff), // en büyük sonlu
            (65520.0, 0x7c00), // eşitlikte çift → sonsuz
            (1e9, 0x7c00),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (6.103_515_6e-5, 0x0400), // en küçük normal 2^-14
            (5.960_464_5e-8, 0x0001), // en küçük alt-normal 2^-24
            (2.980_232_2e-8, 0x0000), // tam 2^-25: eşitlikte çift → 0
            (1.000_488_3, 0x3c00),    // 1 + 2^-11: eşitlikte çift → 1
        ];
        for (x, h) in cases {
            assert_eq!(f32_to_f16(x), h, "{x:e}");
        }
        assert_eq!(f32_to_f16(1.001_464_8), 0x3c02); // 1 + 3·2^-11 → yukarı (çift)
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn every_half_roundtrips_exactly() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert_eq!(
                    f32_to_f16(x) & 0x7e00,
                    0x7e00,
                    "NaN sessiz kalmalı: {h:#06x}"
                );
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "{h:#06x}");
        }
    }

    #[test]
    fn vector_path_matches_scalar_bit_exactly() {
        // Rastgele bitler (NaN, sonsuz, alt-normal, eşitlik durumları dahil) + eşitlik sınırları.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut src: Vec<f32> = (0..100_003)
            .map(|_| f32::from_bits(next() as u32))
            .collect();
        for k in 0..2048u32 {
            src.push(f32::from_bits((k << 13) | 0x1000)); // tam yarım: eşitlikte çift
            src.push(f32::from_bits(0x3300_0000 + (k << 12))); // alt-normal sınırı
        }
        let mut fast = vec![0u16; src.len()];
        let mut slow = vec![0u16; src.len()];
        narrow(&src, &mut fast);
        narrow_scalar(&src, &mut slow);
        for (i, (a, b)) in fast.iter().zip(&slow).enumerate() {
            assert_eq!(a, b, "{} öğe {i}: {:#010x}", backend(), src[i].to_bits());
        }
        let halves: Vec<u16> = (0..=u16::MAX).chain(0..13).collect();
        let mut wide = vec![0f32; halves.len()];
        widen(&halves, &mut wide);
        for (&h, &x) in halves.iter().zip(&wide) {
            let want = f16_to_f32(h);
            assert!(
                x.to_bits() == want.to_bits()
                    || (x.is_nan() && want.is_nan() && f32_to_f16(x) == f32_to_f16(want)),
                "{} {h:#06x}",
                backend()
            );
        }
    }
}

#[cfg(test)]
mod native_alloc_tests {
    #[test]
    fn native_blocks_are_zeroed_aligned_and_freeable() {
        for bytes in [1usize, 63, 64, 4096, 40 << 20] {
            let ptr = super::jni::native_alloc(bytes).expect("ayrılmalı");
            assert_eq!(ptr as usize % 64, 0);
            // SAFETY: blok `bytes` uzunluğunda ve yeni ayrıldı.
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
            assert!(slice.iter().all(|&b| b == 0));
            slice[bytes - 1] = 7;
            // SAFETY: aynı boyutla ayrıldı.
            unsafe { super::jni::native_free(ptr, bytes) };
        }
    }
}
