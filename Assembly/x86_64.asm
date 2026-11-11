# ==============================================================================
#  Video_Cleaner v2.0 — x86_64 SIMD çekirdekleri (AVX2)
# ------------------------------------------------------------------------------
#  Sözdizimi : Intel sözdizimi, LLVM/GNU "as" lehçesi. Bu dosya NASM ile DEĞİL,
#              rustc'nin tümleşik derleyicisiyle derlenir:
#                  Rust/Core.rs → global_asm!(include_str!(...), options(raw))
#              Bu sayede build.rs, NASM veya harici C araç zinciri gerekmez.
#  ABI       : System V AMD64 (Linux ve Android x86_64).
#  Gereksinim: AVX2. Android x86_64 ABI'si yalnızca SSE4.2'yi garanti eder; bu
#              yüzden çağıran taraf (Core.rs → kernels) çalışma anında AVX2
#              denetimi yapar, yoksa taşınabilir Rust yoluna düşer.
#              AVX-512 bilerek kullanılmadı.
#  Sözleşme  : * Her çekirdek yalnızca TAM blokları işler (1 blok = 32 bayt
#                giriş). Blok sayısı son argümandır; kuyruğu Rust tamamlar.
#              * Yalnızca caller-saved yazmaçlar kullanılır (yığın çerçevesi yok).
#              * Bellek erişimleri hizasızdır (vmovdqu); hizalama şartı yoktur.
#  Not       : Bunlar CPU çekirdekleridir. RTX 4090 üzerindeki eğitim CUDA
#              üzerinden PyTorch ile yapılır; bu dosyanın GPU ile ilgisi yoktur.
# ==============================================================================

        .pushsection .rodata.cst32, "aM", @progbits, 32
        .p2align 5
.Lvc_bytes_one:                         # 32 × (+1)       : a + b
        .fill   32, 1, 0x01
.Lvc_bytes_pm:                          # 16 × (+1, -1)   : a - b
        .rept   16
        .byte   0x01, 0xFF
        .endr
.Lvc_words_one:                         # 16 × 1
        .fill   16, 2, 1
.Lvc_words_255:                         # 16 × 255
        .fill   16, 2, 255
        .popsection

        .text

# ------------------------------------------------------------------------------
#  u64 vc_sad_u8_avx2(const u8 *a, const u8 *b, usize blocks)
#      rdi = a, rsi = b, rdx = blocks (×32 bayt)
#  Mutlak farkların toplamı (SAD). Sahne geçişi imzası ve hareket ölçümü için.
# ------------------------------------------------------------------------------
        .globl  vc_sad_u8_avx2
        .hidden vc_sad_u8_avx2
        .type   vc_sad_u8_avx2, @function
        .p2align 4
vc_sad_u8_avx2:
        vpxor   xmm0, xmm0, xmm0                        # toplayıcı: 4 × u64
        test    rdx, rdx
        jz      .Lvc_sad_tail
.Lvc_sad_loop:
        vmovdqu ymm1, ymmword ptr [rdi]
        vpsadbw ymm1, ymm1, ymmword ptr [rsi]           # 4 × (8 baytlık SAD)
        vpaddq  ymm0, ymm0, ymm1
        add     rdi, 32
        add     rsi, 32
        dec     rdx
        jnz     .Lvc_sad_loop
.Lvc_sad_tail:
        vextracti128 xmm1, ymm0, 1
        vpaddq  xmm0, xmm0, xmm1
        vpshufd xmm1, xmm0, 0x4E                        # üst/alt qword yer değiştir
        vpaddq  xmm0, xmm0, xmm1
        vmovq   rax, xmm0
        vzeroupper
        ret
        .size   vc_sad_u8_avx2, . - vc_sad_u8_avx2

# ------------------------------------------------------------------------------
#  void vc_select_u8_avx2(u8 *dst, const u8 *restored, const u8 *original,
#                         const u8 *mask, usize blocks)
#      rdi = dst, rsi = restored, rdx = original, rcx = mask, r8 = blocks
#  dst[i] = mask[i] != 0 ? restored[i] : original[i]
#  Bit-exact koruma kapısı: çarpma/yuvarlama yok, yalnızca bayt seçimi.
#  dst == original (yerinde çalışma) güvenlidir.
# ------------------------------------------------------------------------------
        .globl  vc_select_u8_avx2
        .hidden vc_select_u8_avx2
        .type   vc_select_u8_avx2, @function
        .p2align 4
vc_select_u8_avx2:
        test    r8, r8
        jz      .Lvc_sel_done
        vpxor   xmm5, xmm5, xmm5
        xor     eax, eax
.Lvc_sel_loop:
        vmovdqu ymm0, ymmword ptr [rcx + rax]           # maske
        vmovdqu ymm1, ymmword ptr [rsi + rax]           # onarılmış
        vmovdqu ymm2, ymmword ptr [rdx + rax]           # orijinal
        vpcmpeqb ymm0, ymm0, ymm5                       # maske == 0 → 0xFF
        vpblendvb ymm1, ymm1, ymm2, ymm0                # maske == 0 → orijinal
        vmovdqu ymmword ptr [rdi + rax], ymm1
        add     rax, 32
        dec     r8
        jnz     .Lvc_sel_loop
.Lvc_sel_done:
        vzeroupper
        ret
        .size   vc_select_u8_avx2, . - vc_select_u8_avx2

# ------------------------------------------------------------------------------
#  void vc_haar_fwd_u8_avx2(const u8 *row0, const u8 *row1,
#                           i16 *ll, i16 *hl, i16 *lh, i16 *hh, usize blocks)
#      rdi = row0, rsi = row1, rdx = ll, rcx = hl, r8 = lh, r9 = hh,
#      [rsp+8] = blocks (blok = 32 piksel giriş → bant başına 16 katsayı)
#
#  Kayıpsız tamsayı 2×2 Haar:   | a b |     LL = a+b+c+d    HL = a-b+c-d
#                               | c d |     LH = a+b-c-d    HH = a-b-c+d
#  Not: 4 alt bant toplamda girişle AYNI sayıda katsayı içerir; dönüşüm tek
#  başına belleği küçültmez. Amaç kenar/detay analizi ve frekans denetimidir.
# ------------------------------------------------------------------------------
        .globl  vc_haar_fwd_u8_avx2
        .hidden vc_haar_fwd_u8_avx2
        .type   vc_haar_fwd_u8_avx2, @function
        .p2align 4
vc_haar_fwd_u8_avx2:
        mov     r10, qword ptr [rsp + 8]
        test    r10, r10
        jz      .Lvc_hf_done
        vmovdqa ymm6, ymmword ptr [rip + .Lvc_bytes_one]
        vmovdqa ymm7, ymmword ptr [rip + .Lvc_bytes_pm]
        xor     eax, eax                                # 32 px giriş = 32 bayt çıkış
.Lvc_hf_loop:
        vmovdqu ymm0, ymmword ptr [rdi + rax]           # a0 b0 a1 b1 ...
        vmovdqu ymm1, ymmword ptr [rsi + rax]           # c0 d0 c1 d1 ...
        vpmaddubsw ymm2, ymm0, ymm6                     # a + b
        vpmaddubsw ymm3, ymm0, ymm7                     # a - b
        vpmaddubsw ymm4, ymm1, ymm6                     # c + d
        vpmaddubsw ymm5, ymm1, ymm7                     # c - d
        vpaddw  ymm0, ymm2, ymm4                        # LL
        vpsubw  ymm1, ymm2, ymm4                        # LH
        vpaddw  ymm2, ymm3, ymm5                        # HL
        vpsubw  ymm3, ymm3, ymm5                        # HH
        vmovdqu ymmword ptr [rdx + rax], ymm0
        vmovdqu ymmword ptr [rcx + rax], ymm2
        vmovdqu ymmword ptr [r8  + rax], ymm1
        vmovdqu ymmword ptr [r9  + rax], ymm3
        add     rax, 32
        dec     r10
        jnz     .Lvc_hf_loop
.Lvc_hf_done:
        vzeroupper
        ret
        .size   vc_haar_fwd_u8_avx2, . - vc_haar_fwd_u8_avx2

# ------------------------------------------------------------------------------
#  void vc_haar_inv_i16_avx2(const i16 *ll, const i16 *hl, const i16 *lh,
#                            const i16 *hh, u8 *row0, u8 *row1, usize blocks)
#      rdi = ll, rsi = hl, rdx = lh, rcx = hh, r8 = row0, r9 = row1,
#      [rsp+8] = blocks (blok = bant başına 16 katsayı → 32 piksel çıkış)
#
#  Ters dönüşüm (doygun i16 aritmetiği, sonra kesin yuvarlama ve 0..255 kırpma):
#      a = rsh2(sat(sat(LL+LH) + sat(HL+HH)))    b = rsh2(sat(sat(LL+LH) - sat(HL+HH)))
#      c = rsh2(sat(sat(LL-LH) + sat(HL-HH)))    d = rsh2(sat(sat(LL-LH) - sat(HL-HH)))
#      rsh2(x) = (x + 2) >> 2 = (x >> 2) + ((x >> 1) & 1)   (taşmasız)
#  İleri dönüşümden gelen katsayılar için sonuç bit düzeyinde kayıpsızdır.
# ------------------------------------------------------------------------------
        .macro  VC_RSH2_CLAMP reg, tmp
        vpsrlw  \tmp, \reg, 1
        vpand   \tmp, \tmp, ymm6
        vpsraw  \reg, \reg, 2
        vpaddw  \reg, \reg, \tmp
        vpmaxsw \reg, \reg, ymm5
        vpminsw \reg, \reg, ymm7
        .endm

        .globl  vc_haar_inv_i16_avx2
        .hidden vc_haar_inv_i16_avx2
        .type   vc_haar_inv_i16_avx2, @function
        .p2align 4
vc_haar_inv_i16_avx2:
        mov     r10, qword ptr [rsp + 8]
        test    r10, r10
        jz      .Lvc_hi_done
        vpxor   xmm5, xmm5, xmm5                        # 0
        vmovdqa ymm6, ymmword ptr [rip + .Lvc_words_one]
        vmovdqa ymm7, ymmword ptr [rip + .Lvc_words_255]
        xor     eax, eax
.Lvc_hi_loop:
        vmovdqu ymm0, ymmword ptr [rdi + rax]           # LL
        vmovdqu ymm1, ymmword ptr [rsi + rax]           # HL
        vmovdqu ymm2, ymmword ptr [rdx + rax]           # LH
        vmovdqu ymm3, ymmword ptr [rcx + rax]           # HH
        vpaddsw ymm4, ymm0, ymm2                        # t0 = LL + LH
        vpsubsw ymm0, ymm0, ymm2                        # t1 = LL - LH
        vpaddsw ymm2, ymm1, ymm3                        # t2 = HL + HH
        vpsubsw ymm1, ymm1, ymm3                        # t3 = HL - HH
        vpaddsw ymm3, ymm4, ymm2                        # 4a = t0 + t2
        vpsubsw ymm4, ymm4, ymm2                        # 4b = t0 - t2
        vpaddsw ymm2, ymm0, ymm1                        # 4c = t1 + t3
        vpsubsw ymm0, ymm0, ymm1                        # 4d = t1 - t3
        VC_RSH2_CLAMP ymm3, ymm8                        # a
        VC_RSH2_CLAMP ymm4, ymm8                        # b
        VC_RSH2_CLAMP ymm2, ymm8                        # c
        VC_RSH2_CLAMP ymm0, ymm8                        # d
        vpsllw  ymm4, ymm4, 8
        vpor    ymm3, ymm3, ymm4                        # (a | b << 8) → a0 b0 a1 b1 ...
        vmovdqu ymmword ptr [r8 + rax], ymm3
        vpsllw  ymm0, ymm0, 8
        vpor    ymm2, ymm2, ymm0                        # (c | d << 8) → c0 d0 c1 d1 ...
        vmovdqu ymmword ptr [r9 + rax], ymm2
        add     rax, 32
        dec     r10
        jnz     .Lvc_hi_loop
.Lvc_hi_done:
        vzeroupper
        ret
        .size   vc_haar_inv_i16_avx2, . - vc_haar_inv_i16_avx2

# ------------------------------------------------------------------------------
#  void vc_max5_u8_avx2(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2,
#                       const u8 *r3, const u8 *r4, usize blocks)
#      rdi = dst, rsi = r0, rdx = r1, rcx = r2, r8 = r3, r9 = r4,
#      [rsp+8] = blocks (×32 bayt)
#  dst[i] = max(r0[i], r1[i], r2[i], r3[i], r4[i])
#  5×5 maske genişletme (dilation) ayrılabilir: yatayda kaydırılmış işaretçilerle,
#  dikeyde komşu satırlarla aynı çekirdek çağrılır. Türkçe harf uzantıları (Ç Ğ İ Ö Ş Ü
#  noktaları/çengelleri), kontur ve düşen gölge onarım maskesine böyle katılır.
# ------------------------------------------------------------------------------
        .globl  vc_max5_u8_avx2
        .hidden vc_max5_u8_avx2
        .type   vc_max5_u8_avx2, @function
        .p2align 4
vc_max5_u8_avx2:
        mov     rax, qword ptr [rsp + 8]
        test    rax, rax
        jz      .Lvc_max5_done
        xor     r10, r10
.Lvc_max5_loop:
        vmovdqu ymm0, ymmword ptr [rsi + r10]
        vpmaxub ymm0, ymm0, ymmword ptr [rdx + r10]
        vpmaxub ymm0, ymm0, ymmword ptr [rcx + r10]
        vpmaxub ymm0, ymm0, ymmword ptr [r8 + r10]
        vpmaxub ymm0, ymm0, ymmword ptr [r9 + r10]
        vmovdqu ymmword ptr [rdi + r10], ymm0
        add     r10, 32
        dec     rax
        jnz     .Lvc_max5_loop
.Lvc_max5_done:
        vzeroupper
        ret
        .size   vc_max5_u8_avx2, . - vc_max5_u8_avx2

        .pushsection .rodata.cst4, "aM", @progbits, 4
        .p2align 2
.Lvc_f255:
        .long   0x437f0000                      # 255.0f
.Lvc_d65535:
        .long   65535
        .popsection

# ------------------------------------------------------------------------------
#  void vc_unblend_u16_avx2(u16 *dst, const u16 *img, const u16 *fg,
#                           const u8 *alpha, usize blocks)
#      rdi = dst, rsi = img, rdx = fg, rcx = alpha, r8 = blocks (×16 piksel)
#
#  Analitik ters karışım (doğrusal ışık, Q16: 0..65535):
#      B = round_even( (255·I − α·F) / (255 − α) ),  [0, 65535] aralığına kırpılır.
#  Çağıran α ≤ 254 garanti eder (α > eşik pikselleri modele bırakılır, α = 0 → B = I).
#  Ara değerler f32'de tam sayıdır (|pay| < 2^24); yalnızca bölme yuvarlanır. FMA
#  KULLANILMAZ: Rust başvurusu ve NEON çekirdeğiyle bit-bit aynı sonuç.
# ------------------------------------------------------------------------------
        .globl  vc_unblend_u16_avx2
        .hidden vc_unblend_u16_avx2
        .type   vc_unblend_u16_avx2, @function
        .p2align 4
vc_unblend_u16_avx2:
        test    r8, r8
        jz      .Lvc_unb_done
        vbroadcastss ymm6, dword ptr [rip + .Lvc_f255]
        vpbroadcastd ymm7, dword ptr [rip + .Lvc_d65535]
        vpxor   xmm8, xmm8, xmm8
.Lvc_unb_loop:
        # piksel 0..7
        vpmovzxwd ymm0, xmmword ptr [rsi]
        vpmovzxwd ymm1, xmmword ptr [rdx]
        vpmovzxbd ymm2, qword ptr [rcx]
        vcvtdq2ps ymm0, ymm0
        vcvtdq2ps ymm1, ymm1
        vcvtdq2ps ymm2, ymm2
        vmulps  ymm0, ymm0, ymm6                        # 255·I
        vmulps  ymm1, ymm1, ymm2                        # α·F
        vsubps  ymm0, ymm0, ymm1                        # pay
        vsubps  ymm2, ymm6, ymm2                        # 255 − α
        vdivps  ymm0, ymm0, ymm2
        vcvtps2dq ymm0, ymm0                            # en yakın, eşitlikte çift
        vpmaxsd ymm0, ymm0, ymm8
        vpminsd ymm0, ymm0, ymm7
        # piksel 8..15
        vpmovzxwd ymm3, xmmword ptr [rsi + 16]
        vpmovzxwd ymm4, xmmword ptr [rdx + 16]
        vpmovzxbd ymm5, qword ptr [rcx + 8]
        vcvtdq2ps ymm3, ymm3
        vcvtdq2ps ymm4, ymm4
        vcvtdq2ps ymm5, ymm5
        vmulps  ymm3, ymm3, ymm6
        vmulps  ymm4, ymm4, ymm5
        vsubps  ymm3, ymm3, ymm4
        vsubps  ymm5, ymm6, ymm5
        vdivps  ymm3, ymm3, ymm5
        vcvtps2dq ymm3, ymm3
        vpmaxsd ymm3, ymm3, ymm8
        vpminsd ymm3, ymm3, ymm7
        vpackusdw ymm0, ymm0, ymm3                      # şerit içi: [0-3, 8-11 | 4-7, 12-15]
        vpermq  ymm0, ymm0, 0xD8                        # → [0-3, 4-7, 8-11, 12-15]
        vmovdqu ymmword ptr [rdi], ymm0
        add     rsi, 32
        add     rdx, 32
        add     rcx, 16
        add     rdi, 32
        dec     r8
        jnz     .Lvc_unb_loop
.Lvc_unb_done:
        vzeroupper
        ret
        .size   vc_unblend_u16_avx2, . - vc_unblend_u16_avx2
