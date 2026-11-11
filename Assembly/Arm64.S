// =============================================================================
//  Video_Cleaner v2.0 — ARM64 SIMD çekirdekleri (NEON / Advanced SIMD)
// -----------------------------------------------------------------------------
//  Derleme   : rustc'nin tümleşik derleyicisi; Rust/Core.rs içindeki
//                  global_asm!(include_str!("../Assembly/Arm64.S"), options(raw))
//              C önişlemcisi ÇALIŞMAZ: #include / #define kullanmayın.
//  ABI       : AAPCS64 (Android arm64-v8a ve Linux aarch64).
//  Gereksinim: NEON. ARMv8-A AArch64'te zorunlu olduğu için çalışma anı
//              denetimi gerekmez. SVE2 her cihazda bulunmadığından KULLANILMADI;
//              ileride eklenirse çalışma anında (getauxval/HWCAP2) seçilmelidir.
//  Sözleşme  : * Her çekirdek yalnızca TAM blokları işler (1 blok = 32 bayt
//                giriş). Blok sayısı son argümandır; kuyruğu Rust tamamlar.
//              * Yalnızca v0-v7 ve v16-v31 kullanılır (v8-v15 callee-saved).
//              * x86_64.asm ile bit düzeyinde aynı sonucu üretir; ortak başvuru
//                gerçeklemesi Core.rs → kernels::portable içindedir.
// =============================================================================

        .text

// -----------------------------------------------------------------------------
//  u64 vc_sad_u8_neon(const u8 *a, const u8 *b, usize blocks)
//      x0 = a, x1 = b, x2 = blocks (×32 bayt)
// -----------------------------------------------------------------------------
        .globl  vc_sad_u8_neon
        .hidden vc_sad_u8_neon
        .type   vc_sad_u8_neon, %function
        .p2align 4
vc_sad_u8_neon:
        movi    v16.2d, #0                      // toplayıcı: 2 × u64
        cbz     x2, .Lvc_sad_done
.Lvc_sad_loop:
        ldp     q0, q1, [x0], #32
        ldp     q2, q3, [x1], #32
        uabd    v0.16b, v0.16b, v2.16b
        uabd    v1.16b, v1.16b, v3.16b
        uaddlp  v0.8h, v0.16b                   // 8 × u16 (<= 510)
        uaddlp  v1.8h, v1.16b
        add     v0.8h, v0.8h, v1.8h             // <= 1020
        uaddlp  v0.4s, v0.8h                    // 4 × u32
        uadalp  v16.2d, v0.4s                   // u64'e biriktir (taşma yok)
        subs    x2, x2, #1
        b.ne    .Lvc_sad_loop
.Lvc_sad_done:
        addp    d16, v16.2d
        fmov    x0, d16
        ret
        .size   vc_sad_u8_neon, . - vc_sad_u8_neon

// -----------------------------------------------------------------------------
//  void vc_select_u8_neon(u8 *dst, const u8 *restored, const u8 *original,
//                         const u8 *mask, usize blocks)
//      x0 = dst, x1 = restored, x2 = original, x3 = mask, x4 = blocks
//  dst[i] = mask[i] != 0 ? restored[i] : original[i]   (dst == original olabilir)
// -----------------------------------------------------------------------------
        .globl  vc_select_u8_neon
        .hidden vc_select_u8_neon
        .type   vc_select_u8_neon, %function
        .p2align 4
vc_select_u8_neon:
        cbz     x4, .Lvc_sel_done
.Lvc_sel_loop:
        ldp     q0, q1, [x3], #32               // maske
        ldp     q2, q3, [x1], #32               // onarılmış
        ldp     q4, q5, [x2], #32               // orijinal
        cmeq    v0.16b, v0.16b, #0              // maske == 0 → 0xFF
        cmeq    v1.16b, v1.16b, #0
        bsl     v0.16b, v4.16b, v2.16b          // maske == 0 → orijinal
        bsl     v1.16b, v5.16b, v3.16b
        stp     q0, q1, [x0], #32
        subs    x4, x4, #1
        b.ne    .Lvc_sel_loop
.Lvc_sel_done:
        ret
        .size   vc_select_u8_neon, . - vc_select_u8_neon

// -----------------------------------------------------------------------------
//  void vc_haar_fwd_u8_neon(const u8 *row0, const u8 *row1,
//                           i16 *ll, i16 *hl, i16 *lh, i16 *hh, usize blocks)
//      x0 = row0, x1 = row1, x2 = ll, x3 = hl, x4 = lh, x5 = hh, x6 = blocks
//  LL = a+b+c+d   HL = a-b+c-d   LH = a+b-c-d   HH = a-b-c+d
// -----------------------------------------------------------------------------
        .globl  vc_haar_fwd_u8_neon
        .hidden vc_haar_fwd_u8_neon
        .type   vc_haar_fwd_u8_neon, %function
        .p2align 4
vc_haar_fwd_u8_neon:
        cbz     x6, .Lvc_hf_done
.Lvc_hf_loop:
        ld2     {v0.16b, v1.16b}, [x0], #32     // v0 = a (çift), v1 = b (tek)
        ld2     {v2.16b, v3.16b}, [x1], #32     // v2 = c,        v3 = d
        uaddl   v4.8h,  v0.8b,  v1.8b           // a + b
        uaddl2  v5.8h,  v0.16b, v1.16b
        usubl   v6.8h,  v0.8b,  v1.8b           // a - b
        usubl2  v7.8h,  v0.16b, v1.16b
        uaddl   v16.8h, v2.8b,  v3.8b           // c + d
        uaddl2  v17.8h, v2.16b, v3.16b
        usubl   v18.8h, v2.8b,  v3.8b           // c - d
        usubl2  v19.8h, v2.16b, v3.16b
        add     v20.8h, v4.8h, v16.8h           // LL
        add     v21.8h, v5.8h, v17.8h
        sub     v22.8h, v4.8h, v16.8h           // LH
        sub     v23.8h, v5.8h, v17.8h
        add     v24.8h, v6.8h, v18.8h           // HL
        add     v25.8h, v7.8h, v19.8h
        sub     v26.8h, v6.8h, v18.8h           // HH
        sub     v27.8h, v7.8h, v19.8h
        stp     q20, q21, [x2], #32
        stp     q24, q25, [x3], #32
        stp     q22, q23, [x4], #32
        stp     q26, q27, [x5], #32
        subs    x6, x6, #1
        b.ne    .Lvc_hf_loop
.Lvc_hf_done:
        ret
        .size   vc_haar_fwd_u8_neon, . - vc_haar_fwd_u8_neon

// -----------------------------------------------------------------------------
//  void vc_haar_inv_i16_neon(const i16 *ll, const i16 *hl, const i16 *lh,
//                            const i16 *hh, u8 *row0, u8 *row1, usize blocks)
//      x0 = ll, x1 = hl, x2 = lh, x3 = hh, x4 = row0, x5 = row1, x6 = blocks
//  Doygun i16 toplama/çıkarma, kesin yuvarlamalı >> 2 (srshr), 0..255 kırpma.
// -----------------------------------------------------------------------------
        .globl  vc_haar_inv_i16_neon
        .hidden vc_haar_inv_i16_neon
        .type   vc_haar_inv_i16_neon, %function
        .p2align 4
vc_haar_inv_i16_neon:
        cbz     x6, .Lvc_hi_done
.Lvc_hi_loop:
        ldp     q0, q1, [x0], #32               // LL
        ldp     q2, q3, [x1], #32               // HL
        ldp     q4, q5, [x2], #32               // LH
        ldp     q6, q7, [x3], #32               // HH
        sqadd   v16.8h, v0.8h, v4.8h            // t0 = LL + LH
        sqadd   v17.8h, v1.8h, v5.8h
        sqsub   v18.8h, v0.8h, v4.8h            // t1 = LL - LH
        sqsub   v19.8h, v1.8h, v5.8h
        sqadd   v20.8h, v2.8h, v6.8h            // t2 = HL + HH
        sqadd   v21.8h, v3.8h, v7.8h
        sqsub   v22.8h, v2.8h, v6.8h            // t3 = HL - HH
        sqsub   v23.8h, v3.8h, v7.8h
        sqadd   v0.8h, v16.8h, v20.8h           // 4a = t0 + t2
        sqadd   v1.8h, v17.8h, v21.8h
        sqsub   v2.8h, v16.8h, v20.8h           // 4b = t0 - t2
        sqsub   v3.8h, v17.8h, v21.8h
        sqadd   v4.8h, v18.8h, v22.8h           // 4c = t1 + t3
        sqadd   v5.8h, v19.8h, v23.8h
        sqsub   v6.8h, v18.8h, v22.8h           // 4d = t1 - t3
        sqsub   v7.8h, v19.8h, v23.8h
        srshr   v0.8h, v0.8h, #2
        srshr   v1.8h, v1.8h, #2
        srshr   v2.8h, v2.8h, #2
        srshr   v3.8h, v3.8h, #2
        srshr   v4.8h, v4.8h, #2
        srshr   v5.8h, v5.8h, #2
        srshr   v6.8h, v6.8h, #2
        srshr   v7.8h, v7.8h, #2
        sqxtun  v16.8b,  v0.8h                  // a → u8 (0..255 doygun)
        sqxtun2 v16.16b, v1.8h
        sqxtun  v17.8b,  v2.8h                  // b
        sqxtun2 v17.16b, v3.8h
        st2     {v16.16b, v17.16b}, [x4], #32   // a0 b0 a1 b1 ...
        sqxtun  v18.8b,  v4.8h                  // c
        sqxtun2 v18.16b, v5.8h
        sqxtun  v19.8b,  v6.8h                  // d
        sqxtun2 v19.16b, v7.8h
        st2     {v18.16b, v19.16b}, [x5], #32   // c0 d0 c1 d1 ...
        subs    x6, x6, #1
        b.ne    .Lvc_hi_loop
.Lvc_hi_done:
        ret
        .size   vc_haar_inv_i16_neon, . - vc_haar_inv_i16_neon
