# Video_Cleaner — çalışma kuralları (Claude oturumları için)

- Tek geliştirme dalı: `Video_Cleaner_Beta`. Bütün commit'ler bu dala push edilir.
- Kullanıcıyla Türkçe konuşulur. Ölçülmeyen şey "geçti" yazılmaz; sonuçlar gerçek verilerle GEÇTİ/KALDI olarak raporlanır.
- Kişisel (ör. WhatsApp) videolar depoya veya CI test dosyalarına eklenmez.

## Her push'tan sonra Google Drive yedeği (kullanıcının isteği)

GitHub hesabının kapatılma riskine karşı `.github/workflows/drive-backup.yml`, `Video_Cleaner_Beta` dalına
yapılan HER push'ta otomatik çalışır:

1. Kaynak kodu `Video_Cleaner_Beta_<kısa-sha>_<YYYYMMDD-HHMM>.zip` (git archive) ve tüm geçmiş aynı adla `.bundle` olarak paketlenir.
2. Google Drive'da `Video_Cleaner_Yedek` klasörüne rclone ile yüklenir (`RCLONE_CONF` sırrı, "gdrive" bağlantısı, kapsam drive.file).
3. Yükleme doğrulandıktan sonra klasördeki eski `Video_Cleaner_Beta_*` yedekleri silinir (Drive çöp kutusu).

Claude oturumları bu iş akışını bozmamalı; push sonrası iş akışının yeşil olduğunu kontrol etmeli.
