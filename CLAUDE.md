# Video_Cleaner — çalışma kuralları (Claude oturumları için)

- Tek geliştirme dalı: `Video_Cleaner_Beta`. Bütün commit'ler bu dala push edilir.
- Kullanıcıyla HER ZAMAN, HER KOŞULDA yalnızca Türkçe konuşulur (otomatik bildirimlere, CI ve zamanlanmış kontrollere verilen yanıtlar dahil). Ölçülmeyen şey "geçti" yazılmaz; sonuçlar gerçek verilerle GEÇTİ/KALDI olarak raporlanır.
- Kişisel (ör. WhatsApp) videolar depoya veya CI test dosyalarına eklenmez.

## Her push'tan sonra Google Drive yedeği (kullanıcının isteği)

GitHub hesabının kapatılma riskine karşı `.github/workflows/drive-backup.yml`, `Video_Cleaner_Beta` dalına
yapılan HER push'ta kullanıcının Drive hesabındaki Google Apps Script betiğini (`.github/scripts/drive_backup.gs`)
tetikler (`DRIVE_HOOK_URL` sırrı). Betik son commit'in kaynak zip'ini `Video_Cleaner_Yedek` klasörüne koyar ve eski
`Video_Cleaner_Beta_*` yedeklerini çöp kutusuna taşır; ayrıca saatte bir kendisi de denetler.
(rclone yolu denendi: rclone'un ortak client_id'si 2026'da kapanıyor ve kişisel OAuth uygulamasında "Publish app"
düğmesi Google tarafında kapalı kalıyor; Testing durumunda izin 7 günde bir düştüğü için kullanılmadı.)

Claude oturumları bu iş akışını bozmamalı; push sonrası iş akışının yeşil olduğunu kontrol etmeli.
