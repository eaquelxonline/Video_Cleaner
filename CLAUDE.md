# Video_Cleaner — çalışma kuralları (Claude oturumları için)

- Tek geliştirme dalı: `Video_Cleaner_Beta`. Bütün commit'ler bu dala push edilir.
- Kullanıcıyla Türkçe konuşulur. Ölçülmeyen şey "geçti" yazılmaz; sonuçlar gerçek verilerle GEÇTİ/KALDI olarak raporlanır.
- Kişisel (ör. WhatsApp) videolar depoya veya CI test dosyalarına eklenmez.

## Her push'tan sonra Google Drive yedeği (kullanıcının isteği)

GitHub hesabının kapatılma riskine karşı, `Video_Cleaner_Beta` dalına yapılan HER push'tan sonra:

1. Deponun kaynak kodu paketlenir: `git archive --format=zip -o Video_Cleaner_Beta_<kısa-sha>_<YYYYMMDD-HHMM>.zip HEAD`
   (izlenmeyen/derleme dosyaları dahil edilmez).
2. Google Drive'da `Video_Cleaner_Yedek` klasörüne yüklenir (klasör yoksa oluşturulur).
3. Aynı klasördeki eski `Video_Cleaner_Beta_*.zip` yedekleri çöp kutusuna taşınır; klasörde yalnızca en son yedek kalır.
4. Kullanıcıya yüklenen dosyanın adı ve commit'i bildirilir.

Bu adımlar Claude oturumu tarafından (Google Drive bağlayıcısı ile) yapılır; oturum dışından yapılan push'lar
otomatik yedeklenmez.
