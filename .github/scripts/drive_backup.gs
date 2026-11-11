// Video_Cleaner — Google Drive yedeği (Google Apps Script, kullanıcının kendi Drive hesabında çalışır).
//
// Kurulum: script.google.com → Yeni proje → bu dosyanın tamamını yapıştır → Kaydet.
//   Proje Ayarları → Betik Özellikleri:
//     GITHUB_TOKEN : GitHub ince ayarlı erişim anahtarı (yalnızca Video_Cleaner, Contents: Read-only)
//     HOOK_KEY     : kendi belirlediğin uzun rastgele bir parola (GitHub'daki DRIVE_HOOK_URL'nin sonuna eklenir)
//   "yedekle" işlevini bir kez Çalıştır (izin ver) → "kur" işlevini bir kez Çalıştır (saatlik yedek denetimi)
//   Dağıt → Yeni dağıtım → Web uygulaması → Yürütme: Ben, Erişim: Herkes → Web uygulaması URL'sini kopyala.
//
// Ne yapar: Video_Cleaner_Beta dalının son commit'inin kaynak zip'ini GitHub'dan indirir, Drive'da
// "Video_Cleaner_Yedek" klasörüne "Video_Cleaner_Beta_<sha>_<UTC tarih>.zip" olarak koyar ve aynı klasördeki
// eski Video_Cleaner_Beta_* yedeklerini çöp kutusuna taşır (30 gün geri alınabilir). Önce yeni yedek yüklenir,
// sonra eskiler silinir. Aynı commit zaten yedeklendiyse hiçbir şey yapmaz.

const OWNER = 'eaquelxonline';
const REPO = 'Video_Cleaner';
const BRANCH = 'Video_Cleaner_Beta';
const FOLDER = 'Video_Cleaner_Yedek';
const PREFIX = 'Video_Cleaner_Beta_';

function ozellik_(ad) {
  const deger = PropertiesService.getScriptProperties().getProperty(ad);
  if (!deger) throw new Error(ad + ' betik özelliği tanımlı değil (Proje Ayarları → Betik Özellikleri)');
  return deger;
}

function github_(yol) {
  const yanit = UrlFetchApp.fetch('https://api.github.com/repos/' + OWNER + '/' + REPO + yol, {
    headers: {
      Authorization: 'Bearer ' + ozellik_('GITHUB_TOKEN'),
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    muteHttpExceptions: true,
    followRedirects: true,
  });
  if (yanit.getResponseCode() !== 200) {
    throw new Error('GitHub ' + yanit.getResponseCode() + ': ' + yanit.getContentText().slice(0, 200));
  }
  return yanit;
}

function klasor_() {
  const bulunan = DriveApp.getFoldersByName(FOLDER);
  return bulunan.hasNext() ? bulunan.next() : DriveApp.createFolder(FOLDER);
}

function yedekle() {
  const kilit = LockService.getScriptLock();
  if (!kilit.tryLock(30000)) return 'meşgul (başka bir yedekleme sürüyor)';
  try {
    const sha = JSON.parse(github_('/commits/' + BRANCH).getContentText()).sha;
    const kisa = sha.slice(0, 7);
    const klasor = klasor_();
    if (klasor.searchFiles("title contains '" + PREFIX + kisa + "' and trashed = false").hasNext()) {
      const sonuc = 'güncel: ' + kisa + ' zaten yedekli';
      console.log(sonuc);
      return sonuc;
    }
    const ad = PREFIX + kisa + '_' + Utilities.formatDate(new Date(), 'UTC', 'yyyyMMdd-HHmm') + '.zip';
    const veri = github_('/zipball/' + sha).getBlob().setName(ad).setContentType('application/zip');
    const yeni = klasor.createFile(veri);
    if (yeni.getSize() < 100000) throw new Error('yedek şüpheli derecede küçük: ' + yeni.getSize() + ' bayt');
    let silinen = 0;
    const dosyalar = klasor.getFiles();
    while (dosyalar.hasNext()) {
      const dosya = dosyalar.next();
      if (dosya.getId() !== yeni.getId() && dosya.getName().indexOf(PREFIX) === 0) {
        dosya.setTrashed(true);
        silinen++;
      }
    }
    const sonuc = 'yüklendi: ' + ad + ' (' + yeni.getSize() + ' bayt), çöpe taşınan eski yedek: ' + silinen;
    console.log(sonuc);
    return sonuc;
  } finally {
    kilit.releaseLock();
  }
}

// GitHub Actions her push'tan sonra bu adresi çağırır: <web uygulaması URL'si>?key=<HOOK_KEY>
function doGet(e) {
  if (!e || !e.parameter || e.parameter.key !== ozellik_('HOOK_KEY')) {
    return ContentService.createTextOutput('YETKISIZ');
  }
  try {
    return ContentService.createTextOutput('OK ' + yedekle());
  } catch (hata) {
    return ContentService.createTextOutput('HATA ' + hata.message);
  }
}

// Bir kez çalıştır: GitHub'dan tetikleme gelmese bile saatte bir son commit'i denetler (aynıysa bir şey yapmaz).
function kur() {
  ScriptApp.getProjectTriggers()
    .filter(function (t) { return t.getHandlerFunction() === 'yedekle'; })
    .forEach(function (t) { ScriptApp.deleteTrigger(t); });
  ScriptApp.newTrigger('yedekle').timeBased().everyHours(1).create();
  console.log('Saatlik yedek denetimi kuruldu.');
}
