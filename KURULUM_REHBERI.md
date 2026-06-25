# 🚀 Sentetik Veri Otomasyonu - Adım Adım Kurulum Rehberi (Tek Ortam)

Bu rehber, projedeki tüm modüllerin (Görüntü Robustness ve Akıllı Veri Artırımı) **tek bir birleştirilmiş sanal ortam** kullanılarak en temiz ve hatasız şekilde nasıl kurulacağını adım adım anlatmaktadır.

---

## Adım 0: İdeal Python Sürümünün Kurulumu (Önkoşul)
Bu projenin listelenen paketlerle %100 uyumlu ve en yüksek performansta (sıfır çakışmayla) çalışacağı **ana sürüm Python 3.12.7'dir.** (Projenin `requirements.txt` kilitleri bu sürüme göre oluşturulmuştur).

Eğer bilgisayarınızda Python 3.12.7 kurulu değilse, işletim sisteminize göre aşağıdaki adımları uygulayarak kurabilirsiniz:

**macOS İçin:**
Öncelikle eğer bilgisayarınızda **Homebrew** yüklü değilse, Terminal'i açıp şu komutu yapıştırarak Homebrew'u kurun (Bu işlem macOS için program kurmayı çok kolaylaştırır):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Homebrew kurulduktan sonra (veya zaten yüklüyse), Python 3.12'yi en temiz şekilde kurmak için Terminal'e şunu yazın:
```bash
brew install python@3.12
```

**Windows İçin:**
[Python Resmi İndirme Sayfası (3.12.7)](https://www.python.org/downloads/release/python-3127/) linkine tıklayıp **"Windows installer (64-bit)"** dosyasını indirin. 
> [!IMPORTANT]
> Kurulumu başlatırken karşınıza çıkan ilk ekranda **"Add python.exe to PATH"** kutucuğunu kesinlikle işaretleyin!

**Python Sürümünü Kontrol Etme:**
Kurulumun başarılı olup olmadığını görmek için Terminal (veya PowerShell) ekranına şunu yazın:
```bash
python3.12 --version
```
*(Windows için `py -3.12 --version` veya `python --version` şeklinde de yazabilirsiniz.)* Ekranda `Python 3.12.x` çıktısını görüyorsanız her şey hazır demektir. (Not: Eğer macOS'te `python3 --version` yazdığınızda eski sürüm çıkıyorsa panik yapmayın; Mac'lerde eski sürümler kalmaya devam eder, bizim projemiz o yüzden kodlarda direkt olarak `python3.12` komutunu kullanmaktadır).

**Eski veya Hatalı Sürümü Silmek İsterseniz (Opsiyonel):**
Eğer uyumsuz veya eski bir sürüm (örn. 3.10) kuruluysa ve projenin temiz çalışması için silmek isterseniz:
- **macOS için:** Terminal'e `brew uninstall python@3.10` (veya sileceğiniz sürüm) yazarak temizleyebilirsiniz.
- **Windows için:** Başlat menüsünden `Program Ekle veya Kaldır` (Add or Remove Programs) bölümüne girip listeden "Python" aratın ve eski sürümleri silin.

---

## Adım 1: Projeyi Bilgisayara İndirme (Clone) veya Güncelleme
Terminalinizi (veya PowerShell/Command Prompt) açın ve projeyi ilk defa indiriyorsanız bilgisayarınıza çekin:

```bash
git clone https://github.com/aliturhan0/sentetik_veri_otomasyonu.git
cd sentetik_veri_otomasyonu
```

**Eğer proje zaten bilgisayarınızda mevcutsa, en son güncellemeleri (hata çözümleri vb.) almak için proje klasörü içindeyken şu komutu çalıştırın:**
```bash
git pull
```

## Adım 2: Büyük Yapay Zeka Modellerini İndirme (Git LFS)
Projenin beyni olan büyük AI modelleri (YOLO, RCGAN, EDSR) ve devasa CSV dosyaları Git LFS ile depolanmaktadır. Bu dosyaları gerçek boyutlarıyla indirmek için sırasıyla şu komutları çalıştırın:

```bash
git lfs install
git lfs pull
```

*(Not: Eğer Git LFS sisteminizde kurulu değilse, macOS için `brew install git-lfs`, Windows için Git kurucusundan LFS seçeneğini aktif ederek kurabilirsiniz.)*

## Adım 3: Tek Ortam (Unified Environment) Oluşturma
Her iki arayüzün sorunsuz çalışması için proje kök dizininde Python 3.12 tabanlı tek bir sanal ortam oluşturuyoruz. (Eğer eski `.venv311` veya `otonom_env` klasörleriniz varsa silebilirsiniz).

**macOS/Linux için:**
```bash
python3.12 -m venv env
source env/bin/activate
```

**Windows için:**
```powershell
py -3.12 -m venv env
.\env\Scripts\activate
```

*(Ortam aktif olduğunda terminal satırınızın başında `(env)` yazısı belirecektir.)*

## Adım 4: Gerekli Paketlerin Kurulumu (Requirements)
Sanal ortamınız aktifken, projede ihtiyaç duyulan tüm kütüphaneleri (FastAPI, TensorFlow, YOLO, PySide6 vb.) tek komutla kurun. Kurulum internet hızınıza bağlı olarak birkaç dakika sürebilir:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Not: `requirements.txt` dosyasındaki paket sürümleri birbiriyle tam uyumlu olacak şekilde ayarlanmıştır, herhangi bir sürüm çakışması yaşanmaz.)*

## Adım 5: Sistemi Başlatma 🎉
Kurulum tamamlandı! Ana menüyü açarak projenizi kullanmaya başlayabilirsiniz. Sanal ortamınız aktif olduğu sürece her zaman şu komutla projeyi başlatabilirsiniz:

```bash
python main_launcher.py
```

Karşınıza çıkan arayüzden:
- **"Görüntü Modelini Aç"** butonu ile kamera/vision dayanıklılık hattına,
- **"Veri Artırımı Modelini Aç"** butonu ile tablo/yörünge sentetik veri üretim hattına giriş yapabilirsiniz.

İyi çalışmalar dileriz!

---

## 📥 Ek: Python 3.12 Manuel Kurulum Rehberi (Homebrew Olmadan)

Eğer bilgisayarınızda Homebrew kurulu değilse veya Terminal komutlarıyla uğraşmak istemiyorsanız, Python 3.12'yi doğrudan Python'un resmi web sitesinden indirip normal bir program kurar gibi yükleyebilirsiniz.

---

### 🍎 macOS İçin Adım Adım Kurulum

**1. İndirme sayfasını açın:**
Tarayıcınızda şu linke tıklayın:
👉 [https://www.python.org/downloads/release/python-3127/](https://www.python.org/downloads/release/python-3127/)

**2. Doğru dosyayı bulun:**
Sayfanın en altına kadar kaydırın. **"Files"** başlığı altındaki tablodan şu dosyayı bulup tıklayın:
```
macOS 64-bit universal2 installer
```
*(Dosya adı: `python-3.12.7-macos11.pkg`, boyutu yaklaşık ~45 MB)*

**3. İndirilen .pkg dosyasını açın:**
İndirilenler (Downloads) klasörünüze gidin ve `python-3.12.7-macos11.pkg` dosyasına çift tıklayın.

**4. Kurulum sihirbazını takip edin:**
Açılan pencerede sırasıyla:
- **"Devam Et (Continue)"** → tıklayın
- **"Devam Et (Continue)"** → tıklayın (lisans sözleşmesi)
- **"Kabul Et (Agree)"** → tıklayın
- **"Yükle (Install)"** → tıklayın
- Mac şifrenizi girin → **"Yazılım Yükle (Install Software)"** → tıklayın
- **"Kapat (Close)"** → tıklayın

**5. Kurulumu doğrulayın:**
Terminal'i açın (Spotlight'a "Terminal" yazarak bulabilirsiniz) ve şu komutu yazın:
```bash
python3.12 --version
```
Ekranda `Python 3.12.7` yazısını görüyorsanız kurulum başarılıdır. ✅

> [!NOTE]
> macOS'te `python3 --version` yazdığınızda eski bir sürüm (örn. 3.9 veya 3.11) görünebilir. Bu normaldir. Mac'ler eski sürümleri silmez, yeni sürümü yanına ekler. Bizim projemiz `python3.12` komutunu kullandığı için eski sürümlerden etkilenmez.

---

### 🪟 Windows İçin Adım Adım Kurulum

**1. İndirme sayfasını açın:**
Tarayıcınızda şu linke tıklayın:
👉 [https://www.python.org/downloads/release/python-3127/](https://www.python.org/downloads/release/python-3127/)

**2. Doğru dosyayı bulun:**
Sayfanın en altına kadar kaydırın. **"Files"** başlığı altındaki tablodan şu dosyayı bulup tıklayın:
```
Windows installer (64-bit)
```
*(Dosya adı: `python-3.12.7-amd64.exe`, boyutu yaklaşık ~25 MB)*

**3. İndirilen .exe dosyasını açın:**
İndirilenler (Downloads) klasörünüze gidin ve `python-3.12.7-amd64.exe` dosyasına çift tıklayın.

**4. Kurulum ekranında dikkatli olun:**

> [!IMPORTANT]
> Kurulumun **ilk ekranında** en altta iki kutucuk vardır. **"Add python.exe to PATH"** kutucuğunu kesinlikle işaretleyin! Bu işaretlenmezse Terminal'den Python komutları çalışmaz.

Kutucuğu işaretledikten sonra:
- **"Install Now"** → tıklayın
- Kurulumun bitmesini bekleyin (birkaç dakika sürebilir)
- **"Close"** → tıklayın

**5. Kurulumu doğrulayın:**
Başlat menüsünden **"PowerShell"** veya **"Komut İstemi (CMD)"** açın ve şu komutu yazın:
```powershell
python --version
```
veya
```powershell
py -3.12 --version
```
Ekranda `Python 3.12.7` yazısını görüyorsanız kurulum başarılıdır. ✅

---

> [!TIP]
> Python kurulumu tamamlandıktan sonra bu rehberin **Adım 1**'e geri dönüp projeyi kurmaya devam edebilirsiniz.

---

## 🔧 Sorun Giderme: Akıllı Veri Artırımı Arayüzü Hataları

Eğer tüm paketler hatasız kurulduğu halde **Akıllı Veri Artırımı** arayüzünde aşağıdaki sorunlardan biri yaşanıyorsa, bu bölümdeki adımları uygulayın:

- Seed verisi **0** gösteriyor
- RCGAN modeli ismi görünmüyor (sadece "Bağlandı" yazıyor)
- CSV yükleyince **"There was an error parsing the body"** hatası alınıyor

---

### Düzeltme 1: Seed ve Model Dosyası Algılanmıyor (Seed 0 Sorunu)

**Sebep:** macOS'un iCloud dosya sistemi bazen gerçek dosyaları "placeholder" olarak işaretler. Kod bu dosyaları atlayarak "yok" sayar.

**Dosya:** `akilli_veri_arttirimi/backend/server.py`

**Satır 196-215** arasındaki `get_local_asset_path` fonksiyonunu bulun ve **tamamını** şu şekilde değiştirin:

```python
def get_local_asset_path(path):
    """Placeholder varlıklarda aynı yapıya sahip yerel çalışma kopyasını kullan."""
    candidates = [path]
    try:
        relative = os.path.relpath(path, PROJECT_DIR)
        if not relative.startswith(".."):
            candidates.append(os.path.join(LOCAL_MIRROR_PROJECT_DIR, relative))
    except ValueError:
        pass

    for candidate in candidates:
        try:
            if not os.path.isfile(candidate):
                continue
            size = os.path.getsize(candidate)
            if size <= 0 or is_git_lfs_pointer(candidate):
                continue
            # Dosya 1 MB'dan büyükse gerçek veridir, iCloud kontrolüne gerek yok
            if size > 1024 * 1024:
                return candidate
            if is_dataless_file(candidate):
                continue
            return candidate
        except OSError:
            continue
    return None
```

**Ne değişti:** Dosya boyutu 1 MB'dan büyükse (ki seed dosyası ~700 MB, model dosyası ~360 MB) iCloud kontrolünü atlayarak dosyayı doğrudan kabul eder. Böylece macOS'un yanlış işaretlemesi sorunu ortadan kalkar.

---

### Düzeltme 2: CSV Yüklerken "Error Parsing the Body" Hatası

**Sebep:** Masaüstü uygulaması (pywebview) HTML'i doğrudan bellekte yüklediği için tarayıcının origin değeri `null` olur. Bazı macOS sürümlerinde WebKit bu origin ile dosya yükleme isteğini farklı formatlayabilir.

**Dosya:** `akilli_veri_arttirimi/main.py`

**Satır 146-155** arasındaki `webview.create_window` bloğunu bulun ve şu şekilde değiştirin:

```python
    window = webview.create_window(
        "Sentetik Veri Üretim Hattı",
        url=f"http://{HOST}:{PORT}",
        js_api=DesktopApi(),
        width=1440,
        height=920,
        min_size=(1100, 720),
    )
    window.events.closed += app.stop_backend
    webview.start(gui="cocoa", debug=False)
```

**Ne değişti:** `html=load_desktop_html()` yerine `url=f"http://{HOST}:{PORT}"` kullanarak pencereyi doğrudan backend sunucusuna bağlarız. Bu sayede origin `null` yerine `http://127.0.0.1:8000` olur ve dosya yükleme istekleri sorunsuz çalışır.

> [!IMPORTANT]
> Bu değişikliği yaptığınızda `load_desktop_html` fonksiyonu artık kullanılmaz; silmenize gerek yok, sadece çağrılmayacaktır.

---

### Düzeltmeleri Uyguladıktan Sonra

Dosyaları kaydedip projeyi tekrar başlatın:
```bash
python main_launcher.py
```
Akıllı Veri Artırımı butonuna tıkladığınızda seed verisi satır sayısını göstermeli, model ismi görünmeli ve CSV yükleme sorunsuz çalışmalıdır.
