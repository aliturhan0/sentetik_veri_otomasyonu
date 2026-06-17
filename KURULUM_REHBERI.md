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
Kurulumun başarılı olup olmadığını ve bilgisayarınızda hangi sürümün aktif olduğunu görmek için Terminal (veya PowerShell) ekranına şunu yazın:
```bash
python3 --version
```
*(Windows için `python --version` şeklinde de yazabilirsiniz.)* Ekranda `Python 3.12.x` çıktısını görüyorsanız her şey hazır demektir.

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
