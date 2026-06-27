# Kullanılan Kütüphaneler

Bu dosya, proje kökündeki `requirements.txt` ve `akilli_veri_arttirimi/requirements.txt` dosyalarında yer alan Python kütüphanelerini özetler. Her açıklama en fazla iki cümle olacak şekilde kısa tutulmuştur.

| Kütüphane | Sürüm | Açıklama |
| --- | --- | --- |
| PySide6 | 6.11.1 | Qt tabanlı masaüstü arayüzleri geliştirmek için kullanılır. Projedeki launcher ve GUI ekranlarının temelini oluşturur. |
| PySide6_Addons | 6.11.1 | PySide6 için ek Qt modüllerini sağlar. Gelişmiş arayüz bileşenleri ve yardımcı özellikler için kullanılır. |
| PySide6_Essentials | 6.11.1 | PySide6'nın temel Qt modüllerini içerir. Uygulamanın çekirdek pencere, işlem ve widget özelliklerini destekler. |
| shiboken6 | 6.11.1 | PySide6'nın Python ile Qt/C++ bağlamasını sağlayan altyapı paketidir. PySide6'nın çalışması için gereklidir. |
| torch | 2.12.0 | PyTorch derin öğrenme çatısıdır. RCGAN modelleri, sinir ağları ve tensor işlemleri için kullanılır. |
| torchvision | 0.27.0 | PyTorch ekosisteminde görüntü işleme ve görsel model araçları sağlar. Görüntü tabanlı eğitim ve dönüştürme işlerinde destek paketidir. |
| ultralytics | 8.4.48 | YOLO modelleriyle nesne tespiti ve analiz yapmak için kullanılır. Dedektör pipeline tarafındaki görsel değerlendirme işlerini destekler. |
| ultralytics-thop | 2.0.20 | Ultralytics modellerinde hesaplama maliyeti ve model karmaşıklığı ölçümleri için kullanılır. FLOPs ve parametre analizi gibi metrikleri destekler. |
| opencv-python | 4.13.0.92 | OpenCV'nin temel görüntü işleme fonksiyonlarını sağlar. Görsellerin okunması, yazılması ve işlenmesi için kullanılır. |
| opencv-contrib-python | 4.13.0.92 | OpenCV'nin ek katkı modüllerini içerir. Özellikle `cv2.dnn_superres` gibi gelişmiş görüntü büyütme özellikleri için gereklidir. |
| transformers | 5.12.1 | Hugging Face model ekosistemindeki transformer tabanlı modelleri çalıştırmak için kullanılır. Metin veya çok modlu model entegrasyonlarında altyapı sağlar. |
| safetensors | 0.8.0 | Model ağırlıklarını güvenli ve hızlı biçimde saklamak/yüklemek için kullanılır. Hugging Face ve derin öğrenme iş akışlarında tercih edilir. |
| accelerate | 1.14.0 | Model eğitimi ve çıkarımını farklı donanımlarda kolaylaştırır. CPU, GPU ve çoklu cihaz kurulumlarını yönetmeye yardımcı olur. |
| huggingface_hub | 1.16.1 | Hugging Face Hub üzerinden model ve dosya indirme/yükleme işlemlerini sağlar. Model depolarıyla entegrasyon için kullanılır. |
| tokenizers | 0.22.2 | Hızlı metin parçalama ve tokenizer işlemleri için kullanılır. Transformers paketinin metin ön işleme altyapısını destekler. |
| hf-xet | 1.5.1 | Hugging Face dosya aktarım ve depolama altyapısını hızlandıran yardımcı pakettir. Büyük model veya veri dosyalarında indirme süreçlerini destekler. |
| matplotlib | 3.9.0 | Grafik ve görselleştirme üretmek için kullanılır. Analiz sonuçları, metrikler ve karşılaştırma grafikleri bu paketle çizilir. |
| numpy | 2.4.6 | Sayısal hesaplama ve çok boyutlu dizi işlemleri için temel pakettir. Veri işleme, model girdi hazırlığı ve matematiksel hesaplarda kullanılır. |
| pandas | 2.3.3 | Tablo verilerini okumak, dönüştürmek ve analiz etmek için kullanılır. CSV tabanlı veri setleri ve sentetik veri işlemleri için ana araçtır. |
| pillow | 12.2.0 | Python'da görüntü dosyalarını açmak, dönüştürmek ve kaydetmek için kullanılır. GUI ve üretim pipeline içindeki görsel işlemleri destekler. |
| tqdm | 4.68.2 | Döngüler ve uzun süren işlemler için ilerleme çubuğu sağlar. Eğitim, üretim ve toplu işlem adımlarını takip etmeyi kolaylaştırır. |
| scipy | 1.17.1 | Bilimsel hesaplama, uzaklık ölçümleri ve istatistiksel fonksiyonlar sağlar. Projede benzerlik, dağılım ve matematiksel analizlerde kullanılır. |
| scikit-learn | 1.9.0 | Makine öğrenmesi algoritmaları ve değerlendirme metrikleri sağlar. Sınıflandırma, ölçekleme, veri bölme ve model karşılaştırma için kullanılır. |
| h5py | 3.14.0 | HDF5 dosya formatını Python'da okumak ve yazmak için kullanılır. Büyük sayısal veri ve model dosyalarıyla çalışmayı destekler. |
| fastapi | 0.137.1 | Modern ve hızlı Python API servisleri geliştirmek için kullanılır. `akilli_veri_arttirimi` backend sunucusunun temel web çatısıdır. |
| uvicorn | 0.49.0 | FastAPI gibi ASGI uygulamalarını çalıştıran web sunucusudur. Backend API'nin yerelde veya sunucuda ayağa kalkmasını sağlar. |
| pydantic | 2.13.4 | Veri doğrulama ve tip tabanlı modelleme için kullanılır. FastAPI istek/yanıt şemalarını ve veri kontrollerini destekler. |
| pydantic_core | 2.46.4 | Pydantic'in yüksek performanslı doğrulama çekirdeğidir. Pydantic paketinin çalışma zamanı bağımlılığıdır. |
| starlette | 1.0.0 | FastAPI'nin üzerine kurulduğu ASGI web çatısıdır. Routing, middleware ve response altyapısını sağlar. |
| anyio | 4.13.0 | Farklı async çalışma modelleri arasında uyumluluk sağlar. FastAPI/Starlette tarafındaki eşzamanlı işlemleri destekler. |
| h11 | 0.16.0 | HTTP/1.1 protokolü için düşük seviyeli Python uygulamasıdır. Uvicorn ve HTTP istemci/sunucu altyapısında kullanılır. |
| python-multipart | 0.0.28 | Multipart form ve dosya yükleme isteklerini ayrıştırır. FastAPI'de dosya upload endpointleri için gereklidir. |
| requests | 2.34.2 | HTTP istekleri yapmak için kullanılır. Uygulamanın backend kontrolü ve servislerle iletişim kurması için destek sağlar. |
| urllib3 | 2.6.3 | HTTP bağlantı havuzu ve düşük seviyeli istemci işlevleri sunar. Requests paketinin temel bağımlılıklarından biridir. |
| certifi | 2026.4.22 | Güvenilir CA sertifika paketini sağlar. HTTPS bağlantılarının güvenli doğrulanmasına yardımcı olur. |
| charset-normalizer | 3.4.7 | Metin karakter kodlamasını tespit etmek ve normalize etmek için kullanılır. Requests'in yanıt metinlerini doğru işlemesini destekler. |
| idna | 3.13 | Uluslararası alan adlarının ASCII uyumlu biçime çevrilmesini sağlar. HTTP bağlantılarında alan adı işleme için kullanılır. |
| httpcore | 1.0.9 | HTTPX için düşük seviyeli HTTP çekirdeğidir. Senkron ve asenkron HTTP bağlantı yönetimini destekler. |
| httpx | 0.28.1 | Modern senkron/asenkron HTTP istemcisidir. Harici servislerle API tabanlı iletişim kurmak için kullanılabilir. |
| ctgan | 0.12.1 | Tabular veriler için koşullu GAN tabanlı sentetik veri üretimi sağlar. Projenin sentetik veri artırımı hedefini destekler. |
| rdt | 1.21.0 | Verileri makine öğrenmesi modellerine uygun biçime dönüştüren yardımcı pakettir. CTGAN ve sentetik veri üretim süreçlerinde kullanılır. |
| Faker | 40.15.0 | Gerçeğe benzer sahte veri üretmek için kullanılır. Sentetik veri setlerinde isim, tarih veya benzeri alanları üretmeye yardımcı olur. |
| tensorflow | 2.21.0 | Derin öğrenme modelleri geliştirmek ve çalıştırmak için kullanılan çatıdır. Waymo entegrasyonu ve bazı deneysel pipeline adımlarında kullanılır. |
| keras | 3.14.0 | Yüksek seviyeli sinir ağı API'sidir. TensorFlow ekosisteminde model tanımlama ve eğitim süreçlerini kolaylaştırır. |
| absl-py | 2.4.0 | Abseil Python yardımcı kütüphanesidir. TensorFlow ve ilgili araçlarda bayrak, logging ve yardımcı altyapı için kullanılır. |
| astunparse | 1.6.3 | Python AST yapılarını tekrar kaynak koda dönüştürmek için kullanılır. TensorFlow'un bazı dönüştürme ve uyumluluk işlemlerinde bağımlılıktır. |
| flatbuffers | 25.12.19 | Verileri hızlı ve taşınabilir biçimde serileştirmek için kullanılır. TensorFlow Lite ve model formatları tarafında destek sağlar. |
| gast | 0.7.0 | Python AST işlemleri için uyumluluk katmanı sağlar. TensorFlow'un grafik dönüştürme altyapısında kullanılır. |
| google-pasta | 0.2.0 | Python kodunu ayrıştırma ve yeniden yazma işlemlerinde kullanılır. TensorFlow/Keras dönüşüm araçlarının bağımlılıklarındandır. |
| grpcio | 1.80.0 | gRPC protokolüyle yüksek performanslı servis iletişimi sağlar. TensorFlow ve dağıtık sistem bileşenlerinde destek paketidir. |
| libclang | 18.1.1 | Clang derleyici altyapısına Python erişimi sağlar. TensorFlow kurulumlarında bazı yerel bağlama ve analiz işlerinde bağımlılık olarak bulunur. |
| ml_dtypes | 0.5.4 | Makine öğrenmesi için özel sayısal veri tipleri sağlar. TensorFlow ve JAX ekosistemlerindeki dtype ihtiyaçlarını destekler. |
| opt_einsum | 3.4.0 | Tensor işlemlerinde einsum ifadelerini optimize eder. Derin öğrenme hesaplamalarının daha verimli çalışmasına yardımcı olur. |
| optree | 0.19.0 | İç içe veri yapılarını ağaç gibi işlemek için kullanılır. Keras ve model eğitim altyapısında parametre yapılarını yönetmeye yardımcı olur. |
| protobuf | 7.34.1 | Protocol Buffers veri serileştirme formatını sağlar. TensorFlow model tanımları ve birçok servis iletişiminde kullanılır. |
| termcolor | 3.3.0 | Terminal çıktılarında renkli metin üretmek için kullanılır. Eğitim ve log çıktılarının okunabilirliğini artırabilir. |
| wrapt | 2.1.2 | Fonksiyon ve sınıf sarmalama işlemleri için yardımcı pakettir. TensorFlow gibi paketlerde dekoratör ve uyumluluk altyapısı olarak kullanılır. |
| pywebview | 6.2.1 | Web tabanlı arayüzleri yerel masaüstü penceresinde göstermeyi sağlar. `akilli_veri_arttirimi` uygulamasının web arayüzünü desktop deneyimine taşır. |
| bottle | 0.13.4 | Hafif bir Python web çatısıdır. Pywebview ekosisteminde yerel servis veya yardımcı web katmanı olarak kullanılır. |
| proxy_tools | 0.1.0 | Proxy nesneleri ve yönlendirme yardımcıları sağlar. Pywebview'in bazı platform entegrasyonlarında bağımlılık olarak kullanılır. |
| pyobjc-core | 12.2 | macOS üzerinde Objective-C köprüleme altyapısını sağlar. Pywebview'in macOS yerel pencere özellikleri için gereklidir. |
| pyobjc-framework-Cocoa | 12.2 | macOS Cocoa framework'üne Python erişimi sağlar. macOS'ta yerel pencere ve uygulama davranışlarını destekler. |
| pyobjc-framework-Quartz | 12.2 | macOS Quartz grafik framework'üne Python erişimi sağlar. Ekran, görüntü ve pencere işlemlerinde platform desteği sunar. |
| pyobjc-framework-Security | 12.2 | macOS Security framework'üne Python erişimi sağlar. Güvenlik ve sertifika işlemlerinde platform bağımlılığı olarak kullanılır. |
| pyobjc-framework-UniformTypeIdentifiers | 12.2 | macOS dosya türü tanımlama framework'üne erişim sağlar. Dosya seçme ve tür belirleme işlemlerinde destek olur. |
| pyobjc-framework-WebKit | 12.2 | macOS WebKit framework'üne Python erişimi sağlar. Pywebview'in macOS üzerinde web içerik göstermesini destekler. |
| annotated-doc | 0.0.4 | Tip anotasyonlarından dokümantasyon üretmeyi destekleyen yardımcı pakettir. FastAPI/Pydantic ekosisteminde açıklama metadata'sı için kullanılabilir. |
| annotated-types | 0.7.0 | Python tiplerine ek doğrulama kısıtları eklemeyi sağlar. Pydantic veri doğrulama süreçlerinde kullanılır. |
| click | 8.3.3 | Komut satırı arayüzleri geliştirmek için kullanılır. Uvicorn, Typer ve bazı araçların CLI altyapısını destekler. |
| contourpy | 1.3.3 | Kontur çizimleri için sayısal hesaplama altyapısı sağlar. Matplotlib grafik üretiminde bağımlılık olarak kullanılır. |
| cycler | 0.12.1 | Matplotlib'de renk ve stil döngülerini yönetir. Grafiklerin otomatik stil sıralamasını destekler. |
| filelock | 3.29.0 | Dosya tabanlı kilitleme mekanizması sağlar. Model ve cache dosyalarının eşzamanlı erişiminde kullanılır. |
| fonttools | 4.62.1 | Font dosyalarını işlemek ve dönüştürmek için kullanılır. Matplotlib'in yazı tipi yönetimi tarafında bağımlılıktır. |
| fsspec | 2026.4.0 | Yerel ve uzak dosya sistemleri için ortak erişim arayüzü sağlar. Veri dosyaları ve model kaynaklarına esnek erişimi destekler. |
| Jinja2 | 3.1.6 | Şablon dosyalarından dinamik metin veya HTML üretmek için kullanılır. Web ve dokümantasyon araçlarında yaygın bir bağımlılıktır. |
| joblib | 1.5.3 | Python nesnelerini kaydetme ve paralel görev çalıştırma araçları sağlar. Scikit-learn modellerinin saklanması ve hızlı çalışması için kullanılır. |
| kiwisolver | 1.5.0 | Kısıt çözme algoritması sağlayan yardımcı pakettir. Matplotlib'in layout hesaplamalarında kullanılır. |
| markdown-it-py | 4.0.0 | Markdown metinlerini ayrıştırmak için kullanılır. Rich ve dokümantasyon çıktılarında Markdown işleme desteği sağlar. |
| MarkupSafe | 3.0.3 | HTML/XML çıktılarında güvenli string işleme sağlar. Jinja2 şablon motorunun güvenli çıktı üretmesini destekler. |
| mdurl | 0.1.2 | Markdown içindeki URL'leri ayrıştırmak için kullanılır. Markdown-it-py paketinin bağımlılığıdır. |
| mpmath | 1.3.0 | Keyfi hassasiyetli matematiksel hesaplama sağlar. SymPy'nin sayısal hesaplama altyapısında kullanılır. |
| namex | 0.1.0 | Nesne adlandırma ve serileştirme yardımcıları sağlar. Keras ekosisteminde model bileşenlerinin isimlendirilmesini destekler. |
| narwhals | 2.22.1 | Farklı dataframe kütüphaneleri arasında ortak arayüz sağlar. Veri işleme araçlarının pandas ve polars gibi yapılarla uyumunu artırır. |
| networkx | 3.6.1 | Grafik ve ağ veri yapılarıyla çalışmak için kullanılır. PyTorch ve model analiz bağımlılıklarında destek paketidir. |
| packaging | 26.2 | Python paket sürümlerini ve uyumluluk kurallarını işler. Birçok kütüphanenin sürüm kontrolü ve kurulum mantığında kullanılır. |
| polars | 1.41.2 | Yüksek performanslı dataframe işlemleri için kullanılır. Büyük tablo verilerinde hızlı analiz ve dönüşüm olanağı sağlar. |
| polars-runtime-32 | 1.41.2 | Polars'ın çalışma zamanı bileşenlerini içerir. Polars paketinin platforma uygun şekilde çalışmasını destekler. |
| psutil | 7.2.2 | Sistem kaynaklarını ve süreç bilgilerini okumak için kullanılır. CPU, bellek ve çalışan işlem takibinde yardımcıdır. |
| Pygments | 2.20.0 | Kod ve terminal çıktıları için renklendirme sağlar. Rich ve dokümantasyon araçlarında sözdizimi vurgulama için kullanılır. |
| pyparsing | 3.3.2 | Metin ayrıştırma gramerleri oluşturmak için kullanılır. Matplotlib ve packaging gibi paketlerin ayrıştırma işlemlerini destekler. |
| python-dateutil | 2.9.0.post0 | Tarih ve zaman ayrıştırma işlemlerini kolaylaştırır. Pandas ve zaman serisi işlemlerinde yaygın olarak kullanılır. |
| pytz | 2026.2 | Zaman dilimi verileri ve dönüşümleri sağlar. Pandas tarih/saat işlemlerinde uyumluluk için kullanılır. |
| PyYAML | 6.0.3 | YAML dosyalarını okumak ve yazmak için kullanılır. Model, veri seti veya konfigürasyon dosyalarının işlenmesini destekler. |
| regex | 2026.5.9 | Python'un standart regex modülünden daha gelişmiş düzenli ifade özellikleri sağlar. Tokenizer ve metin işleme bağımlılıklarında kullanılır. |
| rich | 15.0.0 | Terminalde zengin metin, tablo ve ilerleme çıktıları üretir. CLI araçlarının daha okunabilir çıktı vermesini sağlar. |
| setuptools | 81.0.0 | Python paket kurulum ve dağıtım araçlarını sağlar. Bazı bağımlılıkların kurulumu ve paket metadata işlemleri için gereklidir. |
| shellingham | 1.5.4 | Çalışan kabuk ortamını tespit eder. Typer gibi CLI araçlarında shell uyumunu belirlemek için kullanılır. |
| six | 1.17.0 | Python 2/3 uyumluluk yardımcıları sağlar. Projede bazı eski API bağımlılıkları ve `six.moves` kullanımı için bulunur. |
| sympy | 1.14.0 | Sembolik matematik işlemleri için kullanılır. PyTorch bağımlılıklarında matematiksel ifade ve şekil hesaplarını destekler. |
| threadpoolctl | 3.6.0 | Sayısal kütüphanelerin thread havuzlarını kontrol eder. Scikit-learn ve NumPy tabanlı işlemlerde performans yönetimine yardımcı olur. |
| typer | 0.26.7 | Tip anotasyonlarıyla komut satırı arayüzleri geliştirmeyi sağlar. Ultralytics ve bazı yardımcı araçların CLI deneyimini destekler. |
| typing_extensions | 4.15.0 | Yeni Python tip özelliklerini eski/uyumlu sürümlere taşır. Pydantic, FastAPI ve birçok modern pakette tip desteği için kullanılır. |
| typing-inspection | 0.4.2 | Python tip anotasyonlarını çalışma zamanında incelemeye yardımcı olur. Pydantic'in gelişmiş tip çözümleme süreçlerini destekler. |
| tzdata | 2026.2 | IANA zaman dilimi verilerini Python ortamına sağlar. Pandas ve tarih/saat işlemlerinde güncel timezone bilgisi sunar. |
| google.colab | Colab ortamı | Google Colab defterlerinde Drive bağlama ve çalışma ortamı entegrasyonu için kullanılır. Bu paket yerel `requirements.txt` içinde değil, Colab ortamında hazır gelen bir modül olarak kullanılır. |
