# Dikte Original

Yerel macOS dikte uygulaması. Model sabittir:

- Kaynak: <https://huggingface.co/openai/whisper-large-v3>
- Dosya: depoda yayımlanan özgün `model.safetensors`
- Mimari: tam large-v3, 32 encoder + 32 decoder katmanı
- Çalıştırma: PyTorch + Transformers, Apple Silicon'da MPS
- Dönüşüm: yok (Core ML, MLX, ONNX yok)
- Kuantizasyon: yok (resmî FP16 safetensors doğrudan yüklenir)
- Ağ/API: model ilk indirmeden sonra tamamen yerel çalışır; OpenAI API kullanılmaz

## Kurulum ve çalıştırma

```sh
cd /Users/anarmammadov/Documents/Dosyalarim/speechToText-1
xcodegen generate
open DikteOriginal.xcodeproj
```

İlk açılışta uygulama kendi Python ortamını
`~/Library/Application Support/DikteOriginal/runtime` altında kurar. Ardından
resmî model Hugging Face'den
`~/Library/Application Support/DikteOriginal/HuggingFace` altına indirilir.
Bu ilk kurulum internet hızına göre uzun sürebilir. Sonraki açılışlarda aynı
dosyalar doğrudan yerel diskten yüklenir. Uygulama kayıt, çeviri veya metin
yapıştırma yapmadan 90 saniye boşta kalırsa model worker'ını kapatıp PyTorch/MPS
belleğini serbest bırakır. Yeni kayıt anında başlar; model kayıt sürerken yerel
diskten yeniden yüklenir ve kayıt bittikten sonra çeviri yapılır.

Model hazır olduğunda Ayarlar → Model ekranında kaynak URL'si, safetensors
dosyasının gerçek yolu/boyutu, 32+32 katman, PyTorch ve Transformers sürümleri
gösterilir.

## Kullanım

Varsayılan kısayol `⌃⌘` tuşlarını basılı tutmaktır. Konuşma boyunca basılı tut;
bırakınca ses yerel modele gönderilir ve metin aktif uygulamaya yapıştırılır.

Mikrofon, Girdi İzleme ve Erişilebilirlik izinlerinin üçü de gereklidir.
