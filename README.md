# Şerit Takibi
Bu repo LaneATT modelinin bizim kullanımımıza göre şekillendirilmiş formudur.

## Yükleme
* Düzgün bir şekilde pytorch ve onunla uyumlu CUDA sürümünü kurmanız gerekiyor. 
* requirements.txt'de belirtilen kütüphaneleri indirmeniz gerekiyor.
* nms build'i yapmanız gerekiyor.
* **Orijinal repo'daki readme'de detaylı şekilde kurulum anlatılmış**

## Test
ika: Ben linux sistemde anaconda'ya python 3.11.11 kurup içine CUDA 12.4 ve pytorch 2.6 kurarak test ettim şuan sorunsuz çalışıyor.

* Uygun değişiklikleri yaptıktan sonra test_run.py dosyasını çalıştırmanız yeterli olucaktır.
  <br>* OpenCV Kamera bağlantısı Yapmalısınız: Ben ip webcam kullandım (android uygulaması) 
  <br>* Ya da fotoğraf çekme kısmını yoruma alıp kendiniz fotoğrafı değiştirmelisiniz.

* Test_run.py'deki yorumu okuyun.
