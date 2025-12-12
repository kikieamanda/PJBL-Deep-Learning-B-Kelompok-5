# PJBL-Deep-Learning-B-Kelompok-5 #

# “Face Expression Recognition With Convolutional Neural Network (CNN)” #

# DISUSUN OLEH KELOMPOK “V” : #

1. Muhammad Aryasatya Nugroho	( 22083010085 ) - KETUA 

2. Winna Dhestyan Putri	( 22083010015 ) - ANGGOTA 

3. Rizki Amanda	( 22083010045 ) - ANGGOTA 

4. Jasmine Aulia	( 22083010074 ) - ANGGOTA

5. Hanif Ziva Zavira	( 22083010078 ) - ANGGOTA 

# DOSEN PENGAMPU: #

Amri Muhaimin, S.Stat., M.Stat., M.S.  ( 199507232024061002) 


# BAB I #
# PENDAHULUAN #

# 1.1 Latar Belakang #
<p align="justify">
Perkembangan teknologi kecerdasan buatan mendorong peningkatan kebutuhan sistem yang mampu memahami kondisi emosional manusia melalui ekspresi wajah. Ekspresi wajah merupakan salah satu bentuk komunikasi non-verbal paling penting yang mencerminkan keadaan emosional seperti senang, sedih, marah, terkejut, dan takut. Dalam berbagai konteks digital seperti layanan kesehatan mental, pendidikan, keamanan, dan interaksi manusia dengan komputer, kemampuan komputer untuk mengidentifikasi ekspresi wajah secara otomatis sangat dibutuhkan untuk meningkatkan kualitas layanan dan responsivitas sistem.

<p align="justify">
Meskipun demikian, proses pengenalan ekspresi wajah bukanlah tugas yang sederhana. Tantangan muncul karena setiap individu memiliki karakteristik wajah berbeda, termasuk struktur wajah, usia, ras, dan variasi anatomi lainnya, yang membuat model sulit melakukan generalisasi secara optimal. Faktor eksternal seperti variasi pencahayaan, perubahan sudut pandang, pose wajah yang tidak frontal, hingga oklusi seperti masker atau tangan, turut mempengaruhi akurasi sistem dalam mengenali ekspresi pada citra atau video. Ketidakseimbangan jumlah data antar kategori emosi juga menjadi masalah signifikan, karena ekspresi tertentu biasanya muncul lebih sering dibandingkan ekspresi lain sehingga menimbulkan bias pada model.

<p align="justify">
Untuk mengatasi tantangan tersebut, metode berbasis deep learning, khususnya Convolutional Neural Network (CNN) banyak digunakan karena memiliki kemampuan untuk mengekstraksi fitur spasial pada citra secara otomatis. CNN mampu menangkap pola visual seperti kontur, tekstur, dan perbedaan intensitas piksel yang menjadi indikator ekspresi wajah manusia, tanpa membutuhkan proses ekstraksi fitur manual. Berbagai studi menunjukkan bahwa CNN mampu memberikan performa yang kompetitif dalam tugas pengenalan ekspresi wajah ketika dipadukan dengan teknik pra-pemrosesan seperti normalisasi citra, augmentasi, dan deteksi wajah.

<p align="justify">
Melihat potensi tersebut, proyek deteksi ekspresi wajah ini dibangun dengan memanfaatkan arsitektur CNN untuk mengklasifikasikan emosi berdasarkan citra wajah. Dengan menggunakan dataset terstruktur yang memiliki label emosi, model dapat dilatih untuk mengidentifikasi ekspresi secara otomatis. Sistem ini diharapkan mampu membantu proses analisis emosi pada berbagai aplikasi, misalnya pemantauan kondisi psikologis, sistem pembelajaran adaptif, maupun antarmuka cerdas yang mampu merespons kondisi emosional pengguna secara dinamis.

<p align="justify">
Selain itu, proyek ini disertai proses evaluasi model secara menyeluruh menggunakan metrik seperti akurasi, precision, recall, dan F1-score untuk memastikan bahwa model tidak hanya bekerja baik pada data pelatihan tetapi juga mampu menghadapi variasi kondisi nyata. Evaluasi ini memberi gambaran mengenai kemampuan model dalam menghadapi tantangan seperti generalisasi, robust terhadap noise, serta pengaruh imbalance data. Dengan demikian, proyek ini diharapkan tidak hanya menghasilkan model deteksi ekspresi wajah, tetapi juga memberikan pemahaman ilmiah mengenai efektivitas CNN dalam menangani tugas pengenalan emosi berbasis citra.

# 1.2 Rumusan Masalah # 
<p align="justify">
1. Bagaimana merancang model CNN yang mampu melakukan klasifikasi ekspresi wajah secara akurat berdasarkan citra?

<p align="justify">
2. Bagaimana cara mengatasi masalah ketidakseimbangan data antar kategori emosi?

<p align="justify">
3. Bagaimana evaluasi performa model secara kuantitatif menggunakan matrik evaluasi yang tepat?

# 1.3 Tujuan #
<p align="justify">
1. Mengembangkan model CNN untuk mendeteksi dan mengklasifikasikan ekspresi wajah.

<p align="justify">
2. Mengoptimalkan proses pelatihan melalui teknik pra-pemrosesan dan augmentasi data.

<p align="justify">
3. Menguji performa model terhadap berbagai kondisi citra dan variasi ekspresi.

<p align="justify">
4. Melakukan evaluasi model menggunakan akurasi, precision, recall, dan F1-score.

<p align="justify">
5. Menghasilkan sistem yang dapat digunakan sebagai dasar pengembangan aplikasi realtime berbasis pengenalan ekspresi wajah.

# 1.4 Manfaat #
<p align="justify">
1. Manfaat Teoretis: Memberikan kontribusi pada bidang computer vision dan FER terkait penggunaan CNN untuk pengenalan emosi.

<p align="justify">
2. Manfaat Praktis: Menjadi dasar pengembangan aplikasi berbasis analisis emosional seperti sistem pembelajaran adaptif, layanan kesehatan mental, dan teknologi interaksi manusia-mesin. Memberikan model yang dapat digunakan untuk deteksi ekspresi wajah secara otomatis sesuai kebutuhan industri dan penelitian.

<p align="justify">
3. Manfaat Pengembangan Sistem: Memberikan evaluasi empiris yang dapat digunakan untuk perbaikan model di masa mendatang, terutama dalam mengatasi noise, pose, dan ketidakseimbangan data.

# BAB II #
# TINJAUAN PUSTAKA #
# 2.1 Deteksi Ekspresi Wajah (Facial Expression Recognition) # 
<p align="justify">
Deteksi ekspresi wajah (facial expression recognition, FER) merupakan salah satu cabang penting dalam computer vision yang bertujuan mengidentifikasi keadaan emosional manusia berdasarkan citra wajah. FER berperan dalam berbagai bidang seperti analisis perilaku, interaksi manusia-komputer, kesehatan mental, keamanan, serta sistem adaptif berbasis emosi. Beberapa penelitian awal mengandalkan pendekatan berbasis fitur manual, seperti Local Binary Pattern (LBP) dan Histogram of Oriented Gradients (HOG), yang mengekstraksi tekstur dan pola lokal untuk membedakan ekspresi wajah. Namun, metode ini terbatas karena sensitif terhadap perubahan pencahayaan, pose, dan variasi wajah antar individu sehingga kurang mampu beradaptasi pada kondisi tidak terkontrol.

<p align="justify">
Perkembangan metode pembelajaran mendalam kemudian mendorong FER menuju pendekatan berbasis deep learning yang dapat mempelajari representasi fitur langsung dari data citra. Dataset seperti FER-2013 digunakan secara luas sebagai benchmark karena berisi ribuan sampel ekspresi dasar manusia. Studi-studi terbaru menunjukkan bahwa FER memiliki tantangan signifikan, terutama pada ekspresi yang mirip, pencahayaan buruk, oklusi, serta distribusi data yang tidak seimbang. Kondisi tersebut menyebabkan kesalahan klasifikasi pada ekspresi minoritas atau ekspresi halus. Oleh karena itu, FER berkembang menjadi area penelitian aktif yang menekankan peningkatan ketahanan (robustness) dan kemampuan generalisasi model dalam menghadapi variabilitas kondisi nyata.

# 2.2 Convolutional Neural Network (CNN) dalam FER #
<p align="justify">
Convolutional Neural Network (CNN) merupakan arsitektur deep learning yang paling banyak digunakan untuk mendeteksi ekspresi wajah. CNN mampu mengekstraksi fitur penting secara otomatis melalui lapisan konvolusi dan pooling yang mempelajari pola spasial seperti bentuk mata, alis, bibir, dan struktur wajah lain yang relevan untuk menentukan ekspresi. Keunggulan utama CNN adalah kemampuannya menangani citra raw tanpa memerlukan desain fitur manual, sehingga lebih adaptif terhadap variasi bentuk wajah, pencahayaan, dan pose. Berbagai arsitektur seperti VGG-16, ResNet, dan arsitektur ringan berbasis depthwise convolution telah digunakan untuk meningkatkan akurasi serta efisiensi komputasi dalam tugas.

<p align="justify">
Penelitian menunjukkan bahwa CNN dapat mencapai performa yang kompetitif pada dataset standar. Sebagai contoh, implementasi CNN pada dataset ekspresi wajah menghasilkan akurasi di atas 70%–80% pada beberapa studi. Selain itu, muncul pendekatan memanfaatkan arsitektur ringan seperti MobileNetV2 untuk memungkinkan deteksi ekspresi secara realtime dengan konsumsi komputasi rendah. Meski demikian, performa CNN tetap dipengaruhi oleh ukuran dataset, keberagaman data, kualitas anotasi, dan metode pra pemrosesan. Ketidakseimbangan kelas serta oklusi wajah masih menjadi kendala utama sehingga diperlukan strategi seperti augmentasi data, class weighting, atau oversampling untuk meningkatkan generalisasi model.

# 2.3 Ekspresi Wajah #
<p align="justify">
Ekspresi wajah (Face expressions) merupakan bentuk komunikasi non-verbal yang muncul melalui perubahan pada otot-otot wajah dan digunakan manusia untuk menyampaikan emosi, intensi, serta respons sosial. Secara psikologis, ekspresi wajah dianggap sebagai indikator utama untuk mengenali keadaan emosional seseorang karena sifatnya yang spontan dan sulit dimanipulasi oleh individu. Menurut Paul Ekman, terdapat enam emosi dasar yang dapat dikenali secara universal, yaitu bahagia, sedih, marah, takut, jijik, dan terkejut. Keenam emosi ini mencerminkan respons fisiologis manusia terhadap rangsangan tertentu dan menjadi dasar bagi model komputer untuk melakukan klasifikasi ekspresi wajah. 

<p align="justify">
Teori ini kemudian dikembangkan kembali oleh berbagai peneliti modern yang menemukan bahwa kompleksitas emosi dapat diidentifikasi lebih mendalam melalui pola aktivasi Facial Action Coding System (FACS). FACS memetakan gerakan wajah berdasarkan Action Units (AU), seperti gerakan alis, mulut, mata, dan pipi, sehingga analisis ekspresi wajah menjadi lebih objektif dan terukur. Penelitian terbaru menggarisbawahi dua hal penting, yaitu ekspresi wajah tidak selalu merepresentasikan emosi tunggal dan interpretasi ekspresi dipengaruhi norma budaya setempat sehingga perlu kehati-hatian ketika menggeneralisasi temuan lintas budaya. Dengan demikian, teori ekspresi wajah kontemporer menggabungkan perspektif psikologi dasar, temuan neuropsikologis, dan pendekatan kontekstual/konstruktivis untuk memberikan pemahaman yang lebih komprehensif tentang bagaimana wajah berfungsi sebagai emosional dalam interaksi sosial.
 
# 2.4 Evaluasi Model pada Deteksi Ekspresi Wajah #
<p align="justify">
Evaluasi model pada FER umumnya dilakukan menggunakan metrik klasifikasi seperti akurasi, presisi, recall, F1-score, serta confusion matrix untuk mengetahui pola kesalahan antar kelas. Studi pada dataset CK+48 menunjukkan bahwa implementasi CNN dapat mencapai akurasi rata-rata 83.24%, dengan beberapa kelas seperti “anger” menunjukkan tingkat pengenalan yang lebih tinggi dibanding kelas lain. Hal ini menunjukkan bahwa meskipun CNN mampu menghasilkan performa yang baik pada lingkungan terkontrol, distribusi kesalahan tetap bervariasi antar ekspresi. Evaluasi seperti ini penting untuk memahami kelas mana yang sulit dibedakan model serta bagaimana struktur fitur yang dipelajari.

<p align="justify">
Namun, performa model sering menurun ketika diuji pada kondisi dunia nyata (in-the-wild), misalnya ketika terdapat oklusi, pencahayaan buruk, atau pose non-frontal. Penelitian menunjukkan bahwa model CNN yang dilatih pada dataset bersih dapat turun hingga akurasi 33% pada kondisi tidak terkontrol, termasuk wajah yang tertutup sebagian oleh masker atau benda lain. Hal ini menegaskan pentingnya evaluasi menyeluruh, termasuk pengujian robustnes, pengujian pada data tidak seimbang, dan validasi lintas-domain untuk memastikan performa model tidak terbatas pada data terstruktur saja.

# BAB III #
# METODOLOGI #
# 3.1 Dataset #
<p align="justify">
Dataset merupakan komponen utama dalam pengembangan model deep learning, khususnya pada tugas pengenalan ekspresi wajah. Kualitas dataset, jumlah data, serta konsistensi anotasi sangat berpengaruh terhadap kemampuan model dalam mempelajari pola visual yang merepresentasikan ekspresi emosional manusia.
 
<p align="justify">
Salah satu dataset yang banyak digunakan dalam penelitian pengenalan ekspresi wajah adalah Face Expression Recognition 2013 (FER2013). Dataset ini pertama kali diperkenalkan melalui kompetisi Kaggle “Challenges in Representation Learning: Facial Expression Recognition Challenge” dan hingga saat ini masih menjadi benchmark dalam berbagai penelitian karena ukurannya yang besar serta anotasi kelas yang terstandarisasi.

<p align="justify">
Dataset FER2013 terdiri dari sekitar 36.000 citra wajah beresolusi 48 × 48 piksel dalam format grayscale. Seluruh citra telah melalui proses automatic face registration, sehingga posisi wajah relatif terpusat dan memiliki skala yang seragam. Setiap citra merepresentasikan satu ekspresi wajah yang diklasifikasikan ke dalam tujuh kategori, yaitu Angry, Disgust, Fear, Happy, Sad, Surprise, dan Neutral.

<p align="justify">
Pada penelitian ini digunakan versi dataset FER2013 yang tersedia di platform Kaggle dan telah dikonversi ke dalam format direktori berbasis kelas. Dataset tersebut dibagi ke dalam dua subset utama, yaitu training set dan validation set, dengan proporsi sekitar 80% data untuk pelatihan dan 20% data untuk validasi. Pembagian ini memungkinkan proses evaluasi model selama pelatihan tanpa memerlukan pemisahan data tambahan.
 
<p align="justify">
Distribusi jumlah citra pada masing-masing kelas ekspresi ditunjukkan pada Tabel 3.1. Tabel tersebut memberikan gambaran mengenai komposisi data pada training set dan validation set untuk setiap kategori ekspresi wajah.


<p align="center">
Tabel 3.1 Distribusi Dataset FER2013



<p align="center">
<img width="510" height="257" alt="image" src="https://github.com/user-attachments/assets/e43b212d-4b60-4788-a7d2-c23b9f5237d6" />



<p align="justify">
Berdasarkan distribusi tersebut, dapat dilihat bahwa dataset FER2013 memiliki ketidakseimbangan jumlah data antar kelas, di mana kelas Disgust memiliki jumlah sampel yang jauh lebih sedikit dibandingkan kelas lainnya, seperti Happy dan Neutral. Kondisi ini berpotensi memengaruhi performa model dalam mengenali kelas minoritas, sehingga perlu diperhatikan dalam proses pelatihan, salah satunya melalui penerapan teknik data augmentation.

# 3.2 Alur Penelitian #
<p align="justify">
Alur penelitian ini dirancang untuk menggambarkan tahapan sistematis dalam pengembangan model Convolutional Neural Network (CNN) untuk pengenalan ekspresi wajah. Secara umum, proses penelitian dimulai dari persiapan dataset, dilanjutkan dengan preprocessing data, perancangan dan pelatihan model, hingga evaluasi performa model.

<img width="973" height="412" alt="image" src="https://github.com/user-attachments/assets/9baaa8da-2045-42b2-bfdb-170deeafe7f5" />

<p align="center">
Gambar 3.1 Alur Penelitian

<p align="justify">
Alur pengerjaan penelitian ini disusun berdasarkan tahapan umum dalam proses pengenalan ekspresi wajah menggunakan metode Convolutional Neural Network (CNN). Secara garis besar, proses dimulai dari persiapan data hingga evaluasi performa model. Berikut penjelasan masing-masing langkah:

1. Import Library dan Load Dataset 
<p align="justify">  
Tahap pertama adalah mengimpor pustaka yang diperlukan seperti NumPy, Pandas, Matplotlib, dan TensorFlow/Keras. Pada tahap ini juga dilakukan pemanggilan dataset dari direktori train, validation, dan test yang telah disediakan oleh Kaggle.

2. Menampilkan Contoh Citra (Sample Images) 
<p align="justify">
Beberapa contoh gambar dari masing-masing kelas ditampilkan untuk memastikan dataset terbaca dengan benar serta memahami karakteristik visual dari citra 48×48 grayscale.

3. Ringkasan Dataset (Summary Dataframe) 
<p align="justify"> 
Dilakukan pembacaan struktur direktori untuk mengetahui jumlah gambar pada setiap subset (train/validation/test) sekaligus menampilkan ringkasannya dalam bentuk tabel.

4. Analisis Distribusi Kelas
<p align="justify">
Dataset divisualisasikan menggunakan bar chart untuk melihat sebaran jumlah gambar pada setiap kelas ekspresi. Langkah ini penting untuk mengidentifikasi ketidakseimbangan data yang dapat mempengaruhi performa model.

5. Preprocessing dengan ImageDataGenerator
<p align="justify">
Dataset kemudian diproses menggunakan ImageDataGenerator yang mencakup normalisasi citra serta augmentasi sederhana seperti rotasi, zoom, dan horizontal flip. Tujuannya adalah memperkaya variasi data training sekaligus mengurangi risiko overfitting.

6. Membangun Arsitektur CNN
<p align="justify">
Model CNN dirancang menggunakan beberapa convolutional layer, max pooling, flatten, serta dense layer dengan aktivasi softmax pada bagian output. Model kemudian dikompilasi dengan loss categorical crossentropy, optimizer Adam, dan metrik akurasi.

7. Pelatihan Model (Training)
<p align="justify">
Model dilatih menggunakan training set dan divalidasi menggunakan validation set. Callback seperti ModelCheckpoint digunakan agar model terbaik tersimpan otomatis berdasarkan performa validasi.

8. Visualisasi Hasil Training
<p align="justify">
Setelah pelatihan selesai, grafik loss dan accuracy antara training dan validation divisualisasikan untuk melihat perkembangan performa model serta memeriksa indikasi overfitting.

9. Evaluasi Model (Testing & Metrics)
<p align="justify">
Model diuji menggunakan test set. Evaluasi mencakup classification report, confusion matrix, serta visualisasi tingkat akurasi dan kesalahan model dalam mengenali masing-masing ekspresi wajah.

# BAB IV #
# HASIL DAN PEMBAHASAN #
<p align="justify">
Bab ini menyajikan hasil implementasi dan evaluasi model Convolutional Neural Network (CNN) dalam mendeteksi ekspresi wajah menggunakan dataset FER2013. Setiap proses dianalisis berdasarkan output yang dihasilkan dari eksekusi program, meliputi visualisasi data, distribusi kelas, hasil pelatihan, serta evaluasi performa model.

# 4.1 Sample Images (Menampilkan Contoh Citra) #
<p align="justify">
Sebelum model dikembangkan, langkah awal yang penting adalah memahami karakteristik dataset secara visual. Pemeriksaan citra dilakukan untuk melihat kualitas data, variasi wajah, serta pola umum pada setiap kelas ekspresi. Melalui proses ini, potensi kesulitan seperti pencahayaan tidak merata, citra beresolusi rendah, dan kemiripan antar ekspresi dapat diketahui sejak awal. Pada Gambar 4.1 menampilkan beberapa contoh citra dari tujuh kelas ekspresi pada dataset FER2013: Angry, Disgust, Fear, Happy, Sad, Surprise, dan Neutral. Setiap citra berukuran 48×48 piksel dalam format grayscale dan ditampilkan dalam bentuk grid.
<img width="1189" height="1956" alt="image" src="https://github.com/user-attachments/assets/0a1ed107-bd58-4923-867e-ca8e45046c8d" />

<p align="center">
Gambar 4.1 Sample Images

<p align="justify">
Secara umum, kelas Happy dan Surprise memiliki ciri visual yang lebih mudah dikenali, sedangkan Angry, Disgust, dan Sad tampak lebih mirip satu sama lain sehingga berpotensi menimbulkan kesalahan klasifikasi. Variasi pencahayaan, blur, serta minimnya detail pada citra juga menjadi tantangan bagi model CNN dalam mengenali pola wajah secara akurat. Visualisasi ini memberikan gambaran karakteristik visual dari tiap ekspresi. Terlihat bahwa citra dalam dataset memiliki kualitas rendah dan pencahayaan yang bervariasi. Hal ini menjadi tantangan tersendiri bagi model CNN dalam mengekstraksi fitur yang relevan. Meskipun demikian, pola perbedaan ekspresi seperti bentuk alis, kontur mulut, dan ketegangan wajah masih dapat diamati.

# 4.2 Distribusi Data pada Setiap Kelas #
<p align="justify">
Distribusi jumlah sampel pada setiap kelas ditunjukkan pada Gambar 4.2 melalui visualisasi bar chart yang menggambarkan perbedaan jumlah data antar kategori ekspresi. Terlihat bahwa kelas Happy memiliki jumlah sampel terbanyak, sedangkan Disgust merupakan kelas dengan jumlah data paling sedikit. Ketidakseimbangan ini berpotensi menimbulkan bias pada model karena kelas mayoritas lebih dominan dalam proses pembelajaran. Untuk meminimalkan dampak tersebut, dilakukan augmentasi data secara terbatas pada kelas-kelas dengan representasi kecil. Selain itu, Gambar 4.2 juga menyertakan donut chart yang menunjukkan proporsi keseluruhan dataset, sehingga memperjelas komposisi distribusi data secara menyeluruh.
  
<img width="1407" height="584" alt="image" src="https://github.com/user-attachments/assets/340d77ea-d22b-4e4a-b816-e6e27fbc5034" />

<p align="center">
Gambar 4.2 Distribusi Data pada Setiap Kelas dan Proposi Dataset

<p align="justify">
Berdasarkan visualisasi tersebut, dapat dipastikan bahwa dataset memiliki distribusi kelas yang tidak seimbang. Kondisi ini memperjelas bahwa komposisi data yang tidak merata memerlukan penanganan khusus pada tahap pelatihan, seperti penerapan augmentasi untuk mengurangi potensi bias model.

# 4.3 Pelatihan Model CNN #
<p align="justify">
Model CNN dibangun menggunakan empat lapisan konvolusi yang dilengkapi dengan batch normalization dan max pooling, serta dua lapisan fully connected sebelum lapisan softmax sebagai output. Pelatihan dilakukan selama maksimal 50 epoch menggunakan optimizer Adam dengan learning rate 0.0001, serta menerapkan mekanisme early stopping dan model checkpoint untuk mencegah overfitting dan memastikan model terbaik tersimpan.

<img width="865" height="522" alt="image" src="https://github.com/user-attachments/assets/ced4bea2-356c-4644-aa18-f97e4d5e269a" />
<img width="860" height="519" alt="image" src="https://github.com/user-attachments/assets/8be34907-607b-48a5-a980-64f539ba9ee1" />

<p align="center">
Gambar 4.3 Pelatihan Model

<p align="justify">
Pada tahap pelatihan, sistem berhasil memuat 28.821 citra sebagai data training dan 7.066 citra sebagai data validation. Selama proses pelatihan, akurasi training meningkat secara bertahap dari sekitar 20% pada epoch pertama hingga mencapai sekitar 75% pada epoch ke-48. Akurasi validation juga menunjukkan peningkatan dari 17% menjadi sekitar 65%. Mekanisme early stopping menghentikan pelatihan pada epoch ke-48 karena tidak terdapat peningkatan signifikan pada performa validation. Model terbaik diperoleh pada epoch ke-38 dan secara otomatis disimpan pada direktori yang telah ditentukan. Hasil ini menunjukkan bahwa model mampu mempelajari pola ekspresi wajah secara efektif meskipun dataset memiliki kualitas citra yang rendah dan distribusi kelas yang tidak seimbang. Ringkasan log pelatihan secara lengkap ditampilkan pada bagian lampiran.

<img width="1011" height="402" alt="image" src="https://github.com/user-attachments/assets/80719126-9d55-48b8-a03d-565764012c7b" />

<p align="center">
Gambar 4.4 Grafik Akurasi dan Loss Model
  
<p align="justify">
Selama proses pelatihan, model menunjukkan peningkatan performa secara bertahap. Gambar 4.4 memperlihatkan perkembangan akurasi dan loss untuk data training dan validation selama proses pelatihan. Akurasi training menunjukkan peningkatan yang konsisten dari awal hingga akhir pelatihan, sementara akurasi validation mencapai performa terbaik pada sekitar epoch ke-38 sebelum stabil menjelang epoch akhir. Kurva loss pada kedua subset juga mengalami penurunan yang stabil, dengan perbedaan yang tidak terlalu besar antara training loss dan validation loss. Hal ini mengindikasikan bahwa model tidak mengalami overfitting dan mampu mempertahankan kemampuan generalisasi terhadap data yang belum pernah dilihat sebelumnya.

# 4.4 Evaluasi Model #
<p align="justify">
Evaluasi model dilakukan menggunakan validation set yang terdiri dari 7.066 citra. Hasil evaluasi disajikan melalui classification report dan confusion matrix yang divisualisasikan pada Gambar 4.5. dan Gambar 4.6. berikut.

<p align="center">
<img width="344" height="193" alt="image" src="https://github.com/user-attachments/assets/f57839f4-9f36-487b-879a-0b350054c525" />

<p align="center">
Gambar 4.5 Clasification Report

<p align="justify">
Berdasarkan output classification report, performa model dapat diamati melalui nilai precision, recall, dan f1-score untuk masing-masing kelas. Kelas Happy menunjukkan performa terbaik dengan f1-score 0.84, diikuti kelas Surprise dengan f1-score 0.75. Kedua kelas tersebut umumnya memiliki karakteristik ekspresi yang lebih jelas, sehingga lebih mudah dikenali model. Kelas Neutral juga memperlihatkan performa stabil dengan f1-score 0.57. Sebaliknya, kelas seperti Fear (f1-score 0.44) dan Sad (f1-score 0.53) memiliki nilai yang lebih rendah, mencerminkan tantangan dalam membedakan ekspresi yang mirip pada citra grayscale beresolusi rendah. Kelas Disgust, meskipun memiliki jumlah data paling sedikit, memperoleh f1-score 0.59, menunjukkan bahwa ketidakseimbangan data berpengaruh terhadap akurasi prediksi.

<p align="center">
<img width="668" height="584" alt="image" src="https://github.com/user-attachments/assets/fb832335-2823-4e89-af6e-9b38aad5cf81" />

<p align="center">
Gambar 4.6 Confusion Matrix

<p align="justify">
Gambar 4.6 menampilkan confusion matrix yang memberikan gambaran lebih detail mengenai pola kesalahan prediksi model. Dari visualisasi tersebut tampak bahwa ekspresi Angry beberapa kali diklasifikasikan sebagai Disgust, sedangkan ekspresi Fear sering diprediksi sebagai Surprise. Selain itu, ekspresi Sad kerap tertukar dengan Neutral, terutama karena kemiripan fitur pada citra dengan kontras rendah. Pola ini merupakan karakteristik umum pada dataset FER2013 yang memiliki resolusi rendah dan ekspresi yang tumpang tindih.
Secara keseluruhan, model mencapai akurasi sebesar 64% pada validation set. Kombinasi informasi dari classification report dan confusion matrix memberikan pemahaman yang menyeluruh mengenai performa model, termasuk kekuatan model pada kelas tertentu serta jenis kesalahan yang masih perlu diperbaiki.

# BAB V #
# KESIMPULAN #
<p align="justify">
Penelitian ini berhasil membangun model Face Expression Recognition berbasis Convolutional Neural Network (CNN) untuk mengklasifikasikan tujuh ekspresi wajah pada dataset FER2013. Arsitektur CNN yang digunakan mampu mengekstraksi pola visual dari citra grayscale 48×48, sementara proses pra-pemrosesan dan augmentasi membantu meningkatkan keragaman data sehingga model dapat belajar lebih stabil. Distribusi kelas yang tidak seimbang menjadi tantangan utama, di mana kelas dengan jumlah sampel besar seperti Happy dan Surprise menunjukkan performa lebih baik dibandingkan kelas dengan data terbatas seperti Disgust dan Fear. Selama pelatihan, akurasi training meningkat dari sekitar 20% hingga lebih dari 75%, sedangkan akurasi validation mencapai sekitar 65% pada epoch ke-38 sebelum early stopping menghentikan proses. Hal ini menunjukkan bahwa model mampu belajar secara efektif tanpa mengalami overfitting yang signifikan.

<p align="justify">
Evaluasi model menghasilkan akurasi keseluruhan 64% pada validation set. Hasil classification report menunjukkan bahwa kelas Happy memiliki f1-score tertinggi (0.84), sedangkan kelas Fear terendah (0.44). Visualisasi confusion matrix memperlihatkan beberapa pola kesalahan yang umum terjadi, seperti Angry yang sering tertukar dengan Disgust dan Sad yang mirip dengan Neutral. Kesalahan ini wajar mengingat kualitas citra dataset yang rendah dan kemiripan fitur antar ekspresi. Secara keseluruhan, penelitian ini menunjukkan bahwa CNN cukup efektif digunakan untuk pengenalan ekspresi wajah pada dataset resolusi rendah. Meski hasilnya masih dapat ditingkatkan, model ini memberikan fondasi yang baik untuk pengembangan lebih lanjut, seperti penggunaan arsitektur yang lebih kuat, teknik penyeimbangan kelas yang lebih optimal, serta pengujian pada data dunia nyata untuk meningkatkan generalisasi.
