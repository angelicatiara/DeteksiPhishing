import pickle
import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
from nltk.tokenize import RegexpTokenizer
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure

st.title('Prediksi Website Phishing Menggunakan Machine Learning :fish:')
st.subheader('by: Kelompok 7 / Tim Mancing Mania :star:🎣')
st.caption('Members: Tiara Angelica, Annisa Salsabila Ahdyani, Ina Mutmainah')

st.image("https://csirt.umm.ac.id/wp-content/uploads/2022/06/bigstock-Data-Phishing-Hacker-Attack-t-319270852.jpg")

"""
Phishing adalah upaya untuk mendapatkan informasi data seseorang dengan teknik pengelabuan. 
Data yang menjadi sasaran phising adalah data pribadi (nama, usia, alamat), data akun (username dan password), 
dan data finansial (informasi kartu kredit, rekening). Istilah resmi phising adalah phishing, yang berasal dari kata fishing yaitu memancing.

Informasi yang didapat atau dicari oleh phiser adalah berupa password account atau nomor kartu kredit korban. 
Penjebak/phiser menggunakan email, banner atau pop-up window untuk menjebak user agar mengarahkan ke situs web palsu (fake webpage), 
dimana user diminta untuk memberikan informasi pribadinya. Disinilah phiser memanfaatkan kecerobohan dan 
ketidak-telitian user dalam web palsu tersebut untuk mendapatkan informasi.

"""
phishing_data = st.selectbox("Pilih dataset phishing URLs yang sudah kami siapkan untuk digunakan", ("C:\\Users\\TIARA ANGELICA S\\Downloads\\phishing_site_urls.csv","data-sample-2"))
st.markdown("**atau**")
uploaded_file = st.file_uploader("Mau upload datanya sendiri? Monggo...", type='.csv')


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    st.caption('Terima kasih buat datanya :D')

if not uploaded_file:
    df = pd.read_csv(phishing_data)
    st.dataframe(df)
    st.caption('Mari kita gunakan data ini.')

"""
Mengenai Dataset 📊:
* Data ini mengandung 549346 baris yang unique/distinct.
* Kolom label adalah kolom yang akan kita gunakan untuk prediksi dan mempunyai 2 kategori:
1. Good - URL tidak mengandung hal-hal yang berbahaya dan situs tersebut bukan sebuah situs phishing.
2. Bad - URL mengandung hal-hal yang berbahaya bagi komputer users dan situs tersebut adalah sebuat situs phishing.

Tidak ada value yang NULL di dataset.

"""


"""
🤖 Menggunakan Machine Learning untuk Deteksi Website Phishing 🤖

Dapat dilihat di data di atas, ada banyak sekali website phishing yang mirip-mirip dengan website asli.
Oleh karena itu, di project ini, mari kita gunakan machine learning (logistik linear simple) untuk membuat program deteksi URL

"""

st.image('https://i.im.ge/2022/08/08/FZlDGM.jenisphishing.jpg')    

"""
💻 💻 Tunggu apa lagi, mari kita lakukan coding-nya ^.^ 💻 💻 

Yang pertama kita lakukan adalah memasukkan datanya ke dalam program.
"""
st.code(  
    f"""
import numpy as np
import pandas as pd
import streamlit as st

lbl_counts = pd.DataFrame(phishing_data.Label.value_counts())
import seaborn as sns
sns.set_style('darkgrid')
sns.barplot(lbl_counts.index, lbl_counts.Label)
""",language='python',)
st.image('https://i.im.ge/2022/08/08/FZFblq.outputsubplot1.jpg')


"""
KEDUA, Preprocessing

Setelah kita load datanya, maka kita akan melakukan vektorisasi dari URL-URL kita.
Jadi, program ini akan menggunakan CountVectorizer untuk mendapatkan kata-kata menggunakan
tokenizer, oleh karena itu, ada beberapa kata yang perlu kita perhatikan seperti:
1. -> Virus
2. -> .exe
3. -> .dat

Kita akan konversi URL ini ke form vektor menggunakan REGEXP TOKENIZER
* Ini adalah tokenizer yang memisahkan sebuah string menggunakan fungsi regex / regular expression,
yang mencocokan antara tokens yang sudah ada atau dengan separator seperti titik dan koma.

"""
st.code(  
    f"""
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
print(phishing_data.URL[0])
""", language='python',)

st.write("Output: nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/en/cgi-bin/verification/login/70ffb52d079109dca5664cce6f317373/index.php?cmd=_profile-ach&outdated_page_tmpl=p/gen/failed-to-load&nav=0.5.1&login_access=1322408526")

st.code(  
    f"""
clean_text = tokenizer.tokenize(phishing_data.URL[0]) 
print(clean_text)
""", language='python',)
st.write("Output: ['nobell', 'it', 'ffb', 'd', 'dca', 'cce', 'f', 'login', 'SkyPe', 'com', 'en', 'cgi', 'bin', 'verification', 'login', 'ffb', 'd', 'dca', 'cce', 'f', 'index', 'php', 'cmd', 'profile', 'ach', 'outdated', 'page', 'tmpl', 'p', 'gen', 'failed', 'to', 'load', 'nav', 'login', 'access']")

st.write("⏰ Time module")
st.code(  
    f"""
import time
start = time.time()
phishing_data['text_tokenized'] = phishing_data.URL.map(lambda text: tokenizer.tokenize(text))
end = time.time()
time_req = end - start
formatted_time = ":.2f".format(time_req)
print(f"Time required to tokenize text is: formatted_time sec")
""", language='python',)

st.write("Time required to tokenize text is: 2.20 sec")
"""
Mengapa time module itu penting untuk proses machine learning?

Karena biasanya program ini dapat memproses data yang sangat buanyaakkkk, 
sehingga bisa lama 😭
"""

st.code(  
    f"""
phishing_data.sample(7)
""", language='python',)
##INSERT TABLE HERE

"""
KETIGA, ❄️ Snowball Stemmer STNK ❄️
* Library snowball adalah bahasa proses yang didesain untuk membuat algoritma stemming
yang berguna untuk mendapatkan informasi.

Kita akan menggunakan library ini untuk merapikan dan mencukur kembali text yang sudah di tokenized sebelumnya.
"""

st.code(  
    f"""
from nltk.stem.snowball import SnowballStemmer
sbs = SnowballStemmer("english")
#Kita gunakan English karena sumber data situsnya bahasa Inggris

#Lalu, mari kita kasih pointer juga untuk berapa waktu proses yang dibutuhkan.
start = time.time()
phishing_data['text_stemmed'] = phishing_data['text_tokenized'].map(lambda text: [sbs.stem(word) for word in text])
end = time.time()
ime_req = end - start
formatted_time = ":.2f".format(time_req)
print(f"⏳ Time required for stemming all the tokenized text is: \nformatted_time sec")

phishing_data.sample(7)
""", language='python',)

##INSERT TABLE HERE
"""
Sekarang, kita gabungkan kata-kata yang sudah dipangkas, dicukur, dibersihkan, disaring, di-macam-macamkan :D
ke dalam sebuah kalimat.
"""
st.code(  
    f"""
start = time.time()
phishing_data['text_to_sent'] = phishing_data['text_stemmed'].map(lambda text: ' '.join(text))
end = time.time()
time_req = end - start
formatted_time = ":.2f".format(time_req)
print(f"Time required for joining text to sentence is: nformatted_time sec")
""", language='python',)
st.write("Time required to tokenize text is: 0.28 sec")
st.code(  
    f"""
phishing_data.sample(10)
""", language='python',)
#INSERT TABLE HERE

"""
KEEMPAT, Visualisasi 👀👀👀👀👀👀👀👀
Kita akan membuat WordCloud untuk kata-kata yang sering muncul.
"""

st.code(  
    """
phishing_sites = phishing_data[phishing_data.Label == 'bad']
not_phishing_sites = phishing_data[phishing_data.Label == 'good']

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image

def my_wordcloud(text, mask=None, max_words=500, max_font_size=70, figure_size=(8.0, 10.0),
                title=None, title_size=70, image_color=False):
    
    stopwords = set(STOPWORDS)
    my_stopwords = {'com', 'http'}
    stopwords = stopwords.union(my_stopwords)
    
    wordcloud = WordCloud(background_color='#fff', stopwords = stopwords, max_words = max_words, random_state = 42, mask = mask)
    
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    
    if image_color:
        image_color = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation='bilinear');
        
        plt.title(title,fontdict={'size': title_size,'verticalalignment': 'bottom'})
        
    else:
        plt.imshow(wordcloud);
        plt.title(title,fontdict={'size': title_size,'color': '#ff3333','verticalalignment': 'bottom'})

    plt.axis('off');
    plt.tight_layout()
    
d = '../input/images/'

my_data = not_phishing_sites.text_to_sent
my_data.reset_index(drop=True, inplace=True)
not_phishing_common_text = str(my_data)
common_mask = np.array(Image.open(d+'idea.png'))
my_wordcloud(not_phishing_common_text,common_mask,max_words=400, max_font_size=50, title = 'The Most common words use in not phishing URLs:', title_size=20)
""", language='python',)
st.image("https://i.im.ge/2022/08/08/FZ0czF.wordcloud-jpg.png")


"""
KELIMA DAN TERAKHIR: Membuat Model (Akhirnya!🥳🥳🥳)
Disini kita menggunakan sebuat ekstraktor fitur yang bernama CountVectorizer untuk mengambil data vektor dari semua kata-kata,
suku kata, dan token.

Setelah itu, kita akan menggunakan simple logistic regression sebagai algoritma Machine Learning kita.

"""
st.code(  
    f"""
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
feature = CV.fit_transform(phishing_data.text_to_sent)
feature[:5].toarray()
""", language='python',)
st.write("array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]])")

st.code(  
    f"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#Memisahkan data test dan train
rain_X, test_X, train_Y, test_Y = train_test_split(feature, phishing_data.Label)
""", language='python',)

"""
LOGISTIC REGRESSION
adalah jenis analisis statistik yang sering digunakan data analyst untuk pemodelan prediktif. Dalam pendekatan analitik ini, 
variabel dependennya terbatas atau kategoris, bisa berupa A atau B (regresi biner) atau berbagai opsi hingga A, B, C atau D 
(regresi multinomial).
"""
st.code(  
    f"""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_X, train_Y)
lr.score(test_X, test_Y)
""", language='python',)

st.write("0.9641500923891694")

"""
Nah, sekarang kita sudah mempunyai score modelnya. Saatnya untuk melakukan prediksi :D
"""

import webbrowser
url = 'https://www.kaggle.com/angelicatiara/deteksi-phishing-tim-mancing-mania/'
if st.button('Mau lihat proses LIVEnya? KLIK SAYA'):
    webbrowser.open_new_tab(url)
