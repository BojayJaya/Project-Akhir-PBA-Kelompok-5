from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
# import numpy as np
import regex as re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN)</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="data:image/jpeg;base64,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" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://tse2.mm.bing.net/th?id=OIP.STTKkkt17TKUvsAE4wKHCwHaED&pid=Api&P=0&h=180" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">dataset tentang ulasan terhadap wisata dieng dari website tripadvisor. Selanjutnya data ulasan tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma k-nearest neighbor (K-NN) untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)

        st.write("""###### Penjelasan Prepocessing Data : """)
        st.write("""1. Case Folding :""")
        
        st.write("""Case folding adalah proses dalam pemrosesan teks yang mengubah semua huruf dalam teks menjadi huruf kecil atau huruf besar. Tujuan dari case folding adalah untuk mengurangi variasi yang disebabkan oleh perbedaan huruf besar dan kecil dalam teks, sehingga mempermudah pemrosesan teks secara konsisten.""")
        
        st.write("""Dalam case folding, biasanya semua huruf dalam teks dikonversi menjadi huruf kecil dengan menggunakan metode seperti lowercasing. dengan demikian, perbedaan antara huruf besar dan huruf kecil tidak lagi diperhatikan dalam analisis teks, sehingga memungkinkan untuk mendapatkan hasil yang lebih konsisten dan mengurangi kompleksitas dalam pemrosesan teks.""")
        
        st.write("""2. Tokenize :""")

        st.write("""Tokenisasi adalah proses pemisahan teks menjadi unit-unit yang lebih kecil yang disebut token. Token dapat berupa kata, frasa, atau simbol lainnya, tergantung pada tujuan dan aturan tokenisasi yang digunakan.""")

        st.write("""Tujuan utama tokenisasi dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah untuk memecah teks menjadi unit-unit yang lebih kecil agar dapat diolah lebih lanjut, misalnya dalam analisis teks, pembentukan model bahasa, atau klasifikasi teks.""")

        st.write("""3. Filtering (Stopword Removal) :""")

        st.write("""Filtering atau Stopword Removal adalah proses penghapusan kata-kata yang dianggap tidak memiliki makna atau kontribusi yang signifikan dalam analisis teks. Kata-kata tersebut disebut sebagai stop words atau stopwords.""")

        st.write("""Stopwords biasanya terdiri dari kata-kata umum seperti “a”, “an”, “the”, “is”, “in”, “on”, “and”, “or”, dll. Kata-kata ini sering muncul dalam teks namun memiliki sedikit kontribusi dalam pemahaman konten atau pengambilan informasi penting dari teks.""")

        st.write("""Tujuan dari Filtering atau Stopword Removal adalah untuk membersihkan teks dari kata-kata yang tidak penting sehingga fokus dapat diarahkan pada kata-kata kunci yang lebih informatif dalam analisis teks. Dengan menghapus stopwords, kita dapat mengurangi dimensi data, meningkatkan efisiensi pemrosesan, dan memperbaiki kualitas hasil analisis.""")
        st.write("""4. Stemming :""")

        st.write("""Stemming dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah proses mengubah kata ke dalam bentuk dasarnya atau bentuk kata yang lebih sederhana, yang disebut sebagai “stem”. Stemming bertujuan untuk menghapus infleksi atau imbuhan pada kata sehingga kata-kata yang memiliki akar kata yang sama dapat diidentifikasi sebagai bentuk yang setara.""")
        st.write("""###### Penjelasan Ekstraksi Fitur : """)
        st.write("""TF-IDF :""")
        st.write("""Ditahap akhir dari text preprocessing adalah term-weighting .Term-weighting merupakan proses pemberian bobot term pada dokumen. Pembobotan ini digunakan nantinya oleh algoritma Machine Learning untuk klasifikasi dokumen. Ada beberapa metode yang dapat digunakan, salah satunya adalah TF-IDF (Term Frequency-Inverse Document Frequency).""")
        st.write("""TF (Term Frequency) :""")
        st.write("""TF (Term Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam suatu dokumen. Menghitung TF melibatkan perbandingan jumlah kemunculan kata dengan jumlah kata keseluruhan dalam dokumen.""")
        st.write("""Perhitungan TF (Term Frequency) :
        
        TF(term) = (Jumlah kemunculan term dalam dokumen) / (Jumlah kata dalam dokumen)
        """)
        st.write("""DF (Document Frequency) :""")
        st.write("""DF (Document Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam seluruh koleksi dokumen. DF menghitung jumlah dokumen yang mengandung kata tersebut.""")
        st.write("""Perhitungan DF (Document Frequency) :
        
        DF(term) = Jumlah dokumen yang mengandung term
        """)
        st.write("""IDF (Inverse Document Frequency) :""")
        st.write("""IDF (Inverse Document Frequency) adalah ukuran yang menggambarkan seberapa penting sebuah kata dalam seluruh koleksi dokumen. IDF dihitung dengan mengambil logaritma terbalik dari rasio total dokumen dengan jumlah dokumen yang mengandung kata tersebut. Tujuan IDF adalah memberikan bobot yang lebih besar pada kata-kata yang jarang muncul dalam seluruh koleksi dokumen.""")
        st.write("""Perhitungan IDF (Inverse Document Frequency) :
        
        IDF(term) = log((Total jumlah dokumen) / (DF(term)))
        """)
        st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) :""")
        st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang menggabungkan informasi TF dan IDF. TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang sering muncul dalam dokumen tertentu (TF tinggi) dan jarang muncul dalam seluruh koleksi dokumen (IDF tinggi). Metode ini digunakan untuk mengevaluasi kepentingan relatif suatu kata dalam konteks dokumen.""")
        st.write("""Perhitungan TF-IDF (Term Frequency-Inverse Document Frequency) :
        
        TF-IDF(term, document) = TF(term, document) * IDF(term)
        """)
        st.write("""Dalam perhitungan TF-IDF, TF(term, document) adalah nilai TF untuk term dalam dokumen tertentu, dan IDF(term) adalah nilai IDF untuk term di seluruh koleksi dokumen.""")
        st.write("""Mengubah representasi teks ke dalam vektor
        """)
        
        st.write("#### Dataset")
        df = pd.read_csv("hasil_preprocessing.csv")
        # df = df.drop(columns=['nama','sentiment','score'])
        st.write(df)

    elif selected == "Implementation":
        #Getting input from user
        iu = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            def prep_input_data(iu):
                ulasan_case_folding = iu.lower()

                #Cleansing
                clean_tag  = re.sub("@[A-Za-z0-9_]+","", ulasan_case_folding)
                clean_hashtag = re.sub("#[A-Za-z0-9_]+","", clean_tag)
                clean_https = re.sub(r'http\S+', '', clean_hashtag)
                clean_symbols = re.sub("[^a-zA-Z ]+"," ", clean_https)
                
                #Inisialisai fungsi tokenisasi dan stopword
                # stop_factory = StopWordRemoverFactory()
                tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
                tokens = tokenizer.tokenize(clean_symbols)

                #Stop Words
                stop_factory = StopWordRemoverFactory()
                more_stopword = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                                'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                                'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                                '&amp', 'yah']
                data = stop_factory.get_stop_words()+more_stopword
                removed = []
                if tokens not in data:
                    removed.append(tokens)

                #list to string
                gabung =' '.join([str(elem) for elem in removed])

                #Steaming
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(gabung)
                return(ulasan_case_folding,clean_symbols,tokens,gabung,stem)

            #Dataset
            Data_ulasan = pd.read_csv("hasil_preprocessing.csv")
            ulasan_dataset = Data_ulasan['ulasan_hasil_preprocessing']
            sentimen = Data_ulasan['label']

            # TfidfVectorizer 
            # tfidfvectorizer = TfidfVectorizer(analyzer='iu')
            # tfidf_wm = tfidfvectorizer.fit_transform(ulasan_dataset)
            # tfidf_tokens = tfidfvectorizer.get_feature_names_out()
            # df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
            with open('knnk9.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            
            with open('tfidf.pkl', 'rb') as file:
                loaded_data_tfid = pickle.load(file)
            
            tfidf_wm = loaded_data_tfid.fit_transform(ulasan_dataset)

            #Train test split
            training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.2, random_state=42)#Nilai X training dan Nilai X testing 80 20
            # training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.3, random_state=42)#Nilai X training dan Nilai X testing 70 30
            # training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.4, random_state=42)#Nilai X training dan Nilai X testing 60 40
            # training_label, test_label = train_test_split(, test_size=0.2, random_state=42)#Nilai Y training dan Nilai Y testing    

            #model
            clf = loaded_model.fit(training, training_label)
            y_pred = clf.predict(test)

            #Evaluasi
            akurasi = accuracy_score(test_label, y_pred)
            akurasi_persen = akurasi * 100

            #Inputan 
            ulasan_case_folding,clean_symbols,tokens,gabung,stem = prep_input_data(iu)
            st.write('Case Folding')
            st.write(ulasan_case_folding)
            st.write('Cleaning Simbol')
            st.write(clean_symbols)
            st.write('Token')
            st.write(tokens)
            st.write('Stop Removal')
            st.write(gabung)
            st.write('Stemming')
            st.write(stem)

        
            #Prediksi
            v_data = loaded_data_tfid.transform([stem]).toarray()
            y_preds = clf.predict(v_data)

            st.subheader('Akurasi')
            # st.info(akurasi)
            st.info(f"{akurasi_persen:.2f}%")

            st.subheader('Prediksi')
            if y_preds == "positive":
                st.success('Positive')
            else:
                st.error('Negative')

    elif selected == "Tentang Kami":
        st.write("##### Mata Kuliah = Pemrosesan Bahasa Alami -A") 
        st.write('##### Kelompok 5')
        st.write("1. Hambali Fitrianto - 200411100074")
        st.write("2. Pramudya Dwi Febrianto - 200411100042")
        st.write("3. Febrian Achmad Syahputra - 200411100106")
        
