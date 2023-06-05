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
from sklearn.feature_extraction.text import TfidfVectorizer
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
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart', 'check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
        # st.write("""
        # <div style = "position: fixed; left:50px; bottom: 10px;">
        #     <center><a href="https://github.com/HanifSantoso05/Aplikasi-Web-Klasifikasi-Penyakit-Anemia"><span><img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" width="40px" height="40px"></span></a><a style = "margin-left: 20px;" href="http://hanifsantoso05.github.io/datamining/intro.html"><span><img src="https://friconix.com/png/fi-stluxx-jupyter-notebook.png" width="40px" height="40px"></span></a> <a style = "margin-left: 20px;" href="mailto: hanifsans05@gmail.com"><span><img src="https://cdn-icons-png.flaticon.com/512/60/60543.png" width="40px" height="40px"></span></a></center>
        # </div> 
        # """,unsafe_allow_html=True)

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://tse2.mm.bing.net/th?id=OIP.STTKkkt17TKUvsAE4wKHCwHaED&pid=Api&P=0&h=180" width="500" height="300">
        </h3>""",unsafe_allow_html=True)
        # st.write("""
        # Anemia adalah suatu kondisi di mana Anda kekurangan sel darah merah yang sehat untuk membawa oksigen yang cukup ke jaringan tubuh Anda. Penderita anemia, juga disebut hemoglobin rendah, bisa membuat Anda merasa lelah dan lemah.
        # """)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">dataset tentang ulasan masyarakat terhadap pariwisata bukit Jaddih dari ulasan google maps. Selanjutnya data ulasan tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma Multinomial Naive Bayes untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)
        st.write(""" 
        <ol>
            <li>Case folding merupakan tahap untuk mengganti keseluruhan kata kapital pada dataset agar berubah menjadi tidak kapital.</li>
            <li>Cleansing yaitu merupakan proses untuk menghilangkan semua simbol, angka, ataupun emoticon yang terdapat didalam dataset</li>
            <li>Slangword Removing yaitu satu proses yang dilakukan untuk mendeteksi dan menghilangkan kata-kata yang tidak baku di dalam dataset</li>
            <li>Tokenization yaitu proses untuk memisahkan suatu kalimat menjadi beberapa kata untuk memudahkan proses stopword.</li>
            <li>Stopword Removing yaitu proses untuk menghilangkan semua kata hubung yang terdapat pada dataset.</li>
            <li>Steaming yaitu proses yang digunakan untuk menghilangkan semua kata imbuhan dan merubahnya menjadi kata dasar.</li>
        </ol> 
        """,unsafe_allow_html=True)
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
    #         training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.3, random_state=42)#Nilai X training dan Nilai X testing 70 30
    #         training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.4, random_state=42)#Nilai X training dan Nilai X testing 60 40
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
        st.write("##### Mata Kuliah = Pemrosesan Bahasa Alami - A") 
        st.write('##### Kelompok 5')
        st.write("1. Hambali Fitrianto - 200411100074")
        st.write("2. Pramudya Dwi Febrianto - 200411100042")
        st.write("3. Febrian Achmad Syahputra - 200411100106")
