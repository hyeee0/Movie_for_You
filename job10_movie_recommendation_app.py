import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from PyQt5.QtCore import QStringListModel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import linear_kernel
from konlpy.tag import Okt
import re

form_window = uic.loadUiType('./movie_recommendation_app.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 아래 3가지는 추천기능 시스템을 사용하기 위해서 불러온 모델들이다
        self.tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
        with open('./models/tfidf.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')

        self.df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
        self.titles = self.df_reviews['titles']
        self.titles = sorted(self.titles)
        for title in self.titles:
            self.combo_box.addItem(title)

        # 자동완성 기능을 만들어보자    
        model = QStringListModel()
        model.setStringList(self.titles)
        complter = QCompleter() #QCompleter를 사용하면 자동완성이 된다
        complter.setModel(model) # QStringListModel으로 자동완성을 시킨다
        self.line_edit.setCompleter(complter)

        self.combo_box.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommend.clicked.connect(self.btn_slot)

    # 반복되는 코드를 함수로 만들어 주자(제목 함수)
    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]  # '' 안에 영화의 인덱스를 찾아준다
        cosin_sim = linear_kernel(self.tfidf_matrix[movie_idx],
                                  self.tfidf_matrix)  # 두 벡터 사이의 각도가 1일수록 유사도가 높다(코사인심 -> 코사인 유사도)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[1:]))
        print(recommendation)
        self.lbl_recommend.setText(recommendation)

    # 반복되는 코드를 함수로 만들어 주자(키워드 함수)
    def recommendation_by_key_word(self, key_word):
        sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
        words = [key_word]
        for word, _ in sim_word:
            words.append(word)

        sentence = []
        count = 11

        for word in words:
            sentence = sentence + [word] * count
            count -= 1
        sentence = ' '.join(sentence)
        sentence_vec = self.tfidf.transform([sentence])
        cosin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[:10]))  # 상위 10개 프린트
        self.lbl_recommend.setText(recommendation)

    # 반복되는 코드를 함수로 만들어 주자(문장 함수)
    def recommendation_by_sentence(self, key_word):
        review = re.sub('[^가-힣 ]', ' ', key_word)
        okt = Okt()
        token = okt.pos(review, stem=True)
        df_token = pd.DataFrame(token, columns=['word', 'class'])
        df_token = df_token[(df_token['class'] == 'Noun') |
                            (df_token['class'] == 'Verb') |
                            (df_token['class'] == 'Adjective')]
        words = []
        for word in df_token.word:
            if len(word) > 1:  # 단어길이가 1보다 크고 -> 2글자 이상
                words.append(word)  # 리스트에 추가해준다
        cleaned_sentence = ' '.join(words)  # 포함시키지 않은 단어는 공백으로 만든다.

        sentence_vec = self.tfidf.transform([cleaned_sentence])
        cosin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[:10]))  # 상위 10개 프린트
        self.lbl_recommend.setText(recommendation)

    # 텍스트를 입력한 뒤 버튼을 눌러 추천을 받는 방법
    def btn_slot(self):
        # title = self.line_edit.text()
        # self.recommendation_by_movie_title(title)

        # 아래는 키워드 입력한 뒤 '추천'버튼을 누르면 영화를 추천해주는 코드
        # key_word = self.line_edit.text()
        # if key_word in self.titles:
        #     self.recommendation_by_movie_title(key_word)
        # elif key_word in list(self.embedding_model.wv.index_to_key):
        #     self.recommendation_by_key_word(key_word)

        # 아래는 문장을 입력한 뒤 '추천'버튼을 누르면 영화를 추천해주는 코드
        key_word = self.line_edit.text()
        if key_word in self.titles:
            self.recommendation_by_movie_title(key_word)
        elif key_word in list(self.embedding_model.wv.index_to_key):
            self.recommendation_by_key_word(key_word)
        else:
            self.recommendation_by_sentence(key_word)

    # 콤보박스를 사용해서 추천받는 방법
    def combobox_slot(self):
        title = self.combo_box.currentText() # 햔재 입력된 글자에서
        self.recommendation_by_movie_title(title)

    def getRecommendation(self, cosin_sim):
        simScore = list(enumerate(cosin_sim[-1]))  #
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]  # 11개를 추출해준다 / 왜냐면 10개를 추천해줄텐데 첫번째는 자기자신이기에 나중에 빼주기 위해
        movie_idx = [i[0] for i in simScore]  #
        recMovieList = self.df_reviews.iloc[movie_idx, 0]  # 0번 컬럼은 타이틀만 인덱싱
        return recMovieList

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())