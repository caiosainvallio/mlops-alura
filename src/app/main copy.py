from flask import Flask, request, jsonify
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# dataset
df = pd.read_csv('casas.csv')


# modelo 1 (uma variavel)
colunas = ['tamanho', 'preco']
df_1 = df[colunas]

X_1 = df_1.drop('preco', axis=1)
y_1 = df_1['preco']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_1, y_1, test_size=0.3, random_state=42
)
modelo_1 = LinearRegression()
modelo_1.fit(X_train_1, y_train_1)


# modelo 2 (3 variaveis) from pickle
modelo_2 = pickle.load(open('modelo.sav', 'rb'))

# request
colunas_json = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)


# rota -> endpoint -> pontos de acesso a aplicacao
@app.route('/')
def home():
    return 'Minha primeira API.'


@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_eng = tb.translate(from_lang='pt', to='en')
    polaridade = tb_eng.sentiment.polarity
    return f"Polaridade: {polaridade}"


@app.route('/cotacao_1/<int:tamanho>')
def cotacao_1(tamanho):
    preco = modelo_1.predict([[tamanho]])
    return str(preco)


# payload conjunto de dados enviados ao endpoint
@app.route('/cotacao_2/', methods=['POST'])
def cotacao_2():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas_json]
    preco = modelo_2.predict([dados_input])
    return jsonify(preco=preco[0])   



app.run(debug=True)

# pra rodar: python main.py