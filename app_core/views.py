from django.shortcuts import render
#import analyze_user_comment
import string
import nltk
import re
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from .models import Avaliacao

#analise de sentimento

#baixando
nltk_packeges = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet', 'punkt_tab' ]
for package in nltk_packeges:
    try:
        nltk.data.find('tokenizers/{package}') if package.startswith('punkt') else nltk.data.find(f'corpora/{package}') 
    except LookupError:
        nltk.download(package, quiet=True)


sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()


palavras_de_sentimento = {
    "no", "not", "nor", "neither", "never", "none", "isn", "isnt",
    "wasn","wasnt","aren","arent","weren","werent","don","dont","didn",
    "didnt","doesn","doesnt","haven", "havent","hasn","hasnt","hadn","hadnt",
    "won","wont", "wouldn","wouldnt","shouldn","shouldnt","can","cannot","cant",
    "couldn","couldnt","mustn","mustnt","but","however","nt",  
    # Importante: o NLTK tokeniza "don't" como ["do", "n't"], e a regex remove o apóstrofo virando "nt"
}

stop_words = set(stopwords.words('english'))
stop_words_custom = stop_words - palavras_de_sentimento


# 1. Função auxiliar declarada ANTES da view home
def get_dataframe(limit=50):
    livros_df = Avaliacao.objects.all().values()[:limit]
    df = pd.DataFrame(list(livros_df))

    if not df.empty:
        if 'review_time' in df.columns:
            df['review_time'] = pd.to_datetime(df['review_time'], unit='s')
        return df.to_html(classes='dataframe',index=False, escape=False)
    else:
        return "<p class='alert alert-warning text-center'>Nenhum dado encontrado no banco de dados.</p>"

# 2. View da página inicial
def home(request):
    print("Antes de renderizar")

    result = None
    comment = ""

    if request.method == 'POST':
        comment = request.POST.get('comment_text', '').strip()
        if comment:
            result = analyze_user_comment(comment)

        print(f"Antes de renderizar 2: {result}")

 
    df_html = get_dataframe(limit=50)

    context = {
        'result': result,
        'comment': comment,
        'df_html': df_html,
    }

    return render(request, 'home.html', context)


#pre-processando o texto enviado da caixa de texto
def pre_processamento(text):

    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)




    print(stop_words)

    tokens_processados = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words_custom
    ]

    return " ".join(tokens_processados)

#funcao da analise de sentimento(normal e pós-processado)
def analyze_user_comment(text):

    if not text or not isinstance(text, str):
        return None

    texto_limpo = pre_processamento(text)

    scores_normal = sia.polarity_scores(text)
    scores_processado = sia.polarity_scores(texto_limpo)
    
    compound_normal = scores_normal['compound']
    compound_processado = scores_processado['compound']


    if compound_normal >= 0.05:
        sentiment_normal = 'Positivo'
    elif compound_normal <= -0.05:
        sentiment_normal = 'Negativo'
    else:
        sentiment_normal = 'Neutro'

    if compound_processado >= 0.05:
        sentiment_processado = 'Positivo'
    elif compound_processado <= -0.05:
        sentiment_processado = 'Negativo'
    else:
        sentiment_processado = 'Neutro'    
    
    print(f"Analisou{sentiment_normal}")
    print("--------")
    print(f"Analisou{sentiment_processado}")
    
    return {
        'original_text': text,
        'texto_limpo': texto_limpo,

        'sentiment_normal': sentiment_normal,
        'scores_normal': scores_normal,
        
        'sentiment_processado': sentiment_processado,
        'scores_processado': scores_processado
    }


