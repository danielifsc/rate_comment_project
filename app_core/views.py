from django.shortcuts import render
#import analyze_user_comment
import string
import nltk
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk_packeges = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet', 'punkt_tab' ]
for package in nltk_packeges:
    try:
        nltk.data.find('tokenizers/{package}') if package.startswith('punkt') else nltk.data.find(f'corpora/{package}') 
    except LookupError:
        nltk.download(package, quiet=True)


sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
# Create your views here.

def home(request):
    print(f"Antes de renderizar")
    #nltk.download('vader_lexicon', quiet=True)
   
    result = None
    comment = ""

    if request.method == 'POST':
        comment = request.POST.get('comment_text', '').strip()
        if comment:
            result = analyze_user_comment(comment)

        print(f"Antes de renderizar 2{result}")
        #return render(request, 'home.html', {'result': result})
    return render(request, 'home.html', {
        'result': result,
        'comment': comment,
    })

def pre_processamento(text):

    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    print(stop_words)
    tokens_processados = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens_processados)


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


