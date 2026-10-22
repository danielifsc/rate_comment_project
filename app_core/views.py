from django.shortcuts import render
#import analyze_user_comment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



sia = SentimentIntensityAnalyzer()
# Create your views here.

def home(request):
    print(f"Antes de renderizar")
    nltk.download('vader_lexicon', quiet=True)
   
    result = None
    if request.method == 'POST':
        comment = request.POST.get('comment_text', '').strip()
        if comment:
            result = analyze_user_comment(comment)
        print(f"Antes de renderizar 2{result}")
        return render(request, 'home.html', {'result': result})
    return render(request, 'home.html')


def analyze_user_comment(text):

    if not text or not isinstance(text, str):
        return None

    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = 'Positivo'
    elif compound <= -0.05:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutro'
    
    print(f"Analisou{sentiment}")
    
    return {
        'original_text': text,
        'sentiment': sentiment,
        'scores': scores
    }