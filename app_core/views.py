from django.shortcuts import render
#import analyze_user_comment
import string
import nltk
import re
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
import io
import base64
from wordcloud import WordCloud

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

# --- Função auxiliar para converter gráficos em imagem Base64 ---
def get_graph():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def gerar_analise_dados(df):
    context = {}
    if df.empty:
        return context

    # Conversões preliminares
    if 'review_time' in df.columns:
        df['review_time_dt'] = pd.to_datetime(df['review_time'], unit='s', errors='coerce')
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'review_score' in df.columns:
        df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')

    # --- 1. ESTRUTURA E TOTAIS ---
    context['total_avaliacoes'] = df.shape[0]
    
    # --- 2. LIVROS E PREÇOS ---
    if 'price' in df.columns:
        context['preco_medio'] = round(df['price'].mean(), 2) if not df['price'].dropna().empty else 0
        
        # O livro mais caro
        if not df['price'].dropna().empty:
            idx_max = df['price'].idxmax()
            livro_max = df.loc[idx_max]
            context['livro_mais_caro'] = {
                'titulo': livro_max.get('title', 'Título Desconhecido'),
                'preco': livro_max['price']
            }

    # --- 3. AVALIAÇÕES ÚTEIS E POSITIVIDADE ---
    if 'review_score' in df.columns:
        pos = len(df[df['review_score'] >= 4])
        neg = len(df[df['review_score'] <= 2])
        neu = len(df[df['review_score'] == 3])
        
        if pos > neg:
            context['comentario_sentimento'] = f"Há predominância de avaliações POSITIVAS ({pos} positivas vs {neg} negativas)."
        else:
            context['comentario_sentimento'] = f"Há predominância de avaliações NEGATIVAS ({neg} negativas vs {pos} positivas)."

    if 'review_helpfulness' in df.columns:
        # Lógica: converte '3/4' em 0.75. Dá prioridade a resenhas com mais de 3 votos no total para ser relevante
        def calc_help(val):
            try:
                num, den = str(val).split('/')
                num, den = int(num), int(den)
                return (num / den) if den >= 3 else 0
            except:
                return 0
                
        df['help_score'] = df['review_helpfulness'].apply(calc_help)
        uteis = df.nlargest(5, 'help_score')
        context['avaliacoes_uteis'] = []
        for _, row in uteis.iterrows():
            context['avaliacoes_uteis'].append({
                'titulo': row.get('title', 'Sem título'),
                'score': row.get('review_score', '-'),
                'helpfulness': row['review_helpfulness'],
                'texto': str(row.get('review_text', ''))[:150] + '...' # Trunca para não quebrar o layout
            })

    # --- 4. TEMPO (MESES DE PICO) ---
    if 'review_time_dt' in df.columns:
        df['ano_mes'] = df['review_time_dt'].dt.to_period('M')
        avals_tempo = df.groupby('ano_mes').size()
        
        # Gráfico temporal
        plt.figure(figsize=(8,3.5))
        avals_tempo.plot(color='#0d6efd', linewidth=2)
        plt.title('Evolução Temporal')
        plt.xlabel('Data')
        plt.ylabel('Avaliações')
        context['grafico_temporal'] = get_graph()

        # Top 3 meses com mais avaliações
        top_meses = avals_tempo.sort_values(ascending=False).head(3)
        context['meses_mais_ativos'] = [{'periodo': str(p), 'qtd': q} for p, q in top_meses.items()]

    # --- 5. USUÁRIOS E PADRÕES ---
    col_user = 'profile_name' if 'profile_name' in df.columns else ('user_id' if 'user_id' in df.columns else None)
    if col_user:
        top_users = df[col_user].value_counts().head(5).reset_index()
        top_users.columns = ['usuario', 'qtd']
        context['top_10_users'] = top_users.to_dict('records')
        
        # Padrão detectado
        context['padrao_usuarios'] = (
            "Observa-se com frequência o nome 'A Customer' ou perfis anônimos liderando a lista. "
            "Isso ocorre pois muitos usuários não configuram perfil público. Além disso, existe o padrão de "
            "'Super Reviewers' (Lei de Pareto), onde poucos usuários dedicados fazem dezenas de avaliações."
        )

    # --- 6. GRÁFICOS RESTANTES ---
    if 'review_score' in df.columns:
        plt.figure(figsize=(6,3.5))
        sns.countplot(x='review_score', data=df, palette='viridis')
        plt.title('Distribuição de Notas')
        context['grafico_notas'] = get_graph()

    if 'price' in df.columns and 'review_score' in df.columns:
        plt.figure(figsize=(6,3.5))
        sns.scatterplot(x='price', y='review_score', data=df, alpha=0.5, color='#198754')
        plt.title('Preço vs Nota')
        context['grafico_scatter'] = get_graph()

    return context


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


    livros_qs = Avaliacao.objects.all().values()[:5000]
    df = pd.DataFrame(list(livros_qs))
    eda = gerar_analise_dados(df)

    df_html = get_dataframe(limit=50)


    context = {
        'result': result,
        'comment': comment,
        'df_html': df_html,
        'eda': eda,
    }

    return render(request, 'home.html', context)