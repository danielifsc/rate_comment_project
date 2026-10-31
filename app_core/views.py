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




   # print(stop_words)

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
    
    #print(f"Analisou{sentiment_normal}")
    #print("--------")
    #print(f"Analisou{sentiment_processado}")
    
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
    """Função auxiliar para converter gráficos do matplotlib em base64"""
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

    # Pré-processamento
    if 'review_time' in df.columns:
        df['review_time_dt'] = pd.to_datetime(df['review_time'], unit='s', errors='coerce')
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'review_score' in df.columns:
        df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')

    # 1. VISÃO GERAL (Registros, colunas e tipos)
    context['total_linhas'] = df.shape[0]
    context['total_colunas'] = df.shape[1]
    context['tipos_dados'] = [{'coluna': col, 'tipo': str(dtype)} for col, dtype in df.dtypes.items()]

    # 2. DADOS FALTANTES
    nulos = df.isnull().sum()
    nulos_prop = (df.isnull().sum() / len(df)) * 100
    df_nulos = pd.DataFrame({'quantidade': nulos, 'proporcao': nulos_prop})
    df_nulos = df_nulos[df_nulos['quantidade'] > 0].sort_values(by='quantidade', ascending=False)
    context['dados_faltantes'] = [{'coluna': idx, 'qtd': row['quantidade'], 'prop': round(row['proporcao'], 2)} for idx, row in df_nulos.iterrows()]

    # 3. COLUNAS MAIS IMPORTANTES
    context['analise_colunas'] = (
        "As colunas mais relevantes são 'review_score' (mede a satisfação direta), 'review_text' (permite análise de sentimentos e tópicos), "
        "'price' (ajuda a entender a relação custo-benefício) e 'review_helpfulness' (indica a confiabilidade da avaliação para a comunidade)."
    )

    # 4 e 18. LIVROS MAIS POPULARES / TOP 10 AVALIADOS
    if 'title' in df.columns:
        top_books = df['title'].value_counts().head(10).reset_index()
        top_books.columns = ['titulo', 'qtd_avaliacoes']
        context['top_10_livros'] = top_books.to_dict('records')
        context['livros_unicos'] = df['title'].nunique() # 10. Livros únicos
        context['media_avals_por_livro'] = round(context['total_linhas'] / context['livros_unicos'], 2) if context['livros_unicos'] > 0 else 0 # 16. Média avaliações/livro

    # 5, 11 e 12. PREÇOS
    if 'price' in df.columns:
        context['preco_medio'] = round(df['price'].mean(), 2)
        context['livro_mais_caro'] = df.loc[df['price'].idxmax()]['title'] if not df['price'].dropna().empty else "-"
        context['preco_mais_caro'] = df['price'].max()
        context['livros_preco_zero'] = len(df[df['price'] == 0])

    # 6 e 14. DISTRIBUIÇÃO E ESTATÍSTICAS DE NOTAS
    if 'review_score' in df.columns:
        context['media_notas'] = round(df['review_score'].mean(), 2)
        context['mediana_notas'] = df['review_score'].median()
        context['desvio_notas'] = round(df['review_score'].std(), 2)

        pos = len(df[df['review_score'] >= 4])
        neg = len(df[df['review_score'] <= 2])
        context['comentario_sentimento'] = f"Há predominância de avaliações {'POSITIVAS' if pos > neg else 'NEGATIVAS'} ({pos} positivas vs {neg} negativas)."

        # Gráfico 1: Barras de Notas
        plt.figure(figsize=(6, 4))
        sns.countplot(x='review_score', data=df, hue='review_score', palette='viridis', legend=False)
        plt.title('Distribuição das Notas (Review Score)')
        plt.xlabel('Nota')
        plt.ylabel('Frequência')
        context['grafico_notas'] = get_graph()

    # 7 e 15. RELAÇÃO PREÇO vs NOTA (e BOXPLOT)
    if 'price' in df.columns and 'review_score' in df.columns:
        correlacao = df['price'].corr(df['review_score'])
        context['correlacao_preco_nota'] = round(correlacao, 3)
        context['analise_correlacao'] = "Sim, há forte correlação" if abs(correlacao) > 0.5 else "Não há uma tendência forte de que livros mais caros recebam notas melhores, a correlação é fraca."

        # Gráfico 2: Dispersão (Scatter)
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='price', y='review_score', data=df, alpha=0.5, color='#198754')
        plt.title('Dispersão: Preço vs Nota')
        context['grafico_scatter'] = get_graph()

        # Gráfico 3: Boxplot (Preço por Nota Média)
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='review_score', y='price', data=df, hue='review_score', palette='Set2', legend=False)
        plt.title('Variação de Preço por Nota (Boxplot)')
        context['grafico_boxplot'] = get_graph()

    # 8 e 20. TEMPO E EVOLUÇÃO
    if 'review_time_dt' in df.columns:
        df['ano_mes'] = df['review_time_dt'].dt.to_period('M')
        avals_tempo = df.groupby('ano_mes').size()
        
        # Gráfico 4: Evolução Temporal
        plt.figure(figsize=(8, 4))
        avals_tempo.plot(color='#0d6efd', linewidth=2)
        plt.title('Evolução Temporal das Avaliações')
        plt.xlabel('Ano/Mês')
        plt.ylabel('Quantidade')
        context['grafico_temporal'] = get_graph()

        # Meses de pico
        top_meses = avals_tempo.sort_values(ascending=False).head(3)
        context['meses_pico'] = [{'periodo': str(p), 'qtd': q} for p, q in top_meses.items()]

    # 9 e 19. USUÁRIOS
    col_user = 'profile_name' if 'profile_name' in df.columns else ('user_id' if 'user_id' in df.columns else None)
    if col_user:
        top_users = df[col_user].value_counts().head(10).reset_index()
        top_users.columns = ['usuario', 'qtd']
        context['top_10_users'] = top_users.to_dict('records')
        context['padrao_usuarios'] = "Muitos usuários aparecem como 'A Customer' (anônimos) e percebe-se um padrão de super-avaliadores, onde poucos usuários concentram muitas resenhas."

    # 21. AVALIAÇÕES MAIS ÚTEIS
    if 'review_helpfulness' in df.columns:
        def calc_help(val):
            try:
                num, den = str(val).split('/')
                return (int(num) / int(den)) if int(den) > 0 else 0
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
                'texto': str(row.get('review_text', ''))[:100] + '...'
            })

    # 22. NUVEM DE PALAVRAS
    col_texto = 'review_summary' if 'review_summary' in df.columns else ('review_text' if 'review_text' in df.columns else None)
    if col_texto:
        texto_completo = " ".join(df[col_texto].dropna().astype(str).tolist())
        if texto_completo.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(texto_completo)
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            context['grafico_nuvem'] = get_graph()
            
    # --- ANÁLISE GLOBAL DE SENTIMENTOS (VADER) ---
    col_texto = 'review_text' if 'review_text' in df.columns else ('review_summary' if 'review_summary' in df.columns else None)
    
    if col_texto and col_texto in df.columns:
        def obter_compound(texto):
            if not texto or not isinstance(texto, str):
                return 0.0
            texto_limpo = pre_processamento(texto)
            return sia.polarity_scores(texto_limpo)['compound']

        # Calcula o compound score para cada registro
        df['compound_score'] = df[col_texto].apply(obter_compound)
        
        # Contagem das categorias
        pos_count = int((df['compound_score'] >= 0.05).sum())
        neg_count = int((df['compound_score'] <= -0.05).sum())
        neu_count = int(len(df) - (pos_count + neg_count))
        
        # Média geral do compound
        media_comp = round(float(df['compound_score'].mean()), 4)
        
        # Classificação geral baseada na média
        if media_comp >= 0.05:
            sent_geral = 'Positiva 🟢'
        elif media_comp <= -0.05:
            sent_geral = 'Negativa 🔴'
        else:
            sent_geral = 'Neutra ⚪'

        context['total_positivas'] = pos_count
        context['total_neutras'] = neu_count
        context['total_negativas'] = neg_count
        context['media_compound'] = media_comp
        context['sentimento_geral_rotulo'] = sent_geral

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

    df_html = gerar_tabela_html_com_sentimentos(df, limite=50)


    context = {
        'result': result,
        'comment': comment,
        'df_html': df_html,
        'eda': eda,
    }

    return render(request, 'home.html', context)


def calcular_sentimento_linha(texto):
    """
    Recebe um texto, passa pela sua função analyze_user_comment e retorna 
    uma Series do Pandas com a Pontuação e a Classificação visual.
    """
    analise = analyze_user_comment(texto)
    
    if analise:
        score = analise['scores_processado']['compound']
        sentimento = analise['sentiment_processado']
        
        if sentimento == 'Positivo':
            sentimento_visual = 'Positiva 🟢'
        elif sentimento == 'Negativo':
            sentimento_visual = 'Negativa 🔴'
        else:
            sentimento_visual = 'Neutra ⚪'
            
        return pd.Series([round(score, 4), sentimento_visual])
    
    # Fallback caso a análise falhe ou retorne vazio
    return pd.Series([0.0, 'Neutra ⚪'])

def gerar_tabela_html_com_sentimentos(df, limite=50):
    """
    Recebe o DataFrame completo, aplica o limite de linhas para a tela,
    gera as colunas de sentimento e retorna a string HTML da tabela.
    """
    df_table = df.head(limite).copy()
    col_texto = 'review_text' if 'review_text' in df_table.columns else 'review_summary'

    if col_texto not in df_table.columns:
        return "<p>Sem dados de texto para exibir.</p>"

    # Aplica a função de linha para criar as novas colunas
    df_table[['Pontuação Geral', 'Análise de Sentimento']] = df_table[col_texto].apply(calcular_sentimento_linha)

    # Remove colunas auxiliares geradas pela função EDA para não poluir a tela
    colunas_indesejadas = ['review_time_dt', 'ano_mes', 'help_score']
    colunas_exibicao = [c for c in df_table.columns if c not in colunas_indesejadas]
    
    # Retorna o HTML estilizado
    return df_table[colunas_exibicao].to_html(
        classes='table table-sm table-striped table-hover border text-start mt-3', 
        index=False,
        justify='left'
    )