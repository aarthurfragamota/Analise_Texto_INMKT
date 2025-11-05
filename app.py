import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import stanza
import os
import base64
import zipfile
from PIL import Image
import io
import re
import os
import tempfile
from pathlib import Path

# Configurações de cache para diferentes ambientes
def setup_environment():
    if 'STREAMLIT_CLOUD' in os.environ:
        # Streamlit Cloud - usar diretórios temporários
        base_dir = Path(tempfile.gettempdir())
        stanza_dir = base_dir / 'stanza_resources'
        hf_dir = base_dir / 'hf_cache'
        
        # Criar diretórios se não existirem
        stanza_dir.mkdir(exist_ok=True)
        hf_dir.mkdir(exist_ok=True)
        
        os.environ['STANZA_RESOURCES_DIR'] = str(stanza_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(hf_dir)
        os.environ['NLTK_DATA'] = str(base_dir / 'nltk_data')
    else:
        # Ambiente local
        os.environ['STANZA_RESOURCES_DIR'] = r"D:\stanza_resources"
        os.environ['TRANSFORMERS_CACHE'] = r"D:\hf_cache"

# Chamar a configuração no início


# Download recursos (apenas primeira execução)
@st.cache_resource
def download_recursos():
    nltk.download('stopwords')
    try:
        stanza.download('pt')
    except:
        pass  # Já baixado

download_recursos()

# ========================
# CONFIGURAÇÃO STREAMLIT
# ========================

st.set_page_config(page_title="Analisador Completo de Texto", layout="wide")
st.title("Analisador de Sentimento e Sintaxe")

# ========================
# INICIALIZAR SESSION STATE
# ========================

if 'analise_feita' not in st.session_state:
    st.session_state.analise_feita = False
if 'df_analisado' not in st.session_state:
    st.session_state.df_analisado = None
if 'df_sintatica' not in st.session_state:
    st.session_state.df_sintatica = None
if 'nuvens' not in st.session_state:
    st.session_state.nuvens = {}
if 'text_col' not in st.session_state:
    st.session_state.text_col = None
if 'tipo_analise' not in st.session_state:
    st.session_state.tipo_analise = None
if 'textos_originais' not in st.session_state:
    st.session_state.textos_originais = {}
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}
if 'nome_projeto' not in st.session_state:
    st.session_state.nome_projeto = "Analise_Texto"
if 'df_limpo' not in st.session_state:
    st.session_state.df_limpo = None

# ========================
# CARREGAR MODELOS
# ========================

_modelo_sentimento_carregado = False

@st.cache_resource
def carregar_modelo_sentimento():
    global _modelo_sentimento_carregado
    if not _modelo_sentimento_carregado:
        st.sidebar.info("Carregando modelo RoBERTuito...")
        _modelo_sentimento_carregado = True
    
    model_name = "pysentimiento/robertuito-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource
def carregar_modelo_sintaxe():
    return stanza.Pipeline('pt', use_gpu=torch.cuda.is_available())

# ========================
# FUNÇÕES DE LIMPEZA DE DADOS - NOVAS
# ========================

def limpar_dataframe(df, coluna_texto):
    """
    Remove completamente linhas com textos nulos, vazios ou inválidos
    Retorna DataFrame limpo e estatísticas da limpeza
    """
    total_original = len(df)
    
    # Fazer uma cópia para não modificar o original
    df_limpo = df.copy()
    
    # Converter a coluna de texto para string e remover espaços
    df_limpo[coluna_texto] = df_limpo[coluna_texto].astype(str).str.strip()
    
    # Criar máscara para textos válidos
    mascara_validos = (
        df_limpo[coluna_texto].notna() & 
        (df_limpo[coluna_texto] != "") & 
        (df_limpo[coluna_texto] != "nan") &
        (df_limpo[coluna_texto] != "None") &
        (df_limpo[coluna_texto].str.len() >= 3)  # Pelo menos 3 caracteres
    )
    
    # Aplicar filtro
    df_limpo = df_limpo[mascara_validos].reset_index(drop=True)
    
    # Estatísticas da limpeza
    stats_limpeza = {
        'total_original': total_original,
        'total_limpo': len(df_limpo),
        'removidos': total_original - len(df_limpo),
        'percentual_removido': round(((total_original - len(df_limpo)) / total_original) * 100, 2)
    }
    
    return df_limpo, stats_limpeza

def diagnosticar_textos_avancado(df, coluna_texto):
    """Diagnóstico detalhado dos textos antes da limpeza"""
    textos = df[coluna_texto].astype(str)
    
    stats = {
        'total': len(textos),
        'vazios': textos.isna().sum() + (textos.str.strip() == "").sum(),
        'apenas_espacos': (textos.str.strip() == "").sum() - textos.isna().sum(),
        'muito_curtos': (textos.str.len() < 3).sum(),
        'valores_nan': (textos == "nan").sum(),
        'valores_none': (textos == "None").sum(),
        'valores_null': (textos == "null").sum(),
        'exemplos_invalidos': []
    }
    
    # Coletar exemplos inválidos
    for i, texto in enumerate(textos.head(20)):
        texto_str = str(texto).strip()
        if (pd.isna(texto) or 
            texto_str == "" or 
            texto_str == "nan" or 
            texto_str == "None" or 
            texto_str == "null" or 
            len(texto_str) < 3):
            stats['exemplos_invalidos'].append(f"Linha {i+1}: '{texto_str}'")
            if len(stats['exemplos_invalidos']) >= 5:  # Limitar a 5 exemplos
                break
    
    return stats

# ========================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ========================

def preprocessar_texto_para_sentimento(texto):
    """Pré-processamento ESPECÍFICO para análise de sentimentos - menos agressivo"""
    if pd.isna(texto) or texto is None:
        return ""
    
    texto = str(texto).strip()
    
    if not texto:
        return ""
    
    # Manter a capitalização - importante para sentimentos!
    # Remover apenas URLs e caracteres muito problemáticos
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    # Limitar tamanho (modelo aceita até 512 tokens)
    texto = texto[:2000]  # Deixar margem para tokenização
    
    return texto.strip()

def preprocessar_texto_para_nuvens(texto, normalizar_minusculas=True, remover_acentos=True):
    """Pré-processamento para nuvens de palavras"""
    if pd.isna(texto) or texto is None:
        return ""
    
    texto = str(texto).strip()
    
    if not texto:
        return ""
    
    # Normalizar para minúsculas (apenas para nuvens)
    if normalizar_minusculas:
        texto = texto.lower()
    
    # Limpeza mais agressiva para nuvens
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    if remover_acentos:
        import unicodedata
        texto = ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        )
    
    return texto.strip()

# ========================
# FUNÇÕES DE ANÁLISE DE SENTIMENTO
# ========================

def classificar_sentimento_batch_melhorado(_sentiment_analyzer, textos):
    """Versão SIMPLIFICADA - agora recebe apenas textos válidos"""
    resultados = []
    
    # Coletar estatísticas para debug
    debug_stats = {
        'total_textos': len(textos),
        'erros_analise': 0
    }
    
    # Analisar em lotes menores para maior estabilidade
    batch_size = 8
    
    for i in range(0, len(textos), batch_size):
        batch_textos = textos[i:i + batch_size]
        
        try:
            resultados_batch = _sentiment_analyzer(batch_textos)
            
            for resultado in resultados_batch:
                label = resultado['label']
                score = resultado['score']
                
                # CONFIANÇA MODERADA - limiar de 0.4
                limiar_confianca = 0.4
                
                if score >= limiar_confianca:
                    if label == 'POS':
                        resultados.append(("POSITIVO", score))
                    elif label == 'NEG':
                        resultados.append(("NEGATIVO", score))
                    elif label == 'NEU':
                        resultados.append(("NEUTRO", score))
                    else:
                        resultados.append(("NEUTRO", 0.5))
                else:
                    # Score baixo - classificar como NEUTRO com score real
                    resultados.append(("NEUTRO", score))
                    
        except Exception as e:
            debug_stats['erros_analise'] += 1
            st.warning(f"Erro no lote {i//batch_size + 1}: {str(e)[:100]}...")
            
            # Fallback: análise individual para este lote
            for texto in batch_textos:
                try:
                    resultado_individual = _sentiment_analyzer(texto)[0]
                    label = resultado_individual['label']
                    score = resultado_individual['score']
                    
                    limiar_confianca = 0.4
                    
                    if score >= limiar_confianca:
                        if label == 'POS':
                            resultados.append(("POSITIVO", score))
                        elif label == 'NEG':
                            resultados.append(("NEGATIVO", score))
                        elif label == 'NEU':
                            resultados.append(("NEUTRO", score))
                        else:
                            resultados.append(("NEUTRO", 0.5))
                    else:
                        resultados.append(("NEUTRO", score))
                        
                except Exception as e2:
                    # ÚLTIMO RECURSO: classificar como NEUTRO com baixa confiança
                    resultados.append(("NEUTRO", 0.3))
    
    # Garantir que temos resultados para todos os textos
    while len(resultados) < len(textos):
        resultados.append(("NEUTRO", 0.3))
    
    st.session_state.debug_info = debug_stats
    return resultados

# ========================
# FUNÇÕES DE NUVENS DE PALAVRAS
# ========================

def gerar_nuvem_dinamica(textos, titulo, max_palavras=100, min_caracteres=3, stopwords_set=None, remover_acentos=True):
    """Gera nuvem de palavras com parâmetros dinâmicos"""
    if textos.empty:
        st.warning(f"Nao ha dados para gerar a nuvem: {titulo}")
        return None
        
    texto = " ".join(textos.astype(str))
    
    # Aplicar pré-processamento específico para nuvens
    texto = preprocessar_texto_para_nuvens(texto, normalizar_minusculas=True, remover_acentos=remover_acentos)
    
    # Filtrar palavras por tamanho mínimo
    palavras = [p for p in texto.split() if len(p) >= min_caracteres]
    texto_filtrado = " ".join(palavras)
    
    if not texto_filtrado.strip():
        st.warning(f"Texto muito curto para gerar nuvem: {titulo}")
        return None

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=max_palavras,
        stopwords=stopwords_set or stop_words,
        collocations=False
    ).generate(texto_filtrado)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=16, pad=20)
    plt.tight_layout()
    
    # Converter figura para bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_nuvem_por_classe_dinamica(df_sintatica, classe_gramatical, titulo, max_palavras=100, min_caracteres=3, remover_acentos=True):
    """Gera nuvem de palavras para uma classe gramatical específica"""
    palavras_filtradas = df_sintatica[df_sintatica['classe_gramatical'] == classe_gramatical]
    
    if palavras_filtradas.empty:
        st.warning(f"Nao ha {titulo.lower()} para gerar nuvem")
        return None
        
    texto = " ".join(palavras_filtradas['palavra'].astype(str))
    
    # Aplicar pré-processamento
    texto = preprocessar_texto_para_nuvens(texto, normalizar_minusculas=True, remover_acentos=remover_acentos)
    
    # Filtrar palavras por tamanho mínimo
    palavras = [p for p in texto.split() if len(p) >= min_caracteres]
    texto_filtrado = " ".join(palavras)
    
    if not texto_filtrado.strip():
        return None

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=max_palavras,
        stopwords=stop_words,
        collocations=False
    ).generate(texto_filtrado)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=14, pad=15)
    plt.tight_layout()
    
    # Converter figura para bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ========================
# FUNÇÃO DE ANÁLISE SINTÁTICA
# ========================

def analisar_sintatica(textos):
    nlp = carregar_modelo_sintaxe()
    resultados = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    textos_lista = textos.tolist()
    
    for i, texto in enumerate(textos_lista):
        status_text.text(f"Analisando sintaxe: {i+1}/{len(textos_lista)}")
        progress_bar.progress((i + 1) / len(textos_lista))
        
        try:
            texto_processado = preprocessar_texto_para_sentimento(texto)
            if texto_processado:
                doc = nlp(texto_processado)
                for sent in doc.sentences:
                    for word in sent.words:
                        resultados.append({
                            "texto_original": texto[:100] + "..." if len(str(texto)) > 100 else texto,
                            "palavra": word.text,
                            "lema": word.lemma,
                            "classe_gramatical": word.upos,
                            "classe_detalhada": word.xpos,
                            "caracteristicas": word.feats,
                            "relacao_sintatica": word.deprel,
                            "palavra_pai": sent.words[word.head-1].text if word.head > 0 else "ROOT"
                        })
        except Exception as e:
            st.warning(f"Erro na analise sintatica do texto {i+1}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(resultados)

# ========================
# SIDEBAR - CONFIGURAÇÕES
# ========================

with st.sidebar:
    st.header("Configuracoes")
    
    # Nome do projeto para download
    st.session_state.nome_projeto = st.text_input(
        "Nome do projeto (para download):", 
        value=st.session_state.nome_projeto,
        help="Este nome sera usado para nomear os arquivos de download"
    )
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("Envie sua planilha (.xlsx ou .csv)", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

        st.success("Arquivo carregado com sucesso!")
        
        with st.expander("Visualizar dados"):
            st.dataframe(df.head())
            st.write(f"Dimensoes: {df.shape[0]} linhas x {df.shape[1]} colunas")

        # Selecionar coluna de texto
        text_col = st.selectbox("Selecione a coluna de texto para analise:", df.columns)

        # Diagnóstico ANTES da limpeza - NOVO
        with st.expander("Diagnostico e Limpeza de Dados"):
            st.write("**Diagnostico dos Dados Antes da Limpeza:**")
            diagnostico = diagnosticar_textos_avancado(df, text_col)
            
            st.write(f"- Total de linhas: {diagnostico['total']}")
            st.write(f"- Vazios/invalidos: {diagnostico['vazios']}")
            st.write(f"- Apenas espacos: {diagnostico['apenas_espacos']}")
            st.write(f"- Muito curtos (<3 chars): {diagnostico['muito_curtos']}")
            st.write(f"- Valores 'nan': {diagnostico['valores_nan']}")
            st.write(f"- Valores 'None': {diagnostico['valores_none']}")
            st.write(f"- Valores 'null': {diagnostico['valores_null']}")
            
            if diagnostico['exemplos_invalidos']:
                st.write("**Exemplos de linhas que serao removidas:**")
                for exemplo in diagnostico['exemplos_invalidos']:
                    st.write(f"  - {exemplo}")
            
            # Mostrar preview da limpeza
            if st.button("Preview da Limpeza"):
                df_limpo_preview, stats_limpeza = limpar_dataframe(df, text_col)
                st.write("**Resultado da Limpeza:**")
                st.write(f"- Linhas originais: {stats_limpeza['total_original']}")
                st.write(f"- Linhas apos limpeza: {stats_limpeza['total_limpo']}")
                st.write(f"- Linhas removidas: {stats_limpeza['removidos']} ({stats_limpeza['percentual_removido']}%)")
                st.write("**Preview dos dados limpos:**")
                st.dataframe(df_limpo_preview.head())

        # Tipo de análise
        st.markdown("---")
        st.subheader("Tipo de Analise")
        
        tipo_analise = st.radio(
            "Selecione o tipo de analise:",
            ["Analise de Sentimento", "Analise Sintatica", "Nuvem de Palavras", "Analise Completa"]
        )

        # Configurações iniciais para nuvens
        st.markdown("---")
        with st.expander("Configuracoes Iniciais"):
            # Stopwords personalizadas
            stopwords_input = st.text_area("Stopwords personalizadas (separe por virgula):", "",
                                          help="Palavras que devem ser ignoradas na analise")
            custom_stopwords = set([w.strip().lower() for w in stopwords_input.split(",") if w.strip()])
            stop_words = set(stopwords.words("portuguese")).union(custom_stopwords)

            # Configurações básicas para nuvem
            incluir_acentos = st.checkbox("Manter acentos", value=False, help="Manter acentuacao nas palavras")

        # Botão de execução
        st.markdown("---")
        executar_analise = st.button("Executar Analise", type="primary", use_container_width=True)
        
        # Se clicou em executar análise, armazenar nos estados
        if executar_analise:
            # LIMPEZA DOS DADOS ANTES DE QUALQUER ANÁLISE - CORREÇÃO CRÍTICA
            with st.spinner("Limpando dados..."):
                df_limpo, stats_limpeza = limpar_dataframe(df, text_col)
                st.session_state.df_limpo = df_limpo
                
                st.sidebar.success(f"Dados limpos! {stats_limpeza['removidos']} linhas invalidas removidas")
            
            st.session_state.analise_feita = True
            st.session_state.df_original = df_limpo.copy()  # Usar o DataFrame LIMPO
            st.session_state.text_col = text_col
            st.session_state.tipo_analise = tipo_analise
            st.session_state.stop_words = stop_words
            st.session_state.incluir_acentos = incluir_acentos
            st.session_state.custom_stopwords = custom_stopwords
            
            # Armazenar textos originais para regeneração dinâmica
            st.session_state.textos_originais = {
                'geral': df_limpo[text_col].copy(),
                'positivos': None,
                'negativos': None,
                'neutros': None
            }
            
            # Executar análises e armazenar resultados
            with st.spinner("Executando analises..."):
                # Variáveis para armazenar resultados
                df_analisado = df_limpo.copy()  # Trabalhar com dados LIMPOS
                df_sintatica_result = None
                nuvens_result = {}
                
                # Análise de Sentimento
                if tipo_analise in ["Analise de Sentimento", "Analise Completa"]:
                    with st.spinner("Carregando modelo de sentimentos..."):
                        sentiment_analyzer = carregar_modelo_sentimento()
                    
                    with st.spinner("Analisando sentimentos..."):
                        try:
                            # AGORA todos os textos já são válidos!
                            textos_para_analise = df_analisado[text_col].astype(str).tolist()
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # USAR A NOVA FUNÇÃO SIMPLIFICADA
                            resultados = classificar_sentimento_batch_melhorado(sentiment_analyzer, textos_para_analise)
                            
                            sentimentos = [r[0] for r in resultados]
                            scores = [r[1] for r in resultados]
                            
                            df_analisado["SENTIMENTO"] = sentimentos
                            df_analisado["SCORE_SENTIMENTO"] = scores
                            
                            progress_bar.progress(1.0)
                            status_text.text("Concluido!")
                            
                            # Armazenar textos separados por sentimento
                            st.session_state.textos_originais['positivos'] = df_analisado[df_analisado["SENTIMENTO"] == "POSITIVO"][text_col].copy()
                            st.session_state.textos_originais['negativos'] = df_analisado[df_analisado["SENTIMENTO"] == "NEGATIVO"][text_col].copy()
                            st.session_state.textos_originais['neutros'] = df_analisado[df_analisado["SENTIMENTO"] == "NEUTRO"][text_col].copy()
                            
                            # Mostrar estatísticas de debug
                            if 'debug_info' in st.session_state:
                                debug = st.session_state.debug_info
                                st.sidebar.info(f"Analise: {debug['total_textos']} textos, {debug['erros_analise']} erros")
                            
                        except Exception as e:
                            st.error(f"Erro na analise de sentimentos: {e}")

                # Análise Sintática
                if tipo_analise in ["Analise Sintatica", "Analise Completa"]:
                    with st.spinner("Analisando estrutura sintatica..."):
                        try:
                            textos_para_sintaxe = df_analisado[text_col].head(50)  # Já são válidos
                            if len(textos_para_sintaxe) > 0:
                                df_sintatica_result = analisar_sintatica(textos_para_sintaxe)
                            else:
                                st.warning("Nao ha textos validos para analise sintatica.")
                                
                        except Exception as e:
                            st.error(f"Erro na analise sintatica: {e}")

                # Armazenar resultados no session state
                st.session_state.df_analisado = df_analisado
                st.session_state.df_sintatica = df_sintatica_result
                st.session_state.nuvens = nuvens_result
                
            st.success("Analises concluidas!")
        
    else:
        st.info("Envie um arquivo para comecar a analise.")
        st.session_state.analise_feita = False
        st.session_state.df_analisado = None
        st.session_state.df_sintatica = None
        st.session_state.nuvens = {}
        st.session_state.df_limpo = None

    # Status do sistema
    st.markdown("---")
    st.header("Status do Sistema")
    st.write(f"GPU disponivel: {'Sim' if torch.cuda.is_available() else 'Nao'}")
    st.write(f"Analise concluida: {'Sim' if st.session_state.analise_feita else 'Nao'}")
    if st.session_state.df_limpo is not None:
        st.write(f"Dados limpos: {len(st.session_state.df_limpo)} linhas validas")

# ========================
# ÁREA PRINCIPAL - RESULTADOS
# ========================

if st.session_state.analise_feita and st.session_state.df_analisado is not None:
    df_analisado = st.session_state.df_analisado
    df_sintatica = st.session_state.df_sintatica
    text_col = st.session_state.text_col
    tipo_analise = st.session_state.tipo_analise
    stop_words = st.session_state.stop_words
    nome_projeto = st.session_state.nome_projeto
    
    # ========================
    # PAINEL DE CONTROLE DINÂMICO
    # ========================
    st.markdown("---")
    st.subheader("Controle de Nuvens de Palavras")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_palavras = st.number_input("Maximo de palavras por nuvem:", 
                                     min_value=1, max_value=500, value=50, step=1)
    with col2:
        min_caracteres = st.number_input("Minimo de caracteres por palavra:", 
                                       min_value=1, max_value=20, value=3, step=1)
    with col3:
        incluir_acentos_dinamico = st.checkbox("Manter acentos nas nuvens", 
                                             value=st.session_state.incluir_acentos)
    
    # Atualizar stopwords dinamicamente
    stopwords_dinamicas = st.text_area("Stopwords (atualize e aplique):", 
                                     value=", ".join(st.session_state.custom_stopwords),
                                     help="Edite as stopwords e aplique para atualizar as nuvens")
    
    if st.button("Aplicar Configuracoes nas Nuvens"):
        st.session_state.stop_words = set(stopwords.words("portuguese")).union(
            set([w.strip().lower() for w in stopwords_dinamicas.split(",") if w.strip()])
        )
        st.session_state.incluir_acentos = incluir_acentos_dinamico
        st.rerun()

    # ========================
    # EXIBIR RESULTADOS EM EXPANDERS
    # ========================
    
    # Nuvem Principal
    with st.expander("NUVEM PRINCIPAL", expanded=True):
        if tipo_analise in ["Nuvem de Palavras", "Analise Completa"]:
            nuvem_principal = gerar_nuvem_dinamica(
                st.session_state.textos_originais['geral'], 
                "Nuvem de Palavras Geral",
                max_palavras=max_palavras,
                min_caracteres=min_caracteres,
                remover_acentos=not incluir_acentos_dinamico
            )
            if nuvem_principal:
                st.image(nuvem_principal, use_container_width=True)
                st.session_state.nuvens["nuvem_principal"] = nuvem_principal
            
            if "nuvem_principal" in st.session_state.nuvens and st.session_state.nuvens["nuvem_principal"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=f"Baixar Nuvem Principal {nome_projeto}",
                        data=st.session_state.nuvens["nuvem_principal"].getvalue(),
                        file_name=f"Nuvem_Principal_{nome_projeto}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="download_nuvem_principal"
                    )
        else:
            st.info("Selecione 'Nuvem de Palavras' ou 'Analise Completa' para ver esta visualizacao.")

    # Análise de Sentimento
    with st.expander("ANALISE DE SENTIMENTO", expanded=True):
        if tipo_analise in ["Analise de Sentimento", "Analise Completa"] and "SENTIMENTO" in df_analisado.columns:
            st.success("Analise de sentimento concluida!")
            
            # PREVIEW DOS RESULTADOS - AGORA TODOS SÃO VÁLIDOS
            st.subheader("Preview dos Resultados")
            col1, col2 = st.columns(2)
            with col1:
                # Mostrar primeiras linhas com sentimentos
                st.write("**Primeiras classificacoes:**")
                preview_df = df_analisado[[text_col, 'SENTIMENTO', 'SCORE_SENTIMENTO']].head(10)
                st.dataframe(preview_df)
            
            with col2:
                # Métricas
                contagem_sentimentos = df_analisado["SENTIMENTO"].value_counts()
                st.write("**Distribuicao de Sentimentos:**")
                st.dataframe(contagem_sentimentos)
                st.bar_chart(contagem_sentimentos)
            
            # Nuvens por sentimento
            st.subheader("Nuvens por Sentimento")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.textos_originais['positivos'] is not None and not st.session_state.textos_originais['positivos'].empty:
                    nuvem_positiva = gerar_nuvem_dinamica(
                        st.session_state.textos_originais['positivos'],
                        "Nuvem Positiva",
                        max_palavras=max_palavras,
                        min_caracteres=min_caracteres,
                        remover_acentos=not incluir_acentos_dinamico
                    )
                    if nuvem_positiva:
                        st.image(nuvem_positiva, use_container_width=True)
                        st.session_state.nuvens["nuvem_positiva"] = nuvem_positiva
            
            with col2:
                if st.session_state.textos_originais['negativos'] is not None and not st.session_state.textos_originais['negativos'].empty:
                    nuvem_negativa = gerar_nuvem_dinamica(
                        st.session_state.textos_originais['negativos'],
                        "Nuvem Negativa",
                        max_palavras=max_palavras,
                        min_caracteres=min_caracteres,
                        remover_acentos=not incluir_acentos_dinamico
                    )
                    if nuvem_negativa:
                        st.image(nuvem_negativa, use_container_width=True)
                        st.session_state.nuvens["nuvem_negativa"] = nuvem_negativa
            
            col3, col4 = st.columns(2)
            with col3:
                if st.session_state.textos_originais['neutros'] is not None and not st.session_state.textos_originais['neutros'].empty:
                    nuvem_neutra = gerar_nuvem_dinamica(
                        st.session_state.textos_originais['neutros'],
                        "Nuvem Neutra",
                        max_palavras=max_palavras,
                        min_caracteres=min_caracteres,
                        remover_acentos=not incluir_acentos_dinamico
                    )
                    if nuvem_neutra:
                        st.image(nuvem_neutra, use_container_width=True)
                        st.session_state.nuvens["nuvem_neutra"] = nuvem_neutra
            
            # Botão de download para esta seção
            col1, col2 = st.columns(2)
            with col1:
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    df_sentimento = df_analisado[[text_col, 'SENTIMENTO', 'SCORE_SENTIMENTO']]
                    df_sentimento.to_excel(writer, sheet_name='Analise_Sentimentos', index=False)
                    stats_sentimentos = df_analisado['SENTIMENTO'].value_counts().reset_index()
                    stats_sentimentos.columns = ['Sentimento', 'Quantidade']
                    stats_sentimentos.to_excel(writer, sheet_name='Estatisticas', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label=f"Baixar Dados Sentimento {nome_projeto}",
                    data=output_buffer,
                    file_name=f"Analise_Sentimento_{nome_projeto}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_sentimento_excel"
                )
            
            with col2:
                nuvens_sentimento = {k: v for k, v in st.session_state.nuvens.items() if k in ['nuvem_positiva', 'nuvem_negativa', 'nuvem_neutra']}
                if nuvens_sentimento:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for nome, nuvem in nuvens_sentimento.items():
                            nome_arquivo = f"{nome.replace('nuvem_', '').capitalize()}_{nome_projeto}.png"
                            zip_file.writestr(nome_arquivo, nuvem.getvalue())
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label=f"Baixar Nuvens Sentimento {nome_projeto}",
                        data=zip_buffer,
                        file_name=f"Nuvens_Sentimento_{nome_projeto}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_sentimento_zip"
                    )
        else:
            st.info("Selecione 'Analise de Sentimento' ou 'Analise Completa' para ver esta analise.")

    # Análise Sintática
    with st.expander("ANALISE SINTATICA", expanded=True):
        if tipo_analise in ["Analise Sintatica", "Analise Completa"] and df_sintatica is not None:
            st.success(f"Analise sintatica concluida! {len(df_sintatica)} palavras analisadas.")
            
            # Estatísticas
            st.subheader("Distribuicao de Classes Gramaticais")
            contagem_classes = df_sintatica['classe_gramatical'].value_counts()
            st.dataframe(contagem_classes)
            
            # Nuvens por classe gramatical
            st.subheader("Nuvens por Classe Gramatical")
            
            col1, col2, col3 = st.columns(3)
            nuvens_sintaticas = {}
            
            with col1:
                nuvem_verbos = gerar_nuvem_por_classe_dinamica(
                    df_sintatica, "VERB", "Verbos",
                    max_palavras=max_palavras,
                    min_caracteres=min_caracteres,
                    remover_acentos=not incluir_acentos_dinamico
                )
                if nuvem_verbos:
                    st.image(nuvem_verbos, use_container_width=True)
                    nuvens_sintaticas["verbos"] = nuvem_verbos
            
            with col2:
                nuvem_adjetivos = gerar_nuvem_por_classe_dinamica(
                    df_sintatica, "ADJ", "Adjetivos",
                    max_palavras=max_palavras,
                    min_caracteres=min_caracteres,
                    remover_acentos=not incluir_acentos_dinamico
                )
                if nuvem_adjetivos:
                    st.image(nuvem_adjetivos, use_container_width=True)
                    nuvens_sintaticas["adjetivos"] = nuvem_adjetivos
            
            with col3:
                nuvem_substantivos = gerar_nuvem_por_classe_dinamica(
                    df_sintatica, "NOUN", "Substantivos",
                    max_palavras=max_palavras,
                    min_caracteres=min_caracteres,
                    remover_acentos=not incluir_acentos_dinamico
                )
                if nuvem_substantivos:
                    st.image(nuvem_substantivos, use_container_width=True)
                    nuvens_sintaticas["substantivos"] = nuvem_substantivos
            
            # Dados detalhados
            with st.expander("Visualizar analise sintatica completa"):
                st.dataframe(df_sintatica.head(100))
            
            # Botão de download para esta seção
            col1, col2 = st.columns(2)
            with col1:
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    df_sintatica.to_excel(writer, sheet_name='Analise_Sintatica', index=False)
                    stats_classes = df_sintatica['classe_gramatical'].value_counts().reset_index()
                    stats_classes.columns = ['Classe_Gramatical', 'Quantidade']
                    stats_classes.to_excel(writer, sheet_name='Estatisticas', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label=f"Baixar Dados Sintaticos {nome_projeto}",
                    data=output_buffer,
                    file_name=f"Analise_Sintatica_{nome_projeto}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_sintatica_excel"
                )
            
            with col2:
                if nuvens_sintaticas:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for nome, nuvem in nuvens_sintaticas.items():
                            nome_arquivo = f"Nuvem_{nome.capitalize()}_{nome_projeto}.png"
                            zip_file.writestr(nome_arquivo, nuvem.getvalue())
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label=f"Baixar Nuvens Sintaticas {nome_projeto}",
                        data=zip_buffer,
                        file_name=f"Nuvens_Sintaticas_{nome_projeto}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_sintatica_zip"
                    )
        else:
            st.info("Selecione 'Analise Sintatica' ou 'Analise Completa' para ver esta analise.")

    # ========================
    # DOWNLOAD COMPLETO
    # ========================
    st.markdown("---")
    st.subheader("Download Completo")
    
    # Criar arquivo ZIP com tudo
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Adicionar Excel com múltiplas planilhas
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_analisado.to_excel(writer, sheet_name='Dados_Originais', index=False)
            
            if "SENTIMENTO" in df_analisado.columns:
                df_sentimento = df_analisado[[text_col, 'SENTIMENTO', 'SCORE_SENTIMENTO']]
                df_sentimento.to_excel(writer, sheet_name='Analise_Sentimentos', index=False)
                stats_sentimentos = df_analisado['SENTIMENTO'].value_counts().reset_index()
                stats_sentimentos.columns = ['Sentimento', 'Quantidade']
                stats_sentimentos.to_excel(writer, sheet_name='Estatisticas_Sentimentos', index=False)
            
            if df_sintatica is not None:
                df_sintatica.to_excel(writer, sheet_name='Analise_Sintatica', index=False)
                stats_classes = df_sintatica['classe_gramatical'].value_counts().reset_index()
                stats_classes.columns = ['Classe_Gramatical', 'Quantidade']
                stats_classes.to_excel(writer, sheet_name='Estatisticas_Sintaticas', index=False)
        
        excel_buffer.seek(0)
        zip_file.writestr(f"Resultados_Completos_{nome_projeto}.xlsx", excel_buffer.getvalue())
        
        # Adicionar todas as nuvens com nomes personalizados
        for nome, nuvem in st.session_state.nuvens.items():
            nome_arquivo = nome.replace('nuvem_', '').capitalize()
            zip_file.writestr(f"Nuvem_{nome_arquivo}_{nome_projeto}.png", nuvem.getvalue())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label=f"Baixar ZIP Completo {nome_projeto}",
        data=zip_buffer,
        file_name=f"Analise_Completa_{nome_projeto}.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_completo"
    )

else:
    # Mensagem inicial
    st.markdown("""
    ## Bem-vindo ao Analisador de Texto Completo!
    
    **Como usar:**
    1. **Upload**: Envie sua planilha na sidebar
    2. **Configuracao**: Selecione coluna e tipo de analise
    3. **Executar**: Clique no botao na sidebar
    4. **Resultados**: Visualize nas secoes expansíveis abaixo
    5. **Download**: Baixe resultados individuais ou completos
    
    **Recursos disponíveis:**
    - **Nuvens de palavras** personalizáveis
    - **Analise de sentimentos** com RoBERTuito
    - **Analise sintatica** com Stanza
    - **Nuvens por classe gramatical** (verbos, adjetivos, substantivos)
    """)
    
    with st.expander("Mais informacoes"):
        st.markdown("""
        **Tipos de Analise:**
        - **Sentimento**: Classifica textos como Positivo/Negativo/Neutro
        - **Sintatica**: Analisa estrutura gramatical + nuvens categorizadas
        - **Nuvem**: Visualizacao de palavras frequentes
        - **Completa**: Todas as analises combinadas
        
        **Formatos suportados:** CSV, Excel (XLSX)
        **Idioma:** Portugues
        """)

# ========================
# ESTILOS CSS
# ========================
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander > div:first-child {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)