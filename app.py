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

# Configurações de ambiente
os.environ['STANZA_RESOURCES_DIR'] = r"D:\stanza_resources"
os.environ['TRANSFORMERS_CACHE'] = r"D:\hf_cache"

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
# 🔧 CONFIGURAÇÃO STREAMLIT
# ========================
st.set_page_config(page_title="Analisador Completo de Texto", layout="wide")
st.title("📊 Analisador de Sentimento + Sintaxe + Nuvens")

# ========================
# ⚙️ INICIALIZAR SESSION STATE
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

# ========================
# ⚙️ CARREGAR MODELOS (COM CACHE CORRETO)
# ========================
_modelo_sentimento_carregado = False

@st.cache_resource
def carregar_modelo_sentimento():
    global _modelo_sentimento_carregado
    if not _modelo_sentimento_carregado:
        st.sidebar.info("Carregando modelo RoBERTuito... ⏳")
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
# 🧠 FUNÇÕES DE ANÁLISE
# ========================
def gerar_nuvem(textos, titulo, stopwords_set=None):
    if textos.empty:
        st.warning(f"⚠️ Não há dados para gerar a nuvem: {titulo}")
        return None
        
    textos = textos.dropna().astype(str)
    texto = " ".join(textos)
    
    if not incluir_acentos:
        import unicodedata
        texto = ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=top_n,
        stopwords=stopwords_set or stop_words,
        collocations=False
    ).generate(texto)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=14, pad=20)
    
    # Converter figura para bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_nuvem_por_classe(df_sintatica, classe_gramatical, titulo):
    """Gera nuvem de palavras para uma classe gramatical específica"""
    palavras_filtradas = df_sintatica[df_sintatica['classe_gramatical'] == classe_gramatical]
    
    if palavras_filtradas.empty:
        st.warning(f"⚠️ Não há {titulo.lower()} para gerar nuvem")
        return None
        
    texto = " ".join(palavras_filtradas['palavra'].astype(str))
    
    if not incluir_acentos:
        import unicodedata
        texto = ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        )

    wc = WordCloud(
        width=600,
        height=300,
        background_color="white",
        max_words=top_n,
        stopwords=stop_words,
        collocations=False
    ).generate(texto)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=12, pad=15)
    
    # Converter figura para bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def classificar_sentimento_batch(_sentiment_analyzer, textos):
    """Classifica sentimentos em lote para melhor performance"""
    resultados = []
    for texto in textos:
        try:
            if pd.isna(texto) or texto.strip() == "":
                resultados.append(("NEUTRO", 0.0))
                continue
                
            texto = str(texto)[:512]
            resultado = _sentiment_analyzer(texto)[0]
            label = resultado['label']
            score = resultado['score']
            
            if label == 'POS':
                resultados.append(("POSITIVO", score))
            elif label == 'NEG':
                resultados.append(("NEGATIVO", score))
            else:
                resultados.append(("NEUTRO", score))
        except Exception as e:
            st.warning(f"Erro ao analisar texto: {e}")
            resultados.append(("ERRO", 0.0))
            
    return resultados

def analisar_sintatica(textos):
    nlp = carregar_modelo_sintaxe()
    resultados = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    textos_lista = textos.dropna().tolist()
    
    for i, texto in enumerate(textos_lista):
        status_text.text(f"Analisando sintaxe: {i+1}/{len(textos_lista)}")
        progress_bar.progress((i + 1) / len(textos_lista))
        
        try:
            doc = nlp(str(texto))
            for sent in doc.sentences:
                for word in sent.words:
                    resultados.append({
                        "texto_original": texto[:100] + "..." if len(str(texto)) > 100 else texto,
                        "palavra": word.text,
                        "lema": word.lemma,
                        "classe_gramatical": word.upos,
                        "classe_detalhada": word.xpos,
                        "características": word.feats,
                        "relacao_sintatica": word.deprel,
                        "palavra_pai": sent.words[word.head-1].text if word.head > 0 else "ROOT"
                    })
        except Exception as e:
            st.warning(f"Erro na análise sintática do texto {i+1}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(resultados)

# ========================
# 📁 SIDEBAR - CONFIGURAÇÕES
# ========================
with st.sidebar:
    st.header("⚙️ Configurações")
    
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

        st.success("✅ Arquivo carregado com sucesso!")
        
        with st.expander("📋 Visualizar dados"):
            st.dataframe(df.head())
            st.write(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")

        # Selecionar coluna de texto
        text_col = st.selectbox("Selecione a coluna de texto para análise:", df.columns)

        # Tipo de análise
        st.markdown("---")
        st.subheader("🎯 Tipo de Análise")
        
        tipo_analise = st.radio(
            "Selecione o tipo de análise:",
            ["Análise de Sentimento", "Análise Sintática", "Nuvem de Palavras", "Análise Completa"]
        )

        # Configurações avançadas
        st.markdown("---")
        with st.expander("🔧 Configurações Avançadas"):
            # Stopwords personalizadas
            stopwords_input = st.text_area("Stopwords personalizadas (separe por vírgula):", "",
                                          help="Palavras que devem ser ignoradas na análise")
            custom_stopwords = set([w.strip().lower() for w in stopwords_input.split(",") if w.strip()])
            stop_words = set(stopwords.words("portuguese")).union(custom_stopwords)

            # Configurações para nuvem
            top_n = st.slider("Número máximo de palavras:", 50, 500, 100)
            min_chars = st.slider("Mínimo de caracteres por palavra:", 1, 10, 3)
            incluir_acentos = st.checkbox("Manter acentos", value=True)

        # Botão de execução
        st.markdown("---")
        executar_analise = st.button("🔍 Executar Análise", type="primary", use_container_width=True)
        
        # Se clicou em executar análise, armazenar nos estados
        if executar_analise:
            st.session_state.analise_feita = True
            st.session_state.df_original = df.copy()
            st.session_state.text_col = text_col
            st.session_state.tipo_analise = tipo_analise
            st.session_state.stop_words = stop_words
            st.session_state.top_n = top_n
            st.session_state.min_chars = min_chars
            st.session_state.incluir_acentos = incluir_acentos
            st.session_state.custom_stopwords = custom_stopwords
            
            # Executar análises e armazenar resultados
            with st.spinner("Executando análises..."):
                # Variáveis para armazenar resultados
                df_analisado = df.copy()
                df_sintatica_result = None
                nuvens_result = {}
                
                # Análise de Sentimento
                if tipo_analise in ["Análise de Sentimento", "Análise Completa"]:
                    with st.spinner("🔄 Carregando modelo de sentimentos..."):
                        sentiment_analyzer = carregar_modelo_sentimento()
                    
                    with st.spinner("🔄 Analisando sentimentos..."):
                        try:
                            textos_para_analise = df_analisado[text_col].astype(str).tolist()
                            
                            sentimentos, scores = [], []
                            batch_size = 32
                            
                            for i in range(0, len(textos_para_analise), batch_size):
                                batch = textos_para_analise[i:i + batch_size]
                                batch_resultados = classificar_sentimento_batch(sentiment_analyzer, batch)
                                
                                for sentimento, score in batch_resultados:
                                    sentimentos.append(sentimento)
                                    scores.append(score)
                            
                            df_analisado["SENTIMENTO"] = sentimentos
                            df_analisado["SCORE_SENTIMENTO"] = scores
                            
                        except Exception as e:
                            st.error(f"❌ Erro na análise de sentimentos: {e}")

                # Análise Sintática
                if tipo_analise in ["Análise Sintática", "Análise Completa"]:
                    with st.spinner("🔄 Analisando estrutura sintática..."):
                        try:
                            textos_para_sintaxe = df_analisado[text_col].dropna().head(50)
                            if len(textos_para_sintaxe) > 0:
                                df_sintatica_result = analisar_sintatica(textos_para_sintaxe)
                            else:
                                st.warning("⚠️ Não há textos válidos para análise sintática.")
                                
                        except Exception as e:
                            st.error(f"❌ Erro na análise sintática: {e}")

                # Gerar nuvens
                if tipo_analise in ["Nuvem de Palavras", "Análise Completa"]:
                    with st.spinner("🔄 Gerando nuvens de palavras..."):
                        nuvem_principal = gerar_nuvem(df_analisado[text_col], "Nuvem de Palavras Geral")
                        if nuvem_principal:
                            nuvens_result["nuvem_principal"] = nuvem_principal
                        
                        if "SENTIMENTO" in df_analisado.columns:
                            nuvem_positiva = gerar_nuvem(df_analisado[df_analisado["SENTIMENTO"] == "POSITIVO"][text_col], "Nuvem Positiva")
                            if nuvem_positiva:
                                nuvens_result["nuvem_positiva"] = nuvem_positiva
                            
                            nuvem_negativa = gerar_nuvem(df_analisado[df_analisado["SENTIMENTO"] == "NEGATIVO"][text_col], "Nuvem Negativa")
                            if nuvem_negativa:
                                nuvens_result["nuvem_negativa"] = nuvem_negativa
                            
                            nuvem_neutra = gerar_nuvem(df_analisado[df_analisado["SENTIMENTO"] == "NEUTRO"][text_col], "Nuvem Neutra")
                            if nuvem_neutra:
                                nuvens_result["nuvem_neutra"] = nuvem_neutra
                
                # Armazenar resultados no session state
                st.session_state.df_analisado = df_analisado
                st.session_state.df_sintatica = df_sintatica_result
                st.session_state.nuvens = nuvens_result
                
            st.success("✅ Análises concluídas!")
        
    else:
        st.info("📂 Envie um arquivo para começar a análise.")
        # Resetar estados se não há arquivo
        st.session_state.analise_feita = False
        st.session_state.df_analisado = None
        st.session_state.df_sintatica = None
        st.session_state.nuvens = {}

    # Status do sistema
    st.markdown("---")
    st.header("📊 Status do Sistema")
    st.write(f"**GPU disponível:** {'✅' if torch.cuda.is_available() else '❌'}")
    st.write(f"**Análise concluída:** {'✅' if st.session_state.analise_feita else '❌'}")

# ========================
# 🎯 ÁREA PRINCIPAL - RESULTADOS
# ========================
# Usar variáveis do session state para evitar rerun
if st.session_state.analise_feita and st.session_state.df_analisado is not None:
    df_analisado = st.session_state.df_analisado
    df_sintatica = st.session_state.df_sintatica
    nuvens = st.session_state.nuvens
    text_col = st.session_state.text_col
    tipo_analise = st.session_state.tipo_analise
    stop_words = st.session_state.stop_words
    top_n = st.session_state.top_n
    min_chars = st.session_state.min_chars
    incluir_acentos = st.session_state.incluir_acentos

    # ========================
    # 📊 EXIBIR RESULTADOS EM EXPANDERS
    # ========================
    
    # Nuvem Principal
    with st.expander("☁️ NUVEM PRINCIPAL", expanded=True):
        if tipo_analise in ["Nuvem de Palavras", "Análise Completa"]:
            if "nuvem_principal" in nuvens and nuvens["nuvem_principal"]:
                st.image(nuvens["nuvem_principal"], use_column_width=True)
            
            # Botão de download para esta seção
            if "nuvem_principal" in nuvens and nuvens["nuvem_principal"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Baixar Nuvem Principal",
                        data=nuvens["nuvem_principal"].getvalue(),
                        file_name="nuvem_principal.png",
                        mime="image/png",
                        use_container_width=True,
                        key="download_nuvem_principal"
                    )
        else:
            st.info("Selecione 'Nuvem de Palavras' ou 'Análise Completa' para ver esta visualização.")

    # Análise de Sentimento
    with st.expander("🧠 ANÁLISE DE SENTIMENTO", expanded=True):
        if tipo_analise in ["Análise de Sentimento", "Análise Completa"] and "SENTIMENTO" in df_analisado.columns:
            st.success("✅ Análise de sentimento concluída!")
            
            # Métricas
            col1, col2 = st.columns(2)
            with col1:
                contagem_sentimentos = df_analisado["SENTIMENTO"].value_counts()
                st.bar_chart(contagem_sentimentos)
            with col2:
                st.dataframe(contagem_sentimentos)
            
            # Nuvens por sentimento
            st.subheader("☁️ Nuvens por Sentimento")
            col1, col2 = st.columns(2)
            
            with col1:
                if "nuvem_positiva" in nuvens and nuvens["nuvem_positiva"]:
                    st.image(nuvens["nuvem_positiva"], use_column_width=True)
            
            with col2:
                if "nuvem_negativa" in nuvens and nuvens["nuvem_negativa"]:
                    st.image(nuvens["nuvem_negativa"], use_column_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                if "nuvem_neutra" in nuvens and nuvens["nuvem_neutra"]:
                    st.image(nuvens["nuvem_neutra"], use_column_width=True)
            
            # Botão de download para esta seção
            col1, col2 = st.columns(2)
            with col1:
                # Preparar dados para download
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    df_sentimento = df_analisado[[text_col, 'SENTIMENTO', 'SCORE_SENTIMENTO']]
                    df_sentimento.to_excel(writer, sheet_name='Análise_Sentimentos', index=False)
                    stats_sentimentos = df_analisado['SENTIMENTO'].value_counts().reset_index()
                    stats_sentimentos.columns = ['Sentimento', 'Quantidade']
                    stats_sentimentos.to_excel(writer, sheet_name='Estatísticas', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label="📊 Baixar Dados Sentimento",
                    data=output_buffer,
                    file_name="analise_sentimento.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_sentimento_excel"
                )
            
            with col2:
                # Criar ZIP com nuvens de sentimento
                nuvens_sentimento = {k: v for k, v in nuvens.items() if k in ['nuvem_positiva', 'nuvem_negativa', 'nuvem_neutra']}
                if nuvens_sentimento:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for nome, nuvem in nuvens_sentimento.items():
                            zip_file.writestr(f"{nome}.png", nuvem.getvalue())
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="🖼️ Baixar Nuvens Sentimento",
                        data=zip_buffer,
                        file_name="nuvens_sentimento.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_sentimento_zip"
                    )
        else:
            st.info("Selecione 'Análise de Sentimento' ou 'Análise Completa' para ver esta análise.")

    # Análise Sintática
    with st.expander("📝 ANÁLISE SINTÁTICA", expanded=True):
        if tipo_analise in ["Análise Sintática", "Análise Completa"] and df_sintatica is not None:
            st.success(f"✅ Análise sintática concluída! {len(df_sintatica)} palavras analisadas.")
            
            # Estatísticas
            st.subheader("📊 Distribuição de Classes Gramaticais")
            contagem_classes = df_sintatica['classe_gramatical'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(contagem_classes)
            with col2:
                st.dataframe(contagem_classes)
            
            # Nuvens por classe gramatical
            st.subheader("☁️ Nuvens por Classe Gramatical")
            
            col1, col2, col3 = st.columns(3)
            nuvens_sintaticas = {}
            
            with col1:
                nuvem_verbos = gerar_nuvem_por_classe(df_sintatica, "VERB", "Verbos")
                if nuvem_verbos:
                    st.image(nuvem_verbos, use_column_width=True)
                    nuvens_sintaticas["verbos"] = nuvem_verbos
            
            with col2:
                nuvem_adjetivos = gerar_nuvem_por_classe(df_sintatica, "ADJ", "Adjetivos")
                if nuvem_adjetivos:
                    st.image(nuvem_adjetivos, use_column_width=True)
                    nuvens_sintaticas["adjetivos"] = nuvem_adjetivos
            
            with col3:
                nuvem_substantivos = gerar_nuvem_por_classe(df_sintatica, "NOUN", "Substantivos")
                if nuvem_substantivos:
                    st.image(nuvem_substantivos, use_column_width=True)
                    nuvens_sintaticas["substantivos"] = nuvem_substantivos
            
            # Dados detalhados
            with st.expander("🔍 Visualizar análise sintática completa"):
                st.dataframe(df_sintatica.head(100))
            
            # Botão de download para esta seção
            col1, col2 = st.columns(2)
            with col1:
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    df_sintatica.to_excel(writer, sheet_name='Análise_Sintática', index=False)
                    stats_classes = df_sintatica['classe_gramatical'].value_counts().reset_index()
                    stats_classes.columns = ['Classe_Gramatical', 'Quantidade']
                    stats_classes.to_excel(writer, sheet_name='Estatísticas', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label="📊 Baixar Dados Sintáticos",
                    data=output_buffer,
                    file_name="analise_sintatica.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_sintatica_excel"
                )
            
            with col2:
                if nuvens_sintaticas:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for nome, nuvem in nuvens_sintaticas.items():
                            zip_file.writestr(f"nuvem_{nome}.png", nuvem.getvalue())
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="🖼️ Baixar Nuvens Sintáticas",
                        data=zip_buffer,
                        file_name="nuvens_sintaticas.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_sintatica_zip"
                    )
        else:
            st.info("Selecione 'Análise Sintática' ou 'Análise Completa' para ver esta análise.")

    # ========================
    # 📥 DOWNLOAD COMPLETO
    # ========================
    st.markdown("---")
    st.subheader("📦 Download Completo")
    
    # Criar arquivo ZIP com tudo
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Adicionar Excel com múltiplas planilhas
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_analisado.to_excel(writer, sheet_name='Dados_Originais', index=False)
            
            if "SENTIMENTO" in df_analisado.columns:
                df_sentimento = df_analisado[[text_col, 'SENTIMENTO', 'SCORE_SENTIMENTO']]
                df_sentimento.to_excel(writer, sheet_name='Análise_Sentimentos', index=False)
                stats_sentimentos = df_analisado['SENTIMENTO'].value_counts().reset_index()
                stats_sentimentos.columns = ['Sentimento', 'Quantidade']
                stats_sentimentos.to_excel(writer, sheet_name='Estatísticas_Sentimentos', index=False)
            
            if df_sintatica is not None:
                df_sintatica.to_excel(writer, sheet_name='Análise_Sintática', index=False)
                stats_classes = df_sintatica['classe_gramatical'].value_counts().reset_index()
                stats_classes.columns = ['Classe_Gramatical', 'Quantidade']
                stats_classes.to_excel(writer, sheet_name='Estatísticas_Sintáticas', index=False)
        
        excel_buffer.seek(0)
        zip_file.writestr("resultados_analise.xlsx", excel_buffer.getvalue())
        
        # Adicionar todas as nuvens
        for nome, nuvem in nuvens.items():
            zip_file.writestr(f"{nome}.png", nuvem.getvalue())
        
        # Adicionar nuvens sintáticas se existirem
        if df_sintatica is not None:
            nuvens_sintaticas = {}
            nuvens_sintaticas["verbos"] = gerar_nuvem_por_classe(df_sintatica, "VERB", "Verbos")
            nuvens_sintaticas["adjetivos"] = gerar_nuvem_por_classe(df_sintatica, "ADJ", "Adjetivos")
            nuvens_sintaticas["substantivos"] = gerar_nuvem_por_classe(df_sintatica, "NOUN", "Substantivos")
            
            for nome, nuvem in nuvens_sintaticas.items():
                if nuvem:
                    zip_file.writestr(f"nuvem_{nome}.png", nuvem.getvalue())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📦 Baixar ZIP Completo",
        data=zip_buffer,
        file_name="analise_texto_completa.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_completo"
    )

else:
    # Mensagem inicial
    st.markdown("""
    ## 🚀 Bem-vindo ao Analisador de Texto Completo!
    
    **Como usar:**
    1. 📂 **Upload**: Envie sua planilha na sidebar
    2. ⚙️ **Configuração**: Selecione coluna e tipo de análise
    3. 🔍 **Executar**: Clique no botão na sidebar
    4. 📊 **Resultados**: Visualize nas seções expansíveis abaixo
    5. 📥 **Download**: Baixe resultados individuais ou completos
    
    **Recursos disponíveis:**
    - ☁️ **Nuvens de palavras** personalizáveis
    - 🧠 **Análise de sentimentos** com RoBERTuito
    - 📝 **Análise sintática** com Stanza
    - 🔤 **Nuvens por classe gramatical** (verbos, adjetivos, substantivos)
    """)
    
    with st.expander("ℹ️ Mais informações"):
        st.markdown("""
        **Tipos de Análise:**
        - **Sentimento**: Classifica textos como Positivo/Negativo/Neutro
        - **Sintática**: Analisa estrutura gramatical + nuvens categorizadas
        - **Nuvem**: Visualização de palavras frequentes
        - **Completa**: Todas as análises combinadas
        
        **Formatos suportados:** CSV, Excel (XLSX)
        **Idioma:** Português
        """)

# ========================
# 🎨 ESTILOS CSS
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
