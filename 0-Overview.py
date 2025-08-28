import pandas  as pd
import numpy as np
import streamlit as st
from datetime import date, datetime
import altair as alt
import plotly.express as px
import traceback
from openai import OpenAI
import re



st.set_page_config(page_title='Dashboard',layout="wide")
# Title
with open("styles.css", "r") as css_file:
    css_content = css_file.read()

# Injecter le CSS dans une balise <style> avec st.html
st.html(f"""
    <style>
    {css_content}
    </style>
    <div class='dashboard-container'>
        <h2> Call Center Dashboard</h2>
    </div>
""")
st.divider()
# Laod Data from excel file

#@st.cache_data
def read_excel_file():
    return pd.read_excel("data.xlsx")

# --- Chargement et stockage dans session_state ---
def load_data():
    if "df" not in st.session_state:
        st.session_state.df = read_excel_file()
    return st.session_state.df

# --- Charger le DataFrame et arrondir ---
data_filtre = load_data().round(0)
data_filtre['Date']=pd.to_datetime(data_filtre['Date'])
data_filtre['Mois']=data_filtre['Date'].dt.month_name()
data_filtre['Year']=data_filtre['Date'].dt.year
data_filtre['Date'] = data_filtre['Date'].dt.date

# sidebar 

min_date = data_filtre['Date'].min()
max_date = data_filtre['Date'].max()
st.sidebar.write('Min Date',min_date)
start_date =st.sidebar.date_input('Start date', min_value=min_date,max_value=max_date,value=max_date - pd.Timedelta(days=30),format='DD-MM-YYYY')
end_date =st.sidebar.date_input('End date', min_value=min_date,max_value=max_date,format='DD-MM-YYYY')

monthly_drop_colums = ['Name', 'Sexe','Date']
daily_drop_colums = ['Name', 'Sexe']

monthly_groupby_colums = ['Year','Mois','LOB']
daily_groupby_colums = ['Mois','Date','LOB']  

choix = st.sidebar.radio(
    'Group by ',
    [ 'Month','Date'])

if choix =='Month':
 dop_c = monthly_drop_colums
 groub_c = monthly_groupby_colums
 axe = 'Mois'

else :
    dop_c = daily_drop_colums
    groub_c = daily_groupby_colums
    axe = 'Date'

    #st.write(choix)

# Tableau 1 

st.subheader(f'Summary Data by line of business From {start_date} to {end_date} :',divider=True)

def summary_lob(data_filtre,start_date,end_date,drop_c,groub_c ):
    monthly_data = data_filtre[(data_filtre['Date'] >= start_date) & 
                               (data_filtre['Date'] <= end_date)]
    
    monthly_data = monthly_data.drop(columns=drop_c).groupby(groub_c).sum()
    
    monthly_data['Hold Time%']=((monthly_data['Time on hold']/monthly_data['Connection time'])*100).round(2)
    monthly_data['wrapup%']=((monthly_data['wrapup']/monthly_data['handling time'])*100).round(2)
    monthly_data['HOLD CALL%']=((monthly_data['Calls put on hold']/monthly_data['Calls handled'])*100).round(2)
    monthly_data['AHT']=((monthly_data['handling time']/monthly_data['Calls handled'])/60).round(2)
    monthly_data['ACT']=((monthly_data['Duration of conversation']/monthly_data['Calls handled'])/60).round(2)
    monthly_data['AUX%']=((monthly_data['Aux Time']/monthly_data['Connection time'])*100).round(2)
    monthly_data['AVAIL°%']=((monthly_data['Time available']/monthly_data['Connection time'])*100).round(2)
    
    monthly_data = monthly_data.reset_index()

    month_reference = ['January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December']

    if axe == 'Mois':
        monthly_data['Mois'] = pd.Categorical(monthly_data['Mois'], categories=month_reference, ordered=True)
        monthly_data = monthly_data.sort_values(['Year', 'Mois'])
        st.write(axe)
    else :
        st.write(axe)
        monthly_data = monthly_data.sort_values(['Date'])
    return monthly_data

data_monthly_filtre = summary_lob(data_filtre,start_date,end_date,dop_c,groub_c)



Sum_metric=['Connection time','handling time','Calls handled','Calls put on hold','Time available','Aux Time','Duration of conversation']
# Convertir toutes les colonnes entières en int Python




st.dataframe(data_monthly_filtre,hide_index=True)

st.divider()

def sel1():
    selection = st.pills("Select Metric", Sum_metric, default='Calls handled')
    return selection


# Multi color for custemers Graph 
def color_c():

    col1, col2, col3 = st.columns(3)
    with col1:
        C1 =st.color_picker('Commercial',"#12DEE6")
    with col2:
        C2 =st.color_picker('Technique Internet',"#1080C2")
    with col3:
        C3 =st.color_picker('Technique Mobil 4G',"#E2E2BA")
    
    return C1,C2,C3

C1, C2, C3 = color_c()


# Create graph1 
def grph1(data_monthly_filtre, selection, start_date, end_date,axe):
    fig1 = px.bar(
        data_monthly_filtre,
        x= axe, 
        y=selection,
        color='LOB',
        title=f'{selection} by Month and LOB from {start_date} to {end_date}',
        #labels={"Mois": "Month-Year", "Calls handled": "Calls Handled"},
        color_discrete_sequence=[C1, C2, C3],
        text=selection,
        #facet_col="Year",
        #category_orders={"Mois": month_order},
        barmode='group',
        
    )
    if axe == 'Date':
        fig1.update_xaxes(
            tickformat="%d-%m-%Y",
            dtick="D1"  # Show one tick per day
        )
    return fig1
      



calc_metric=['Hold Time%','wrapup%','HOLD CALL%','AHT','ACT','AUX%','AVAIL°%']

def sel2():
    selection2= st.pills("Select Calculate Metric", calc_metric,default='AHT')
    return  selection2 

# Create graph2
def grph2(data_monthly_filtre, selection2, start_date, end_date,axe):
        fig2 = px.line(
        data_monthly_filtre,
        x=axe, 
        y=selection2,
        color='LOB',
        title=f'{selection2} by Month and LOB from {start_date} to {end_date}',
        color_discrete_sequence=[C1, C2, C3],
        text=selection2,
        #animation_frame='Year',
        #facet_col="Year",
        
    )
    
    # Update traces to include mode and line properties
        fig2.update_traces(mode='lines+markers+text', line=dict(width=3))
        fig2.update_layout(
        width=800,
        height=600
    )
        if axe == 'Date':
            fig2.update_xaxes(
            tickformat="%d-%m-%Y",
            dtick="D1"  # Show one tick per day
        )
        return fig2
      

#st.plotly_chart(fig2)

tab1, tab2 = st.tabs(["Sum_Metric", "Calculat_Metric"],)
with tab1:
    selection = sel1()  
    fig1 = grph1(data_monthly_filtre, selection, start_date, end_date,axe)
    st.plotly_chart(fig1, use_container_width=True)

# Place fig2 in tab2
with tab2:
    selection2 = sel2()
    fig2 = grph2(data_monthly_filtre, selection2, start_date, end_date,axe)
    st.plotly_chart(fig2, use_container_width=True)

data_monthly_filtre=(data_monthly_filtre[data_monthly_filtre['LOB']=='Commercial'])


client = OpenAI(api_key="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")

# ==============================
# 2) Fonction utilitaire : Nettoyage du code IA
# ==============================
def clean_ai_code(ai_code: str) -> str:
    """
    Supprime les balises Markdown ```python ... ``` que l'IA pourrait ajouter,
    pour ne garder QUE le code exécutable.
    """
    if not ai_code:
        return ""  # Si pas de code généré
    
    ai_code = ai_code.strip()  # Supprime les espaces au début/fin

    # Vérifie si le code commence par des balises ``` (Markdown)
    if ai_code.startswith("```"):
        parts = ai_code.split("```")  # Coupe le texte par blocs
        ai_code = parts[1] if len(parts) > 1 else parts[0]

    # Supprime un éventuel "python" que l'IA ajoute parfois après ```
    ai_code = ai_code.removeprefix("python").strip()
    return ai_code


# ==============================
# 3) Fonction principale : Communication avec l'IA
# ==============================
def ask_ai(df, question):
    """
    - Prépare un échantillon du DataFrame (head 5)
    - Envoie la question + les données à l'IA
    - Récupère le code généré par l'IA
    - Nettoie et exécute ce code dans un environnement sécurisé
    - Retourne le code généré + l'environnement d'exécution
    """

    # --- a) Création d'un échantillon des données (head 5)
    df_preview = df.head(5).copy()

    # Conversion des colonnes texte/catégorielles en string (pour éviter les erreurs IA)
    for col in df_preview.select_dtypes(['category', 'object']).columns:
        df_preview[col] = df_preview[col].astype(str)

    # --- b) Message système = "personnalité" de l'IA
    # Ici on dicte les règles strictes à l'IA :
    # - Libs autorisées (pandas, numpy, plotly.express, altair, streamlit)
    # - Format d'affichage des graphes
    # - Pas de matplotlib/seaborn/import
    # - Ajouter un commentaire (st.write) pour analyser les résultats
    system_message = """
    Tu es un assistant d'analyse de données Streamlit.
    - Utilise uniquement pandas, numpy, plotly.express (px), altair (alt) et streamlit (st).
    - Pour les graphiques, crée un objet fig et affiche-le avec st.plotly_chart(fig) ou st.altair_chart(fig).
    - avant d'appliquer un filtre sur la colonne date, merci de convertir la date à utliser en format date exemple pour les data à la date du 18-03-2025 : df_filtered = df[df['Date'] == 2025-03-18].
    - Ne jamais utiliser matplotlib ou seaborn. Ne pas inclure d'import.
    - Retourne uniquement du code Python exécutable, sans texte explicatif ni balises ```
    - Si la question n'est pas claire, demande des précisions.
    - Si la question nécessite plusieurs graphiques, crée plusieurs objets fig (fig1, fig2, etc.) et affiche-les séparément.
    - il faut toujours ajouter un commentaire pour évaluer et décrire les resulats trouver avecdans un  st.write() , tu peux aussi donner ton avis ou meme proposer des actions correctif.
    """

    try:
        # --- c) Envoi de la requête à l'IA
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Voici le DataFrame (head 5) :\n{df_preview.to_string(index=False)}\n\nQuestion : {question}"}
            ],
            temperature=0  # = réponse déterministe (moins de créativité)
        )

        # --- d) Extraction et nettoyage du code
        ai_code = response.choices[0].message.content
        ai_code = clean_ai_code(ai_code)

        # --- e) Vérification des patterns interdits
        forbidden_patterns = [r"matplotlib", r"plt\.", r"seaborn", r"sns\.", r"import "]
        for pattern in forbidden_patterns:
            if re.search(pattern, ai_code, re.IGNORECASE):
                raise ValueError("⚠️ Code généré contient une librairie non autorisée.")

        # --- f) Exécution sécurisée du code
        # On définit un "mini-univers" (local_env) avec seulement les ressources autorisées
        local_env = {
            "df": df.copy(), "pd": pd, "np": np, "px": px, "st": st,
            "alt": alt, "datetime": datetime, "date": date
        }

        # Exécution du code généré par l’IA (⚠️ surveiller en prod)
        exec(ai_code, {}, local_env)

        return ai_code, local_env

    except Exception as e:
        # En cas d’erreur, on renvoie un message + le code IA fautif
        raise RuntimeError(f"❌ Erreur lors de l'exécution du code IA : {e}\nCode IA:\n{ai_code if 'ai_code' in locals() else ''}")


# ==============================
# 4) Interface Utilisateur (Streamlit)
# ==============================

df = data_filtre  # DataFrame principal (déjà défini avant)

# Zone de chat input
question = st.chat_input("Comment puis-je vous aider ? :")

if question:
    st.title("🤖 Agent IA pour Analyse du DataFrame")
    with st.spinner("L'agent IA réfléchit..."):
        try:
            # --- a) On interroge l'IA
            ai_code, results = ask_ai(df, question)

            # --- b) Affichage du code généré
            st.subheader("📝 Code généré par l'IA :")
            st.code(ai_code, language="python")

            # --- c) Affichage des résultats
            st.subheader("📊 Data Frame :")
            
            for key, val in results.items():
                st.dataframe(val)
        #         # Filtrage des variables internes inutiles
        #         if key.startswith("__") or key in ["df", "pd", "np"]:
        #             continue

        #         # Affichage adapté selon le type d'objet
        #         if isinstance(val, pd.DataFrame):
        #             st.dataframe(val)
        #         elif hasattr(val, "show"):
        #             val.show()
        #         elif "plotly" in str(type(val)):
        #             st.plotly_chart(val)
        #         elif "altair" in str(type(val)):
        #             st.altair_chart(val)
        #         else:
        #             st.write(f": {val}")

        except Exception as e:
            st.error(str(e))



