import pandas as pd
import streamlit as st
import numpy as np
from datetime import date, datetime
import plotly.express as px

st.set_page_config(page_title="Kpis per Agent",layout='wide',page_icon=":bar_chart:", initial_sidebar_state="expanded")
agent_df= st.session_state.df


# Convertir la colonne Date si nécessaire
agent_df['Date'] = pd.to_datetime(agent_df['Date']).dt.date  # supprime l'heure

# Dates min et max
min_date = agent_df['Date'].min()
max_date = agent_df['Date'].max()

# Widgets de sélection de dates
st.sidebar.write('Min Date:', min_date)
start_date = st.sidebar.date_input(
    'Start date', 
    min_value=min_date, 
    max_value=max_date, 
    value=max_date - pd.Timedelta(days=30), 
    format='DD-MM-YYYY'
)
end_date = st.sidebar.date_input(
    'End date', 
    min_value=min_date, 
    max_value=max_date, 
    value=max_date, 
    format='DD-MM-YYYY'
)

# --- Filtrer par dates ---
df_filtered = agent_df[(agent_df["Date"] >= start_date) & (agent_df["Date"] <= end_date)]

agent_search = st.sidebar.text_input(
    "Search for a name", value="", placeholder="Full or partial name"
)

# Sélecteur avec autocomplétion (liste déroulante)
all_agents = df_filtered["Name"].dropna().unique().tolist()
agent_selected = st.sidebar.selectbox(
    "Choose a specific name", ["Tous"] + sorted(all_agents)
)



# --- Filtrer par recherche texte ---
if agent_search:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(agent_search, case=False, na=False)]

# --- Filtrer par agent sélectionné ---
if agent_selected != "Tous":
    df_filtered = df_filtered[df_filtered["Name"] == agent_selected]
# --- Résumé par agent ---
def summary_agent(df):
    # Conserver 'Name' pour le groupby, sélectionner seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include='number').columns

    # Grouper par 'Name' et sommer uniquement les colonnes numériques
    agent_summary = df.groupby("Name")[numeric_cols].sum().reset_index()

    return agent_summary


agent_summary = summary_agent(df_filtered)

df_filtered['Hold Time%']=((df_filtered['Time on hold']/df_filtered['Connection time'])*100).round(2)
df_filtered['wrapup%']=((df_filtered['wrapup']/df_filtered['handling time'])*100).round(2)
df_filtered['HOLD CALL%']=((df_filtered['Calls put on hold']/df_filtered['Calls handled'])*100).round(2)
df_filtered['AHT']=((df_filtered['handling time']/df_filtered['Calls handled'])/60).round(2)
df_filtered['ACT']=((df_filtered['Duration of conversation']/df_filtered['Calls handled'])/60).round(2)
df_filtered['AUX%']=((df_filtered['Aux Time']/df_filtered['Connection time'])*100).round(2)
df_filtered['AVAIL°%']=((df_filtered['Time available']/df_filtered['Connection time'])*100).round(2)

# --- Afficher ---
st.title("👤 Performance by Agent")
st.dataframe(agent_summary,hide_index=True)
if agent_selected != "Tous":
    st.subheader(f'{agent_selected} ⁝⁝',divider=True)
    a,b,c,d = st.columns(4)
    a.metric("AHT", value=df_filtered['AHT'].mean().round(2),delta=df_filtered['AHT'].std().round(2))
    b.metric("ACT", value=df_filtered['ACT'].mean().round(2),delta=df_filtered['ACT'].std().round(2))
    c.metric("HOLD CALL%", value=df_filtered['HOLD CALL%'].mean().round(2),delta=df_filtered['HOLD CALL%'].std().round(2))
    d.metric("AUX%", value=df_filtered['AUX%'].mean().round(2),delta=df_filtered['AUX%'].std().round(2))
     
    
   # Création du pie chart 
    pie_cols = ['handling time', 'Time available','Time on hold', 'Aux Time', 'wrapup', 'Duration of conversation']
    pie_data = agent_summary[pie_cols].sum()
    fig_pie = px.pie(
    names=pie_cols,
    values=pie_data.values,
    title="Distribution of total time by type"
)

    

    # Affichage du graphique en barres : connexion par agent et par date
    if not df_filtered.empty:
        # On agrège par agent et par date pour la connexion
        bar_df = df_filtered.groupby(['Name', 'Date'])['Connection time'].sum().reset_index()
        fig_bar = px.bar(
            bar_df,
            x='Date',
            y=(bar_df['Connection time'] / 3600).round(2),  # Conversion en heures
            color='Name',
            title="Connection time by agent and date",
            labels={'Connection time': 'Connection time (secondes)', 'Date': 'Date', 'Name': 'Agent'},
            #barmode='group'
            text_auto= True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)

        # Courbe avec deux lignes : une pour AHT, une pour Calls handled
        # Ligne pour AHT, barres pour Calls handled
        fig_line = px.line(
            df_filtered,
            x='Date',
            y=['AHT','ACT'],
            #title="AHT et Calls handled par agent et par date",
            labels={'AHT': 'AHT', 'Date': 'Date'},
            markers=True,
            
        )
        fig_line.update_layout(legend=dict(orientation="h",
                                            yanchor='bottom', 
                                            y=-0.5, xanchor="center", x=0.5))

        fig_bar_calls = px.bar(
            df_filtered,
            x='Date',
            y='Calls handled',
            title="Calls handled par agent et par date",
            labels={'Calls handled': 'Calls handled', 'Date': 'Date'},
            opacity=0.5,
            text='Calls handled',
            color='Calls handled',
            color_continuous_scale=px.colors.sequential.Viridis,
         
        )
        # Superposer les deux graphiques
        for trace in fig_bar_calls.data:
            fig_line.add_trace(trace)
        fig_line.update_traces(mode='lines+markers', selector=dict(type='scatter'))
        fig_line.update_layout(legend_title_text='KPI')
        st.subheader("AHT & Calls handled ")
    
        st.plotly_chart(fig_line, use_container_width=True)

        
