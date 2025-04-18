import streamlit as st
import pandas as pd
from typing import Union, List, Literal
from pydantic import BaseModel
from difflib import SequenceMatcher
import unicodedata
import re
import numpy as np

class ConversationAnalysis(BaseModel):
    """Modèle Pydantic pour l'analyse complète"""
    questions_utilisateur_clean: str
    type_demande: Literal["reclamation", "information", "assistance_technique", "suivi_demande"]
    theme_principal: str
    sous_theme: str
    sentiment: Literal["positif", "neutre", "negatif"]
    phase_vente: Literal["avant_vente", "apres_vente"]
    scope_status: Literal["in_scope", "out_of_scope"]
    demande_handover: bool
    is_resolved: bool
    raison_negative: Union[str, None]
    intention_out_of_scope: Union[str, None]
    theme_handover: Union[str, None]
    produits_avant_vente: Union[List[str], None]
    langue_detectee: Literal["fr", "en", "es", "de", "it", "pt", "nl", "autre"]
    topics_validation: List[str]

def is_similar(str1: str, str2: str, threshold: float = 0.8) -> bool:
    """Vérifie si deux chaînes sont similaires en utilisant SequenceMatcher"""
    # Check if inputs are strings, if not convert them to strings or handle NaN
    if not isinstance(str1, str):
        if pd.isna(str1):
            return False
        str1 = str(str1)
    
    if not isinstance(str2, str):
        if pd.isna(str2):
            return False
        str2 = str(str2)
    
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() > threshold

def normalize_text(text: str) -> str:
    """Normalise le texte en retirant les accents, la ponctuation et les espaces multiples"""
    # Check if text is a string, if not convert it to string or return empty string
    if not isinstance(text, str):
        if pd.isna(text):  # Handle NaN values
            return ""
        text = str(text)  # Convert other types to string
    
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def get_diverse_examples(questions: pd.Series, n: int = 3, similarity_threshold: float = 0.6) -> list:
    """Sélectionne n questions diverses en évitant les questions trop similaires"""
    # Filter out NaN values
    questions = questions.dropna()
    
    if len(questions) <= n:
        return questions.tolist()
    
    unique_questions = {}
    for q in questions:
        # Skip non-string values
        if not isinstance(q, str):
            continue
            
        normalized = normalize_text(q)
        if normalized not in unique_questions:
            unique_questions[normalized] = q
    
    questions_list = list(unique_questions.values())
    if len(questions_list) <= n:
        return questions_list
    
    questions_by_length = sorted(questions_list, key=len)
    total_questions = len(questions_by_length)
    
    third = total_questions // 3
    short_questions = questions_by_length[:third]
    medium_questions = questions_by_length[third:2*third]
    long_questions = questions_by_length[2*third:]
    
    diverse_examples = []
    groups = [short_questions, medium_questions, long_questions]
    
    for group in groups:
        if not group or len(diverse_examples) >= n:
            continue
            
        for candidate in group:
            candidate_norm = normalize_text(candidate)
            is_too_similar = any(
                is_similar(candidate_norm, normalize_text(example), similarity_threshold)
                for example in diverse_examples
            )
            
            if not is_too_similar:
                diverse_examples.append(candidate)
                break
    
    remaining_slots = n - len(diverse_examples)
    if remaining_slots > 0:
        remaining_questions = [q for q in questions_list if q not in diverse_examples]
        for q in remaining_questions:
            if len(diverse_examples) >= n:
                break
                
            q_norm = normalize_text(q)
            is_too_similar = any(
                is_similar(q_norm, normalize_text(example), similarity_threshold)
                for example in diverse_examples
            )
            
            if not is_too_similar:
                diverse_examples.append(q)
    
    return diverse_examples

def analyze_negative_sentiments(df: pd.DataFrame) -> dict:
    """Analyse les sentiments négatifs groupés par thème et sous-thème"""
    # Filter out rows with NaN in theme_principal or sous_theme
    negative_df = df[(df['sentiment'] == 'negatif') & 
                     df['theme_principal'].notna() & 
                     df['sous_theme'].notna()].copy()
    
    def unique_list(x):
        return list(dict.fromkeys(x))
    
    grouped = negative_df.groupby(['theme_principal', 'sous_theme']).agg({
        'questions_utilisateur_clean': unique_list,
        'theme_principal': 'count'
    }).rename(columns={'theme_principal': 'count'})
    
    result = {}
    for (theme, subtheme), data in grouped.iterrows():
        if theme not in result:
            result[theme] = {}
        
        # Filter out NaN values from questions
        questions = [q for q in data['questions_utilisateur_clean'] if isinstance(q, str) and not pd.isna(q)]
        unique_questions = pd.Series(list(dict.fromkeys(questions)))
        
        result[theme][subtheme] = {
            'count': data['count'],
            'questions': get_diverse_examples(unique_questions)
        }
    
    return result

def analyze_out_of_scope(df: pd.DataFrame) -> dict:
    """Analyse les conversations out of scope groupées par thème et sous-thème"""
    out_scope_df = df[df['scope_status'] == 'out_of_scope'].copy()
    
    def unique_list(x):
        return list(dict.fromkeys(x))
    
    grouped = out_scope_df.groupby(['theme_principal', 'sous_theme']).agg({
        'questions_utilisateur_clean': unique_list,
        'theme_principal': 'count'
    }).rename(columns={'theme_principal': 'count'})
    
    result = {}
    for (theme, subtheme), data in grouped.iterrows():
        if theme not in result:
            result[theme] = {}
        unique_questions = pd.Series(list(dict.fromkeys(data['questions_utilisateur_clean'])))
        result[theme][subtheme] = {
            'count': data['count'],
            'questions': get_diverse_examples(unique_questions)
        }
    
    return result

def analyze_handover(df: pd.DataFrame) -> dict:
    """Analyse les conversations avec demande de handover groupées par thème et sous-thème"""
    handover_df = df[df['demande_handover'] == True].copy()
    
    def unique_list(x):
        return list(dict.fromkeys(x))
    
    grouped = handover_df.groupby(['theme_principal', 'sous_theme']).agg({
        'questions_utilisateur_clean': unique_list,
        'theme_principal': 'count'
    }).rename(columns={'theme_principal': 'count'})
    
    result = {}
    for (theme, subtheme), data in grouped.iterrows():
        if theme not in result:
            result[theme] = {}
        unique_questions = pd.Series(list(dict.fromkeys(data['questions_utilisateur_clean'])))
        result[theme][subtheme] = {
            'count': data['count'],
            'questions': get_diverse_examples(unique_questions)
        }
    
    return result

def analyze_unresolved(df: pd.DataFrame) -> dict:
    """Analyse les conversations non résolues groupées par thème et sous-thème"""
    unresolved_df = df[df['is_resolved'] == False].copy()
    
    def unique_list(x):
        return list(dict.fromkeys(x))
    
    grouped = unresolved_df.groupby(['theme_principal', 'sous_theme']).agg({
        'questions_utilisateur_clean': unique_list,
        'theme_principal': 'count'
    }).rename(columns={'theme_principal': 'count'})
    
    result = {}
    for (theme, subtheme), data in grouped.iterrows():
        if theme not in result:
            result[theme] = {}
        unique_questions = pd.Series(list(dict.fromkeys(data['questions_utilisateur_clean'])))
        result[theme][subtheme] = {
            'count': data['count'],
            'questions': get_diverse_examples(unique_questions)
        }
    
    return result

def display_date_stats(df: pd.DataFrame):
    """Display statistics about the date range in the dataset"""
    st.subheader("📅 Période couverte")
    
    df['date'] = pd.to_datetime(df['date'])
    
    start_date = df['date'].min()
    end_date = df['date'].max()
    total_days = (end_date - start_date).days
    
    unique_months = df['date'].dt.to_period('M').nunique() if total_days >= 30 else 0
    unique_weeks = df['date'].dt.to_period('W').nunique() if total_days >= 7 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de mois", unique_months)
    col2.metric("Nombre de semaines", unique_weeks)
    col3.metric("Nombre de jours", total_days + 1)
    
    st.write(f"Du {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}")

def calculate_trend_changes(df: pd.DataFrame, period_type: str) -> dict:
    df['date'] = pd.to_datetime(df['date'])
    freq = 'W' if period_type == 'weekly' else ('Q' if period_type == 'quarterly' else 'M')
    
    full_period = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq=freq)
    all_themes = df['theme_principal'].unique()
    
    empty_df = pd.DataFrame(
        0,
        index=pd.PeriodIndex(full_period.to_period(freq)),
        columns=all_themes
    )
    
    df['period'] = df['date'].dt.to_period(freq)
    period_groups = df.groupby(['period', 'theme_principal']).size().unstack(fill_value=0)
    
    period_groups = empty_df.add(period_groups, fill_value=0)
    
    overall_changes = {}
    first_period = period_groups.iloc[0]
    last_period = period_groups.iloc[-1]
    
    for theme in period_groups.columns:
        initial_count = first_period[theme]
        final_count = last_period[theme]
        values = period_groups[theme].tolist()
        
        if initial_count == 0:
            pct_change = float('inf') if final_count > 0 else 0
        else:
            pct_change = (final_count - initial_count) / initial_count
            
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        trend_direction = 'croissant' if increases > decreases else 'décroissant'
        
        overall_changes[theme] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'pct_change': pct_change,
            'evolution': values,
            'trend_direction': trend_direction,
            'max_value': max(values),
            'min_value': min(values)
        }
    
    sorted_changes = sorted(
        overall_changes.items(),
        key=lambda x: (
            x[1]['initial_count'] == 0,
            -abs(x[1]['pct_change']) if x[1]['pct_change'] != float('inf') else 0
        )
    )
    
    overall_positive = []
    overall_negative = []
    new_themes = []
    
    for theme, data in sorted_changes:
        if data['initial_count'] == 0 and data['final_count'] > 0:
            new_themes.append((theme, data))
        elif data['pct_change'] > 0:
            overall_positive.append((theme, data))
        elif data['pct_change'] < 0:
            overall_negative.append((theme, data))
    
    return {
        'overall_positive': overall_positive,
        'overall_negative': overall_negative,
        'new_themes': new_themes,
        'periods': period_groups.index.strftime('%Y-%m-%d').tolist()
    }

def is_analysis_complete(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check if the DataFrame has all required analysis columns with valid values"""
    required_columns = {
        'questions_utilisateur_clean': str,
        'type_demande': ['reclamation', 'information', 'assistance_technique', 'suivi_demande'],
        'theme_principal': str,
        'sous_theme': str,
        'sentiment': ['positif', 'neutre', 'negatif'],
        'phase_vente': ['avant_vente', 'apres_vente'],
        'scope_status': ['in_scope', 'out_of_scope'],
        'demande_handover': bool,
        'is_resolved': bool,
        'langue_detectee': ['fr', 'en', 'es', 'de', 'it', 'pt', 'nl', 'autre']
    }
    
    missing_columns = []
    
    for col, validation in required_columns.items():
        if col not in df.columns:
            missing_columns.append(col)
            continue
            
        if isinstance(validation, list):
            if not df[col].dropna().isin(validation).all():
                missing_columns.append(f"{col} (invalid values)")
                
    return len(missing_columns) == 0, missing_columns

def calculate_impact_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les métriques d'impact et le sonaar score pour chaque sous-thème"""
    # Calculer les totaux pour chaque catégorie problématique
    total_unresolved = len(df[~df['is_resolved']])
    total_negative = len(df[df['sentiment'] == 'negatif'])
    total_handover = len(df[df['demande_handover']])
    total_outscope = len(df[df['scope_status'] == 'out_of_scope'])
    
    # Grouper par thème et sous-thème
    grouped = df.groupby(['theme_principal', 'sous_theme']).agg({
        'is_resolved': lambda x: sum(~x),
        'sentiment': lambda x: sum(x == 'negatif'),
        'demande_handover': sum,
        'scope_status': lambda x: sum(x == 'out_of_scope'),
        'questions_utilisateur_clean': list  # Collecter toutes les questions
    }).reset_index()
    
    # Calculer les impacts en pourcentage
    grouped['impact_resolution'] = (grouped['is_resolved'] / total_unresolved * 100).round(1)
    grouped['impact_sentiment'] = (grouped['sentiment'] / total_negative * 100).round(1)
    grouped['impact_handover'] = (grouped['demande_handover'] / total_handover * 100).round(1)
    grouped['impact_scope'] = (grouped['scope_status'] / total_outscope * 100).round(1)
    
    # Calculer le sonaar score
    grouped['sonaar_score'] = (
        0.4 * grouped['impact_resolution'] +
        0.3 * grouped['impact_sentiment'] +
        0.2 * grouped['impact_handover'] +
        0.1 * grouped['impact_scope']
    ).round(1)
    
    # Ajouter une colonne pour les catégories
    def get_categories(row):
        categories = []
        if row['is_resolved'] > 0:
            categories.append(f"Non résolu ({row['is_resolved']})")
        if row['sentiment'] > 0:
            categories.append(f"Sentiment négatif ({row['sentiment']})")
        if row['demande_handover'] > 0:
            categories.append(f"Handover ({row['demande_handover']})")
        if row['scope_status'] > 0:
            categories.append(f"Out of scope ({row['scope_status']})")
        return ', '.join(categories)
    
    # Ajouter les verbatims
    def get_verbatims(questions):
        return "\n".join(get_diverse_examples(pd.Series(questions), n=3))
    
    grouped['categories'] = grouped.apply(get_categories, axis=1)
    grouped['verbatims'] = grouped['questions_utilisateur_clean'].apply(get_verbatims)
    
    # Trier par sonaar score décroissant et prendre le top 10
    return grouped.sort_values('sonaar_score', ascending=False).head(10)

def display_statistics(df: pd.DataFrame, period_comparison: str = None, period_type: str = None, period_label: str = None):
    """Affiche les statistiques demandées sous forme de colonnes"""
    # Create a copy of the DataFrame to avoid warnings
    df = df.copy()
    
    st.subheader("📊 Statistiques clés")
    
    # Analyse des feedbacks
    st.markdown("### Feedbacks")
    
    # Calculer les moyennes de feedback seulement si les colonnes existent
    if 'feedbackPositive' in df.columns and 'feedbackNegative' in df.columns:
        avg_positive = df['feedbackPositive'].mean()
        avg_negative = df['feedbackNegative'].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Moyenne Feedback Positif", f"{avg_positive:.2f}")
        col2.metric("Moyenne Feedback Négatif", f"{avg_negative:.2f}")
    else:
        st.info("Aucune donnée de feedback disponible dans le fichier.")

    # Répartition des langues
    if 'langue_detectee' in df.columns:
        st.markdown("### Langues")
        lang_dist = df['langue_detectee'].value_counts()
        lang_pct = df['langue_detectee'].value_counts(normalize=True).round(3) * 100
        for lang, count in lang_dist.items():
            st.write(f"- {lang}: {count} ({lang_pct[lang]:.1f}%)")
            
            # Top 5 sous-thèmes pour cette langue
            st.markdown(f"#### Top 5 sous-thèmes pour {lang}")
            lang_df = df[df['langue_detectee'] == lang]
            subtheme_counts = lang_df.groupby(['sous_theme', 'theme_principal']).size().reset_index(name='count')
            top_5_subthemes = subtheme_counts.nlargest(5, 'count')
            
            # Calculer les pourcentages
            total_lang_convs = len(lang_df)
            top_5_subthemes['percentage'] = (top_5_subthemes['count'] / total_lang_convs * 100).round(1)
            
            # Préparer les données pour l'affichage
            display_df = top_5_subthemes.rename(columns={
                'sous_theme': 'Sous-thème',
                'theme_principal': 'Thème principal',
                'count': 'Nombre',
                'percentage': 'Pourcentage'
            })[['Sous-thème', 'Thème principal', 'Nombre', 'Pourcentage']].assign(
                Pourcentage=lambda x: x['Pourcentage'].apply(lambda v: f"{v}%")
            )
            
            # Convertir en dictionnaire pour l'affichage
            st.dataframe(
                display_df,
                hide_index=True,
                column_config={
                    "Sous-thème": st.column_config.TextColumn("Sous-thème"),
                    "Thème principal": st.column_config.TextColumn("Thème principal"),
                    "Nombre": st.column_config.NumberColumn("Nombre"),
                    "Pourcentage": st.column_config.TextColumn("Pourcentage")
                }
            )

    # Répartition des thèmes et sous-thèmes
    st.markdown("### Thèmes et sous-thèmes")
    
    # Filtrer les lignes avec des valeurs NaN dans theme_principal ou sous_theme
    valid_themes_df = df.dropna(subset=['theme_principal', 'sous_theme'])
    
    if len(valid_themes_df) < len(df):
        st.warning(f"Attention: {len(df) - len(valid_themes_df)} lignes ont été ignorées car elles contiennent des valeurs manquantes dans les colonnes 'theme_principal' ou 'sous_theme'.")
    
    # Grouper les données avec les questions
    theme_data = valid_themes_df.groupby(['theme_principal', 'sous_theme'])
    
    # Créer un DataFrame pour les counts et pourcentages
    theme_subtheme = theme_data.size().reset_index(name='count')
    theme_subtheme['percentage'] = (theme_subtheme['count'] / len(valid_themes_df) * 100).round(1)
    
    # Calculer les totaux par thème et trier
    theme_totals = theme_subtheme.groupby('theme_principal')['count'].sum().sort_values(ascending=False)
    
    # Créer un tableau avec les thèmes et sous-thèmes
    table_data = []
    
    # Parcourir les thèmes triés
    for theme in theme_totals.index:
        # Filtrer et trier les sous-thèmes pour ce thème
        theme_rows = theme_subtheme[theme_subtheme['theme_principal'] == theme].sort_values('count', ascending=False)
        theme_total = theme_totals[theme]
        theme_percentage = (theme_total / len(valid_themes_df) * 100).round(1)
        
        # Ajouter chaque sous-thème
        first_row = True
        for _, row in theme_rows.iterrows():
            # Récupérer les questions pour ce sous-thème
            theme_df = valid_themes_df[(valid_themes_df['theme_principal'] == theme) & 
                         (valid_themes_df['sous_theme'] == row['sous_theme'])]
            
            # Vérifier si la colonne questions_utilisateur_clean existe
            if 'questions_utilisateur_clean' in theme_df.columns:
                example_questions = get_diverse_examples(theme_df['questions_utilisateur_clean'])
                verbatims = "\n".join(example_questions) if example_questions else ""
            else:
                verbatims = ""
            
            table_data.append([
                f"{theme} ({theme_percentage}%)" if first_row else "",
                f"{row['sous_theme']} ({row['percentage']}%)",
                f"{row['count']}",
                verbatims
            ])
            first_row = False
    
    # Créer et afficher le tableau
    theme_table_df = pd.DataFrame(
        table_data,
        columns=["Thématique", "Sous-catégorie", "Nombre de conversations", "Verbatims"]
    )
    st.dataframe(
        theme_table_df,
        hide_index=True,
        column_config={
            "Thématique": st.column_config.TextColumn("Thématique"),
            "Sous-catégorie": st.column_config.TextColumn("Sous-catégorie"),
            "Nombre de conversations": st.column_config.NumberColumn("Nombre de conversations"),
            "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
        }
    )

    # Répartition des types de demandes
    st.markdown("### Types de demandes")
    type_dist = df['type_demande'].value_counts()
    type_pct = df['type_demande'].value_counts(normalize=True).round(3) * 100
    for type_dem, count in type_dist.items():
        st.write(f"- {type_dem}: {count} ({type_pct[type_dem]:.1f}%)")

    # Répartition des sentiments avec analyse détaillée
    st.markdown("### Sentiments")
    sentiment_dist = df['sentiment'].value_counts()
    sentiment_pct = df['sentiment'].value_counts(normalize=True).round(3) * 100
    for sentiment, count in sentiment_dist.items():
        st.write(f"- {sentiment}: {count} ({sentiment_pct[sentiment]:.1f}%)")
    
    # Analyse des sentiments négatifs
    if 'negatif' in sentiment_dist.index:
        st.markdown("#### Facteurs de Sentiment Négatif")
        negative_reasons = analyze_negative_sentiments(df)
        
        # Calculer le total des cas négatifs
        total_negative = sum(data['count'] for theme_data in negative_reasons.values() for data in theme_data.values())
        
        # Calculer les totaux par thème et trier les thèmes
        theme_totals = {
            theme: sum(data['count'] for data in theme_data.values())
            for theme, theme_data in negative_reasons.items()
        }
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Créer les données triées pour le tableau
        negative_table_data = []
        for theme, theme_total in sorted_themes:
            theme_percentage = round((theme_total / total_negative * 100), 1)
            
            # Trier les sous-thèmes par nombre d'occurrences
            subthemes_data = [
                (subtheme, data)
                for subtheme, data in negative_reasons[theme].items()
            ]
            sorted_subthemes = sorted(subthemes_data, key=lambda x: x[1]['count'], reverse=True)
            
            # Ajouter la première sous-catégorie avec le thème principal
            first_subtheme, first_data = sorted_subthemes[0]
            subtheme_percentage = round((first_data['count'] / theme_total * 100), 1)
            example_questions = get_diverse_examples(pd.Series(first_data['questions']), n=3)
            verbatims = "\n".join(example_questions) if example_questions else ""
            negative_table_data.append([
                f"{theme} ({theme_percentage}%)",
                f"{first_subtheme} ({subtheme_percentage}%)",
                f"{first_data['count']}",
                verbatims
            ])
            
            # Ajouter les autres sous-catégories triées
            for subtheme, data in sorted_subthemes[1:]:
                subtheme_percentage = round((data['count'] / theme_total * 100), 1)
                example_questions = get_diverse_examples(pd.Series(data['questions']), n=3)
                verbatims = "\n".join(example_questions) if example_questions else ""
                negative_table_data.append([
                    "",
                    f"{subtheme} ({subtheme_percentage}%)",
                    f"{data['count']}",
                    verbatims
                ])
        
        # Créer et afficher le tableau
        negative_table_df = pd.DataFrame(
            negative_table_data,
            columns=["Thématique", "Sous-catégorie", "Nombre de conversations", "Verbatims"]
        )
        st.dataframe(
            negative_table_df,
            hide_index=True,
            column_config={
                "Thématique": st.column_config.TextColumn("Thématique"),
                "Sous-catégorie": st.column_config.TextColumn("Sous-catégorie"),
                "Nombre de conversations": st.column_config.NumberColumn("Nombre de conversations"),
                "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
            }
        )

    # Répartition avant/après vente
    st.markdown("### Phase de vente")
    phase_dist = df['phase_vente'].value_counts()
    phase_pct = df['phase_vente'].value_counts(normalize=True).round(3) * 100
    for phase, count in phase_dist.items():
        st.write(f"- {phase}: {count} ({phase_pct[phase]:.1f}%)")

    # Répartition scope avec analyse détaillée
    st.markdown("### Scope")
    scope_dist = df['scope_status'].value_counts()
    scope_pct = df['scope_status'].value_counts(normalize=True).round(3) * 100
    for scope, count in scope_dist.items():
        st.write(f"- {scope}: {count} ({scope_pct[scope]:.1f}%)")
    
    # Analyse des out of scope
    if 'out_of_scope' in scope_dist.index:
        st.markdown("#### Analyse des conversations hors scope")
        out_scope_reasons = analyze_out_of_scope(df)
        
        # Calculer le total des cas out of scope
        total_outscope = sum(data['count'] for theme_data in out_scope_reasons.values() for data in theme_data.values())
        
        # Calculer les totaux par thème et trier les thèmes
        theme_totals = {
            theme: sum(data['count'] for data in theme_data.values())
            for theme, theme_data in out_scope_reasons.items()
        }
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Créer les données triées pour le tableau
        outscope_table_data = []
        for theme, theme_total in sorted_themes:
            theme_percentage = round((theme_total / total_outscope * 100), 1)
            
            # Trier les sous-thèmes par nombre d'occurrences
            subthemes_data = [
                (subtheme, data)
                for subtheme, data in out_scope_reasons[theme].items()
            ]
            sorted_subthemes = sorted(subthemes_data, key=lambda x: x[1]['count'], reverse=True)
            
            # Ajouter la première sous-catégorie avec le thème principal
            first_subtheme, first_data = sorted_subthemes[0]
            subtheme_percentage = round((first_data['count'] / theme_total * 100), 1)
            example_questions = get_diverse_examples(pd.Series(first_data['questions']), n=3)
            verbatims = "\n".join(example_questions) if example_questions else ""
            outscope_table_data.append([
                f"{theme} ({theme_percentage}%)",
                f"{first_subtheme} ({subtheme_percentage}%)",
                f"{first_data['count']}",
                verbatims
            ])
            
            # Ajouter les autres sous-catégories triées
            for subtheme, data in sorted_subthemes[1:]:
                subtheme_percentage = round((data['count'] / theme_total * 100), 1)
                example_questions = get_diverse_examples(pd.Series(data['questions']), n=3)
                verbatims = "\n".join(example_questions) if example_questions else ""
                outscope_table_data.append([
                    "",
                    f"{subtheme} ({subtheme_percentage}%)",
                    f"{data['count']}",
                    verbatims
                ])
        
        # Créer et afficher le tableau
        outscope_table_df = pd.DataFrame(
            outscope_table_data,
            columns=["Thématique", "Sous-catégorie", "Nombre de conversations", "Verbatims"]
        )
        st.dataframe(
            outscope_table_df,
            hide_index=True,
            column_config={
                "Thématique": st.column_config.TextColumn("Thématique"),
                "Sous-catégorie": st.column_config.TextColumn("Sous-catégorie"),
                "Nombre de conversations": st.column_config.NumberColumn("Nombre de conversations"),
                "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
            }
        )

    # Répartition handover avec analyse détaillée
    st.markdown("### Handover")
    handover_dist = df['demande_handover'].value_counts()
    handover_pct = df['demande_handover'].value_counts(normalize=True).round(3) * 100
    for handover, count in handover_dist.items():
        st.write(f"- {'Oui' if handover else 'Non'}: {count} ({handover_pct[handover]:.1f}%)")
    
    # Analyse des demandes de handover
    if True in handover_dist.index:
        st.markdown("#### Analyse des demandes de handover")
        handover_reasons = analyze_handover(df)
        
        # Calculer le total des cas handover
        total_handover = sum(data['count'] for theme_data in handover_reasons.values() for data in theme_data.values())
        
        # Calculer les totaux par thème et trier les thèmes
        theme_totals = {
            theme: sum(data['count'] for data in theme_data.values())
            for theme, theme_data in handover_reasons.items()
        }
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Créer les données triées pour le tableau
        handover_table_data = []
        for theme, theme_total in sorted_themes:
            theme_percentage = round((theme_total / total_handover * 100), 1)
            
            # Trier les sous-thèmes par nombre d'occurrences
            subthemes_data = [
                (subtheme, data)
                for subtheme, data in handover_reasons[theme].items()
            ]
            sorted_subthemes = sorted(subthemes_data, key=lambda x: x[1]['count'], reverse=True)
            
            # Ajouter la première sous-catégorie avec le thème principal
            first_subtheme, first_data = sorted_subthemes[0]
            subtheme_percentage = round((first_data['count'] / theme_total * 100), 1)
            example_questions = get_diverse_examples(pd.Series(first_data['questions']), n=3)
            verbatims = "\n".join(example_questions) if example_questions else ""
            handover_table_data.append([
                f"{theme} ({theme_percentage}%)",
                f"{first_subtheme} ({subtheme_percentage}%)",
                f"{first_data['count']}",
                verbatims
            ])
            
            # Ajouter les autres sous-catégories triées
            for subtheme, data in sorted_subthemes[1:]:
                subtheme_percentage = round((data['count'] / theme_total * 100), 1)
                example_questions = get_diverse_examples(pd.Series(data['questions']), n=3)
                verbatims = "\n".join(example_questions) if example_questions else ""
                handover_table_data.append([
                    "",
                    f"{subtheme} ({subtheme_percentage}%)",
                    f"{data['count']}",
                    verbatims
                ])
        
        # Créer et afficher le tableau
        handover_table_df = pd.DataFrame(
            handover_table_data,
            columns=["Thématique", "Sous-catégorie", "Nombre de conversations", "Verbatims"]
        )
        st.dataframe(
            handover_table_df,
            hide_index=True,
            column_config={
                "Thématique": st.column_config.TextColumn("Thématique"),
                "Sous-catégorie": st.column_config.TextColumn("Sous-catégorie"),
                "Nombre de conversations": st.column_config.NumberColumn("Nombre de conversations"),
                "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
            }
        )

    # Affichage du taux global de conversations after hours
    st.markdown("### After Hours")
    if 'after_hours' in df.columns:
        after_hours_total = df['after_hours'].value_counts()
        if True in after_hours_total.index:
            total_convs = len(df)
            after_hours_pct = (after_hours_total[True] / total_convs * 100)
            st.write(f"- Conversations en after hours: {after_hours_pct:.1f}% ({after_hours_total[True]} sur {total_convs} conversations)")
    else:
        st.info("Aucune donnée after hours disponible dans le fichier.")

    # Ajout des statistiques de résolution
    st.markdown("### Résolution")
    resolution_dist = df['is_resolved'].value_counts()
    resolution_pct = df['is_resolved'].value_counts(normalize=True).round(3) * 100
    total_resolved = resolution_dist.get(True, 0)
    for resolved, count in resolution_dist.items():
        st.write(f"- {'Résolu' if resolved else 'Non résolu'}: {count} ({resolution_pct[resolved]:.1f}%)")
    
    # Nouvelles métriques de résolution
    st.markdown("#### Métriques détaillées de résolution")
    
    try:
        # Taux de résolution en after_hours
        if 'after_hours' in df.columns:
            after_hours_resolved = df[df['is_resolved'] == True]['after_hours'].value_counts()
            if True in after_hours_resolved.index:
                after_hours_rate = float(after_hours_resolved[True]) / float(total_resolved) * 100
                st.write(f"- Conversations résolues en after hours: {after_hours_rate:.1f}% ({after_hours_resolved[True]} sur {total_resolved} conversations résolues)")
        
        # Nombre d'échanges moyen des conversations résolues
        resolved_convs = df[df['is_resolved'] == True].copy()
        if not resolved_convs.empty and 'turn_count' in resolved_convs.columns:
            # Convertir et nettoyer les données
            resolved_convs['turn_count'] = pd.to_numeric(resolved_convs['turn_count'], errors='coerce')
            
            # Filtrer les valeurs non-null avant de calculer les moyennes
            valid_turns = resolved_convs['turn_count'].dropna()
            
            if not valid_turns.empty:
                avg_turns = float(valid_turns.mean())
                st.write(f"- Nombre moyen d'échanges des conversations résolues: {avg_turns:.1f} tours")
        
        # Taux de résolution en un seul tour
        if 'turn_count' in df.columns:
            valid_turns = pd.to_numeric(df['turn_count'], errors='coerce')
            one_turn_resolved = df[(valid_turns == 1) & (df['is_resolved'] == True)]
            if not one_turn_resolved.empty:
                one_turn_rate = float(len(one_turn_resolved)) / float(total_resolved) * 100
                st.write(f"- Conversations résolues en un seul tour: {one_turn_rate:.1f}% ({len(one_turn_resolved)} sur {total_resolved} conversations résolues)")
    
    except Exception as e:
        st.warning(f"Erreur lors du calcul des métriques détaillées : {str(e)}")
        st.write("Certaines métriques n'ont pas pu être calculées en raison d'erreurs dans les données.")

    # Analyse croisée résolution/thèmes
    st.markdown("#### Taux de résolution par thème")
    theme_resolution = df.groupby('theme_principal')['is_resolved'].agg(['count', 'sum'])
    theme_resolution['resolution_rate'] = (theme_resolution['sum'] / theme_resolution['count'] * 100).round(1)
    theme_resolution = theme_resolution.sort_values('resolution_rate', ascending=False)
    
    for theme in theme_resolution.index:
        count = theme_resolution.loc[theme, 'count']
        resolved = theme_resolution.loc[theme, 'sum']
        rate = theme_resolution.loc[theme, 'resolution_rate']
        st.write(f"- {theme}: {rate:.1f}% ({resolved}/{count} conversations)")
    
    # Analyse croisée résolution/type de demande
    st.markdown("#### Taux de résolution par type de demande")
    type_resolution = df.groupby('type_demande')['is_resolved'].agg(['count', 'sum'])
    type_resolution['resolution_rate'] = (type_resolution['sum'] / type_resolution['count'] * 100).round(1)
    type_resolution = type_resolution.sort_values('resolution_rate', ascending=False)
    
    for type_dem in type_resolution.index:
        count = type_resolution.loc[type_dem, 'count']
        resolved = type_resolution.loc[type_dem, 'sum']
        rate = type_resolution.loc[type_dem, 'resolution_rate']
        st.write(f"- {type_dem}: {rate:.1f}% ({resolved}/{count} conversations)")
    
    # Analyse croisée résolution/sentiment
    st.markdown("#### Taux de résolution par sentiment")
    sentiment_resolution = df.groupby('sentiment')['is_resolved'].agg(['count', 'sum'])
    sentiment_resolution['resolution_rate'] = (sentiment_resolution['sum'] / sentiment_resolution['count'] * 100).round(1)
    sentiment_resolution = sentiment_resolution.sort_values('resolution_rate', ascending=False)
    
    for sentiment in sentiment_resolution.index:
        count = sentiment_resolution.loc[sentiment, 'count']
        resolved = sentiment_resolution.loc[sentiment, 'sum']
        rate = sentiment_resolution.loc[sentiment, 'resolution_rate']
        st.write(f"- {sentiment}: {rate:.1f}% ({resolved}/{count} conversations)")

    # Analyse croisée feedback/résolution
    st.markdown("#### Analyse des feedbacks par résolution")
    if 'feedbackPositive' in df.columns and 'feedbackNegative' in df.columns:
        feedback_resolution = df.groupby('is_resolved').agg({
            'feedbackPositive': 'mean',
            'feedbackNegative': 'mean'
        }).round(2)
        
        st.write("Moyenne des feedbacks par statut de résolution :")
        for status in feedback_resolution.index:
            status_label = "Résolu" if status else "Non résolu"
            pos_feedback = feedback_resolution.loc[status, 'feedbackPositive']
            neg_feedback = feedback_resolution.loc[status, 'feedbackNegative']
            st.write(f"- {status_label}: Positif {pos_feedback:.2f}, Négatif {neg_feedback:.2f}")
    else:
        st.info("Aucune donnée de feedback disponible pour l'analyse par résolution.")

    # Analyse des conversations non résolues
    if False in resolution_dist.index:
        st.markdown("#### Analyse des conversations non résolues")
        unresolved_reasons = analyze_unresolved(df)
        
        # Calculer le total des cas non résolus
        total_unresolved = sum(data['count'] for theme_data in unresolved_reasons.values() for data in theme_data.values())
        
        # Calculer les totaux par thème et trier les thèmes
        theme_totals = {
            theme: sum(data['count'] for data in theme_data.values())
            for theme, theme_data in unresolved_reasons.items()
        }
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Créer les données triées pour le tableau
        unresolved_table_data = []
        for theme, theme_total in sorted_themes:
            theme_percentage = round((theme_total / total_unresolved * 100), 1)
            
            # Trier les sous-thèmes par nombre d'occurrences
            subthemes_data = [
                (subtheme, data)
                for subtheme, data in unresolved_reasons[theme].items()
            ]
            sorted_subthemes = sorted(subthemes_data, key=lambda x: x[1]['count'], reverse=True)
            
            # Ajouter la première sous-catégorie avec le thème principal
            first_subtheme, first_data = sorted_subthemes[0]
            subtheme_percentage = round((first_data['count'] / theme_total * 100), 1)
            example_questions = get_diverse_examples(pd.Series(first_data['questions']), n=3)
            verbatims = "\n".join(example_questions) if example_questions else ""
            unresolved_table_data.append([
                f"{theme} ({theme_percentage}%)",
                f"{first_subtheme} ({subtheme_percentage}%)",
                f"{first_data['count']}",
                verbatims
            ])
            
            # Ajouter les autres sous-catégories triées
            for subtheme, data in sorted_subthemes[1:]:
                subtheme_percentage = round((data['count'] / theme_total * 100), 1)
                example_questions = get_diverse_examples(pd.Series(data['questions']), n=3)
                verbatims = "\n".join(example_questions) if example_questions else ""
                unresolved_table_data.append([
                    "",
                    f"{subtheme} ({subtheme_percentage}%)",
                    f"{data['count']}",
                    verbatims
                ])
        
        # Créer et afficher le tableau
        unresolved_table_df = pd.DataFrame(
            unresolved_table_data,
            columns=["Thématique", "Sous-catégorie", "Nombre de conversations", "Verbatims"]
        )
        st.dataframe(
            unresolved_table_df,
            hide_index=True,
            column_config={
                "Thématique": st.column_config.TextColumn("Thématique"),
                "Sous-catégorie": st.column_config.TextColumn("Sous-catégorie"),
                "Nombre de conversations": st.column_config.NumberColumn("Nombre de conversations"),
                "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
            }
        )

    # Ajouter la nouvelle section d'analyse d'impact après les statistiques existantes
    st.markdown("### 🎯 Analyse d'Impact et Sonaar Score")
    
    st.write("""
    Cette section présente une analyse des 10 sous-thèmes les plus problématiques basée sur quatre critères :
    - Impact sur la résolution (40% du score)
    - Impact sur le sentiment (30% du score)
    - Impact sur le handover (20% du score)
    - Impact sur le scope (10% du score)
    """)
    
    # Calculer et afficher les métriques d'impact
    impact_metrics = calculate_impact_metrics(df)
    
    # Créer un DataFrame pour l'affichage
    display_df = impact_metrics[['theme_principal', 'sous_theme', 'impact_resolution', 
                               'impact_sentiment', 'impact_handover', 'impact_scope',
                               'sonaar_score', 'categories', 'verbatims']]
    
    # Renommer les colonnes pour l'affichage
    display_df.columns = ['Thème', 'Sous-thème', 'Impact Résolution (%)', 
                         'Impact Sentiment (%)', 'Impact Handover (%)',
                         'Impact Hors Scope (%)', 'Sonaar Score', 'Catégories', 'Verbatims']
    
    # Afficher le tableau
    st.dataframe(
        display_df,
        hide_index=True,
        column_config={
            "Thème": st.column_config.TextColumn("Thème"),
            "Sous-thème": st.column_config.TextColumn("Sous-thème"),
            "Impact Résolution (%)": st.column_config.NumberColumn("Impact Résolution (%)", format="%.1f%%"),
            "Impact Sentiment (%)": st.column_config.NumberColumn("Impact Sentiment (%)", format="%.1f%%"),
            "Impact Handover (%)": st.column_config.NumberColumn("Impact Handover (%)", format="%.1f%%"),
            "Impact Hors Scope (%)": st.column_config.NumberColumn("Impact Hors Scope (%)", format="%.1f%%"),
            "Sonaar Score": st.column_config.NumberColumn("Sonaar Score", format="%.1f"),
            "Catégories": st.column_config.TextColumn("Catégories", width="large"),
            "Verbatims": st.column_config.TextColumn("Verbatims", width="large")
        }
    )
    
    # Option d'export
    st.markdown("#### 📥 Exporter l'analyse")
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Télécharger l'analyse d'impact (CSV)",
        data=csv,
        file_name="analyse_impact.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(page_title="Analyseur de Conversations", layout="wide")
    
    st.title("✨ sonaar insights")
    st.write("Téléchargez votre fichier CSV contenant les conversations pré-analysées.")
    
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier CSV
            df = pd.read_csv(uploaded_file)
            
            # Vérifier la présence des colonnes requises
            is_complete, missing_cols = is_analysis_complete(df)
            
            if not is_complete:
                st.error(f"Le fichier CSV ne contient pas toutes les colonnes requises ou contient des valeurs invalides. Colonnes manquantes ou invalides: {', '.join(missing_cols)}")
                return
            
            # Vérifier et convertir la colonne date
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
                except Exception as e:
                    try:
                        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
                    except Exception as e:
                        st.error(f"Erreur de format de date. Les dates doivent être au format ISO8601 ou YYYY-MM-DD. Erreur: {str(e)}")
                        return
                display_date_stats(df)
            
            # Aperçu des données
            st.subheader("📄 Aperçu des données")
            st.dataframe(df.head())
            
            # Configuration des paramètres d'analyse
            with st.expander("⚙️ Configuration de l'analyse", expanded=True):
                # Affichage du nombre total de lignes
                total_rows = len(df)
                st.info(f"Nombre total de conversations : {total_rows}")
                
                # Option de comparaison temporelle
                period_comparison = st.radio(
                    "Type de comparaison temporelle",
                    options=['Aucune', 'Hebdomadaire', 'Mensuelle', 'Trimestrielle'],
                    horizontal=True
                )
            
            # Vérification de la couverture temporelle
            if period_comparison != 'Aucune':
                min_date = df['date'].min()
                max_date = df['date'].max()
                date_diff = (max_date - min_date).days
                
                if period_comparison == 'Hebdomadaire' and date_diff < 7:
                    st.warning("Les données couvrent moins d'une semaine. L'analyse hebdomadaire ne sera pas pertinente.")
                elif period_comparison == 'Mensuelle' and date_diff < 30:
                    st.warning("Les données couvrent moins d'un mois. L'analyse mensuelle ne sera pas pertinente.")
                elif period_comparison == 'Trimestrielle' and date_diff < 90:
                    st.warning("Les données couvrent moins d'un trimestre. L'analyse trimestrielle ne sera pas pertinente.")
            
            # Calcul des variables pour l'analyse temporelle
            period_type = None
            period_label = None
            if period_comparison != 'Aucune':
                period_type = 'weekly' if period_comparison == 'Hebdomadaire' else ('quarterly' if period_comparison == 'Trimestrielle' else 'monthly')
                period_label = "semaine" if period_comparison == 'Hebdomadaire' else ("trimestre" if period_comparison == 'Trimestrielle' else "mois")
            
            # Affichage des statistiques
            display_statistics(df, period_comparison, period_type, period_label)
            
            # Export CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name="analyse_conversations.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")

if __name__ == "__main__":
    main()