from __future__ import annotations
from typing import Any

# Prompt système simplifié pour un modèle 3B
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de rapports. 
Ta mission est de répondre aux questions en utilisant UNIQUEMENT les extraits fournis.

RÈGLES CRUCIALES :
1. Si l'information n'est pas dans les extraits, réponds exactement : "Je ne trouve pas cette information dans les documents indexés."
2. Ne fais aucune supposition. Si un chiffre n'est pas écrit noir sur blanc, il n'existe pas.
3. Pour chaque fait ou chiffre cité, indique la page entre crochets, par exemple : [Page 10].
4. Sois bref et précis. Ne fais pas d'introduction inutile.
5.Quand une information source est présentée sous forme de liste ou de liste
à puces (avec des "o", "•", "▪", ou des tirets), reproduis cette structure
dans ta réponse en utilisant le format Markdown pour les listes (une ligne
par élément, précédée de "-"). Ne fusionne jamais plusieurs points de liste
en un seul paragraphe continu.

6.Traduis intégralement chaque phrase en français naturel — ne laisse jamais
de mots ou de fragments de phrase en anglais au milieu d'une phrase française
(par exemple, jamais "review si any application" — dis plutôt "vérifier si
une application")."""

def build_user_prompt(question: str, sources: list[Any]) -> str:
    context_parts = []
    for i, src in enumerate(sources, 1):
        # On inclut le numéro de page et le texte
        page = src.chunk.page_start
        text = src.chunk.text
        context_parts.append(f"--- EXTRAIT {i} (Page {page}) ---\n{text}")
    
    context = "\n\n".join(context_parts)
    
    return f"""Voici les extraits du document :
{context}

QUESTION : {question}

RÉPONSE :"""
