from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import AppConfig, get_config_path
from pipeline import RAGPipeline
from sync import sync_folder


st.set_page_config(
    page_title="Lear — Assistant documentaire local",
    page_icon="L",
    layout="wide",
)


LEAR_LOGO = Path(__file__).resolve().parent / "assets" / "lear_logo.png"


def apply_lear_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lear-red: #ed1b2f;
            --lear-red-dark: #c81025;
            --lear-black: #111111;
            --lear-gray: #5b6068;
        }
        .stApp { background: #ffffff; }
        [data-testid="stHeader"] { background: #ffffff; }
        [data-testid="stSidebar"] { border-right: 1px solid #e5e5e5; }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: var(--lear-black); }
        h1, h2, h3 {
            color: var(--lear-black);
            letter-spacing: -0.02em;
        }
        h1 { font-weight: 650; }
        .lear-brand {
            border-bottom: 4px solid var(--lear-red);
            padding: 0 0 0.75rem 0;
            margin-bottom: 1.5rem;
        }
        .lear-kicker {
            color: var(--lear-red);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .lear-subtitle {
            color: var(--lear-gray);
            font-size: 1rem;
            margin-top: -0.4rem;
        }
        div.stButton > button[kind="primary"] {
            background: var(--lear-red);
            border-color: var(--lear-red);
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--lear-red-dark);
            border-color: var(--lear-red-dark);
        }
        [data-testid="stChatMessage"] { border-radius: 8px; }
        .lear-note {
            border-left: 4px solid var(--lear-red);
            background: #f7f7f7;
            padding: 0.75rem 1rem;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    left, right = st.columns([1, 2.8], vertical_alignment="center")
    with left:
        if LEAR_LOGO.exists():
            st.image(str(LEAR_LOGO), width=220)
    with right:
        st.markdown(
            '<div class="lear-kicker">Lear Corporation · Meknes</div>',
            unsafe_allow_html=True,
        )
        st.markdown("# Assistant documentaire local")
        st.markdown(
            '<div class="lear-subtitle">Recherche fiable et confidentielle dans les documents de l’entreprise</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="lear-brand"></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(
    database_path: str,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
) -> RAGPipeline:
    return RAGPipeline(
        database_path=database_path,
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )


def reload_pipeline() -> None:
    get_pipeline.clear()
    st.rerun()


def save_model_automatically() -> None:
    selected = st.session_state.get("model_value", "").strip()
    if not selected:
        return
    config = AppConfig.load()
    if config.ollama_model != selected:
        config.ollama_model = selected
        config.save()
        get_pipeline.clear()
        st.session_state["model_saved_message"] = f"Modèle enregistré automatiquement : {selected}"


def display_sync_report(report: dict) -> None:
    indexed = len(report.get("indexed", []))
    skipped = len(report.get("skipped", []))
    removed = len(report.get("removed", []))
    errors = report.get("errors", [])
    if errors:
        st.warning(f"Synchronisation terminée avec {len(errors)} erreur(s).")
        with st.expander("Détails des erreurs"):
            st.json(errors)
    else:
        st.success(
            f"Synchronisation terminée : {indexed} indexé(s), "
            f"{skipped} inchangé(s), {removed} supprimé(s)."
        )


def format_sources(answer) -> None:
    if not answer.sources:
        return
    with st.expander("Sources utilisées"):
        for index, source in enumerate(answer.sources, start=1):
            chunk = source.chunk
            metadata = chunk.metadata or {}
            document = metadata.get("document_name", f"document #{chunk.document_id}")
            page_start = metadata.get("page_start", chunk.page_start)
            page_end = metadata.get("page_end", chunk.page_end)
            section = (
                metadata.get("section")
                or metadata.get("parent_heading")
                or "section non renseignée"
            )
            st.markdown(
                f"**[S{index}]** `{document}` — pages **{page_start}-{page_end}** — "
                f"section : **{section}** — type `{chunk.chunk_type}`"
            )
            st.caption(chunk.text)


def main() -> None:
    apply_lear_theme()
    config = AppConfig.load()
    if "folder_value" not in st.session_state:
        st.session_state.folder_value = str(config.documents_path)
    if "model_value" not in st.session_state:
        st.session_state.model_value = config.ollama_model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_brand_header()
    st.markdown(
        '<div class="lear-note">Déposez vos PDF dans le dossier configuré, '
        "synchronisez l’index, puis posez vos questions. Les réponses sont générées "
        'localement à partir des documents indexés.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if LEAR_LOGO.exists():
            st.image(str(LEAR_LOGO), width=170)
        st.header("Configuration")
        folder = st.text_input(
            "Dossier contenant les PDF",
            key="folder_value",
            help="Tous les PDF de ce dossier et de ses sous-dossiers seront indexés.",
        )
        if st.button("Enregistrer le dossier", use_container_width=True):
            selected = Path(folder).expanduser().resolve()
            config.documents_folder = str(selected)
            config.save()
            st.success(f"Dossier enregistré : {selected}")
            reload_pipeline()

        st.divider()
        current_pipeline = get_pipeline(
            str(config.database_file),
            config.embedding_model,
            config.ollama_model,
            config.ollama_base_url,
        )
        installed_models = current_pipeline.ollama.list_models()
        model_options = sorted(set(installed_models + [config.ollama_model]))
        st.selectbox(
            "Modèle Ollama",
            options=model_options or [config.ollama_model],
            index=(
                model_options.index(st.session_state.model_value)
                if st.session_state.model_value in model_options
                else 0
            ),
            key="model_value",
            on_change=save_model_automatically,
        )
        saved_message = st.session_state.pop("model_saved_message", None)
        if saved_message:
            st.success(saved_message)
        if st.button("Actualiser les modèles", use_container_width=True):
            st.rerun()

        st.divider()
        st.caption(f"Configuration : {get_config_path()}")
        st.caption(f"Base SQLite : {config.database_file}")

    active_folder = Path(config.documents_folder).expanduser().resolve()
    pipeline = get_pipeline(
        str(config.database_file),
        config.embedding_model,
        config.ollama_model,
        config.ollama_base_url,
    )

    col_sync, col_stats = st.columns([1, 2])
    with col_sync:
        if st.button("Synchroniser maintenant", type="primary", use_container_width=True):
            with st.spinner("Indexation du dossier en cours…"):
                report = sync_folder(pipeline, active_folder)
            display_sync_report(report)
    with col_stats:
        st.info(
            f"Dossier actif : `{active_folder}`  \n"
            f"Documents indexés : **{pipeline.repository.document_count()}** · "
            f"Chunks : **{pipeline.repository.count_chunks()}**"
        )

    if not active_folder.exists():
        st.warning("Le dossier actif n’existe pas encore. La synchronisation le créera automatiquement.")

    st.divider()
    st.subheader("Assistant documentaire")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("answer") is not None:
                format_sources(message["answer"])

    question = st.chat_input("Posez une question sur les PDF indexés…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Recherche des passages et génération de la réponse…"):
                    answer = pipeline.ask(question)
                st.markdown(answer.text)
                format_sources(answer)
                st.caption(f"Latence : {json.dumps(answer.latency_ms, ensure_ascii=False)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer.text, "answer": answer}
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.info(
                    "Sous Windows, ouvrez PowerShell et exécutez :\n\n"
                    "```powershell\nollama serve\n```\n\n"
                    "Dans une deuxième fenêtre PowerShell, vérifiez le modèle :\n\n"
                    f"```powershell\nollama pull {config.ollama_model}\nollama list\n```"
                )


if __name__ == "__main__":
    main()
