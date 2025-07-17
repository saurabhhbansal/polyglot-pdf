import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdf_processing.pdf_translate import process_pdf
from translation_models.translators import (
    GeminiTranslator,
    EnglishToSpanishTranslator,
    QwenJapaneseTranslator,
    GoogleTranslateTranslator,
)
import tempfile
import shutil
import datetime
import concurrent.futures
import time

st.set_page_config(
    page_title="PolygotPDF",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📄"
)

# --- Red, Black, and White Theme ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #111;
        color: #fff;
    }
    .css-1d391kg, .css-1v0mbdj, .stFileUploader, .stTextInput, .stTextArea, .stSelectbox, .stNumberInput, .stDateInput, .stTimeInput, .stMultiSelect, .stSlider, .stRadio, .stCheckbox {
        background: #181818 !important;
        color: #fff !important;
        border-radius: 8px;
        border: 1px solid #333 !important;
    }
    .stButton>button {
        background: #e50914 !important;
        color: #fff !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(229,9,20,0.15);
        transition: background 0.2s, color 0.2s;
    }
    .stButton>button:hover {
        background: #fff !important;
        color: #e50914 !important;
        border: 1px solid #e50914 !important;
    }
    .stDownloadButton>button {
        background: #fff !important;
        color: #e50914 !important;
        border-radius: 8px;
        border: 2px solid #e50914 !important;
        font-weight: bold;
        transition: background 0.2s, color 0.2s;
    }
    .stDownloadButton>button:hover {
        background: #e50914 !important;
        color: #fff !important;
        border: 2px solid #fff !important;
    }
    .stSpinner>div>div {
        color: #e50914 !important;
    }
    .stAlert, .stSuccess, .stInfo, .stError {
        border-radius: 8px;
        font-weight: bold;
    }
    .stSuccess {
        background: #e5091422 !important;
        color: #fff !important;
        border-left: 6px solid #e50914 !important;
    }
    .stError {
        background: #e5091444 !important;
        color: #fff !important;
        border-left: 6px solid #e50914 !important;
    }
    .stInfo {
        background: #fff2 !important;
        color: #fff !important;
        border-left: 6px solid #fff !important;
    }
    .stFileUploader label {
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Restore headings ---
st.title("PolyglotPDF")
st.caption("Translate your PDFs into any language. Make your documents truly global.")
st.markdown("Upload a PDF and get your translated document in seconds.")

# --- Supported languages and translators ---
LANGUAGES = {
    "Japanese": "ja",
    "Spanish": "es"
}
# --- Add Qwen model for Japanese ---
TRANSLATOR_OPTIONS = {
    "Japanese": ["Google Gemini", "Qwen", "Google Translate"],
    "Spanish": ["Google Gemini", "Helsinki-NLP", "Google Translate"]
}

# Remove is_translating logic and always enable widgets
# --- Combined multi-select for (language, translator) pairs ---
combined_options = []
for lang in LANGUAGES:
    for translator in TRANSLATOR_OPTIONS[lang]:
        combined_options.append(f"{lang} ({translator})")

# Place the file uploader at the correct location
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Always define file_base and now_str so they are available everywhere
file_base = os.path.splitext(uploaded_file.name)[0] if uploaded_file else ""
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

translate_btn = st.button("Translate PDF", type="primary", use_container_width=True)

# Ensure session state is initialized before any parallel processing
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ''
if 'openrouter_api_key' not in st.session_state:
    st.session_state['openrouter_api_key'] = ''
if 'show_api_key_dialog' not in st.session_state:
    st.session_state['show_api_key_dialog'] = False
if 'translation_results' not in st.session_state:
    st.session_state['translation_results'] = {}
if 'translation_status' not in st.session_state:
    st.session_state['translation_status'] = {}
if 'translation_result_keys' not in st.session_state:
    st.session_state['translation_result_keys'] = []

stop_key = "stop_translation"
if stop_key not in st.session_state:
    st.session_state[stop_key] = False

# --- Combined API Key Dialog for both Gemini and OpenRouter ---
@st.dialog("Enter your API Keys")
def api_key_dialog():
    gemini_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get('gemini_api_key', ''))
    openrouter_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.get('openrouter_api_key', ''))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", key='save_both_keys'):
            st.session_state['gemini_api_key'] = gemini_key
            st.session_state['openrouter_api_key'] = openrouter_key
            st.session_state['show_api_key_dialog'] = False
            st.rerun()
    with col2:
        if st.button("Cancel", key='cancel_both_keys'):
            st.session_state['show_api_key_dialog'] = False
            st.rerun()

# Show combined dialog if either key is missing or either button is clicked
if not st.session_state['gemini_api_key'] or not st.session_state['openrouter_api_key'] or st.session_state['show_api_key_dialog']:
    st.session_state['show_api_key_dialog'] = True
    api_key_dialog()

# --- Combined row for translation options and single API key button, all centered ---
if 'selected_translators' not in st.session_state:
    st.session_state['selected_translators'] = {lang: set(TRANSLATOR_OPTIONS[lang]) for lang in LANGUAGES}

opt_col, key_col = st.columns([8, 1])
with opt_col:
    st.markdown('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
    st.markdown('**Select translation options:**')
    check_cols = st.columns(len(LANGUAGES))
    selected_pairs = []
    for idx, lang in enumerate(LANGUAGES):
        with check_cols[idx]:
            st.markdown(f'<div style="text-align:center;font-weight:bold;">{lang}</div>', unsafe_allow_html=True)
            for translator in TRANSLATOR_OPTIONS[lang]:
                checked = translator in st.session_state['selected_translators'][lang]
                new_checked = st.checkbox(translator, value=checked, key=f'{lang}_{translator}_chk')
                if new_checked:
                    st.session_state['selected_translators'][lang].add(translator)
                else:
                    st.session_state['selected_translators'][lang].discard(translator)
                if new_checked:
                    selected_pairs.append(f"{lang} ({translator})")
    st.markdown('</div>', unsafe_allow_html=True)
with key_col:
    st.write("")
    st.write("")
    api_btn_container = st.container()
    with api_btn_container:
        api_btn = st.button('🔑 API Keys', help='Set API Keys', key='change_api_keys_btn')
        st.markdown("""
            <style>
            div[data-testid=\"column\"] button#change_api_keys_btn {
                height: auto !important;
                width: auto !important;
                min-width: 120px !important;
                min-height: 48px !important;
                font-size: 1.2em !important;
                border-radius: 8px !important;
                padding: 0 !important;
                margin-top: 8px !important;
                margin-bottom: 8px !important;
                background: #23232b !important;
                color: #ffd700 !important;
                border: 1px solid #333 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if api_btn:
        st.session_state['show_api_key_dialog'] = True

# --- Notification logic for translation complete message ---
if 'translation_notification' not in st.session_state:
    st.session_state['translation_notification'] = None
if 'translation_notification_time' not in st.session_state:
    st.session_state['translation_notification_time'] = 0

def show_translation_notification(message):
    st.session_state['translation_notification'] = message
    st.session_state['translation_notification_time'] = time.time()

def clear_translation_notification():
    st.session_state['translation_notification'] = None
    st.session_state['translation_notification_time'] = 0

def format_elapsed_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m} min {s} sec"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h} hr {m} min {s} sec"

if translate_btn and uploaded_file and selected_pairs:
    # Clear previous results for this file/language(s)
    for pair in selected_pairs:
        lang_name, translator_name = pair.split(" (")
        lang_name = lang_name.strip()
        translator_name = translator_name.replace(")", "").strip()
        lang_code = LANGUAGES[lang_name]
        key = f"{file_base}_{lang_code}_{translator_name.replace(' ', '').lower()}"
        st.session_state['translation_results'].pop(key, None)
        st.session_state['translation_status'].pop(key, None)
    st.session_state['translation_result_keys'] = []

    output_dir = os.path.join("outputs", f"{file_base}_{now_str}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        # Make a working copy for processing
        work_pdf_path = os.path.join(tmpdir, "work_" + uploaded_file.name)
        shutil.copy2(pdf_path, work_pdf_path)
        os.makedirs(output_dir, exist_ok=True)

        # Remove the progress bar entirely
        # progress_bar = st.progress(0)
        total = len(selected_pairs) # Total number of translations to run
        start_time = time.time()
        files_generated = 0

        # Show Stop button only while processing
        stop_placeholder = st.empty()
        if stop_placeholder.button("Stop Translation", type="secondary"):
            st.session_state[stop_key] = True

        # In the translation loop, instantiate the correct translator for each language
        translation_results = []
        for idx, pair in enumerate(selected_pairs):
            lang_name, translator_name = pair.split(" (")
            lang_name = lang_name.strip()
            translator_name = translator_name.replace(")", "").strip()
            lang_code = LANGUAGES[lang_name]
            key = f"{file_base}_{lang_code}_{translator_name.replace(' ', '').lower()}"
            st.session_state['translation_status'][key] = 'processing'
            output_pdf = os.path.join(output_dir, f"translated_{file_base}_{lang_code}_{translator_name.replace(' ', '').lower()}.pdf")
            output_json = os.path.join(output_dir, f"translated_{file_base}_{lang_code}_{translator_name.replace(' ', '').lower()}.json")
            try:
                def should_stop():
                    return st.session_state.get(stop_key, False)
                if translator_name == "Google Gemini":
                    if not st.session_state['gemini_api_key']:
                        st.error("Please enter your Gemini API key to translate.")
                        continue
                    translator = GeminiTranslator(lang_code, api_key=st.session_state['gemini_api_key'])
                elif translator_name == "Helsinki-NLP":
                    if lang_name == "Spanish":
                        translator = EnglishToSpanishTranslator()
                    else:
                        st.error(f"Helsinki-NLP does not support {lang_name}.")
                        continue
                elif translator_name == "Qwen":
                    if not st.session_state['openrouter_api_key']:
                        st.error("Please enter your OpenRouter API key to translate with Qwen.")
                        continue
                    translator = QwenJapaneseTranslator(st.session_state['openrouter_api_key'])
                elif translator_name == "Google Translate":
                    target_lang = 'ja' if lang_name == 'Japanese' else 'es'
                    translator = GoogleTranslateTranslator(target_lang)
                else:
                    st.error(f"Unknown translator: {translator_name}")
                    continue
                
                with st.spinner(f"Translating {lang_name} using {translator_name}..."):
                    process_pdf(
                        input_pdf=work_pdf_path,
                        output_pdf=output_pdf,
                        json_file=output_json,
                        font_path="NotoSansJP-Regular.ttf",
                        translator=translator,
                        dist_thr=20,
                        pad=6,
                        fsize=10,
                        target_lang=lang_code,
                        should_stop=should_stop
                    )
                if os.path.exists(output_pdf):
                    with open(output_pdf, "rb") as f:
                        pdf_bytes = f.read()
                    with open(output_json, "rb") as f:
                        json_bytes = f.read()
                    st.session_state['translation_results'][key] = {
                        'lang': lang_name,
                        'translator': translator_name,
                        'pdf_bytes': pdf_bytes,
                        'json_bytes': json_bytes,
                        'pdf_name': os.path.basename(output_pdf),
                        'json_name': os.path.basename(output_json)
                    }
                    st.session_state['translation_status'][key] = 'done'
                    translation_results.append(key)
                    st.success(f"{lang_name} ({translator_name}) translation complete!")
                else:
                    st.session_state['translation_status'][key] = 'error: output file not created'
                    st.error(f"Translation failed or file not created for {lang_name} ({translator_name}).")
            except Exception as e:
                st.session_state['translation_status'][key] = f'error: {e}'
                st.error(f"Translation failed for {lang_name} ({translator_name}): {e}")
        # --- Persist translation result keys ---
        st.session_state['translation_result_keys'] = translation_results
        stop_placeholder.empty()
        elapsed = time.time() - start_time
        if st.session_state.get(stop_key):
            st.session_state[stop_key] = False  # Reset stop flag
            if files_generated > 0:
                st.toast(f"Translation stopped. {files_generated} file(s) generated. Total time: {format_elapsed_time(elapsed)}.")
                # --- Always show download buttons for all results in session state ---
                if st.session_state.get('translation_result_keys'):
                    st.markdown("### Download your translated files:")
                    for key in st.session_state['translation_result_keys']:
                        result = st.session_state['translation_results'][key]
                        st.download_button(
                            label=f"Download PDF: {result['lang']} ({result['translator']})",
                            data=result['pdf_bytes'],
                            file_name=result['pdf_name'],
                            mime="application/pdf",
                            key=f"dl_pdf_{key}"
                        )
            else:
                st.toast(f"No files were generated before stopping.")
        else:
            st.toast(f"All selected translations are complete! Total time: {format_elapsed_time(elapsed)}.")
            # --- Always show download buttons for all results in session state ---
            if st.session_state.get('translation_result_keys'):
                st.markdown("### Download your translated files:")
                for key in st.session_state['translation_result_keys']:
                    result = st.session_state['translation_results'][key]
                    st.download_button(
                        label=f"Download PDF: {result['lang']} ({result['translator']})",
                        data=result['pdf_bytes'],
                        file_name=result['pdf_name'],
                        mime="application/pdf",
                        key=f"dl_pdf_{key}"
                    )

# --- Always show download buttons for all results in session state ---
if st.session_state.get('translation_results'):
    st.markdown("### Download your translated files:")
    for key, result in st.session_state['translation_results'].items():
        st.download_button(
            label=f"Download PDF: {result['lang']} ({result['translator']})",
            data=result['pdf_bytes'],
            file_name=result['pdf_name'],
            mime="application/pdf",
            key=f"dl_pdf_{key}"
        )

if not (translate_btn and uploaded_file and selected_pairs):
    st.info(" Upload a PDF, select language(s), and click 'Translate PDF' to get started.")

# --- Show notification if present and not expired ---
if st.session_state['translation_notification']:
    elapsed = time.time() - st.session_state['translation_notification_time']
    if elapsed < 60:
        col1, col2 = st.columns([10,1])
        with col1:
            st.success(st.session_state['translation_notification'])
        with col2:
            if st.button('✖', key='dismiss_notification'):
                clear_translation_notification()
    else:
        clear_translation_notification()
