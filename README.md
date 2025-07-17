# PolyglotPDF

A professional PDF translation tool that allows you to translate PDF documents into multiple languages using advanced AI models and APIs.

![App Screenshot](https://github.com/user-attachments/assets/55ced4b9-b54d-4a5c-88f8-70ca3423d21c)

## Features
- Translate PDFs to Japanese, Spanish, and more using Google Gemini, Qwen, and other models
- Streamlit web interface for easy uploads and downloads
- Table, image, and hyperlink preservation
- Batch translation and API key management

## Project Structure
```
polyglot-pdf/
│
├── app/                        # Streamlit UI code
│   └── main.py                 # Main Streamlit app
│
├── pdf_processing/             # PDF processing logic
│   ├── __init__.py
│   └── pdf_translate.py
│
├── translation_models/         # Translation model classes
│   ├── __init__.py
│   └── translators.py
│
├── outputs/                    # Output files (ignored by git)
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and instructions
├── .gitignore                  # Files/folders to ignore in git
```

## Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **(Optional) Install system dependencies:**
   - [Camelot-py](https://camelot-py.readthedocs.io/en/master/user/install-deps.html) may require Ghostscript and Tkinter.

3. **Run the app:**
   ```sh
   streamlit run app/main.py
   ```

4. **API Keys:**
   - You will need API keys for Google Gemini and/or OpenRouter (Qwen). Enter them in the app UI when prompted.

## Usage
- Upload a PDF, select target language(s) and translation model(s), and click "Translate PDF".
- Download your translated PDF(s) from the results section.

## Notes
- Outputs and large files are ignored by git (see `.gitignore`).
- For best results, use high-quality PDFs with selectable text.

## License
MIT License 
>>>>>>> e38997b (Initial commit: organized project structure and code)
