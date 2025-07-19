# PolyglotPDF

A professional multilingual PDF translation tool that preserves document formatting while translating content using state-of-the-art AI models. Built with advanced PDF processing pipelines and intelligent layout preservation algorithms.

![App Screenshot](https://github.com/user-attachments/assets/55ced4b9-b54d-4a5c-88f8-70ca3423d21c)

- **Multi-Model AI Translation**: Supports Google Gemini 2.0 Flash, Qwen 3 models, Helsinki-NLP transformers, and Google Translate with dynamic model selection[1]
- **Intelligent Layout Preservation**: Advanced bounding box adaptation and geometric calculations to maintain document structure[1]
- **Complex Document Processing**: Handles tables, images, hyperlinks, and multi-column layouts with precision[1]
- **Professional Web Interface**: Built with Streamlit featuring real-time progress tracking, session management, and batch processing[1]
- **Comprehensive Language Support**: Currently supports Japanese and Spanish with extensible architecture for additional languages[1]

## 🏗️ Technical Architecture

### Core Processing Pipeline

The PolyglotPDF system employs a sophisticated multi-stage processing pipeline:

```
PDF Input → Text/Image/Table Extraction → Content Analysis → Translation → Layout Reconstruction → PDF Output
```

### Key Components

#### 1. PDF Processing Engine (`pdf_processing/pdf_translate.py`)
- **Text Extraction**: Uses PyMuPDF (fitz) for precise text span extraction with bounding box coordinates[1]
- **Image Processing**: Leverages Spire.PDF library for image extraction with position metadata[1]
- **Table Detection**: Implements Camelot-py with lattice-based table recognition algorithms[1]
- **Layout Analysis**: Advanced rectangle detection and clustering algorithms for content grouping[1]

#### 2. Translation Models (`translation_models/translators.py`)
**Google Gemini Translator**:
- Implements sophisticated rate limiting with class-level shared limiters (30 RPM, 1M TPM, 200 RPD)[1]
- Features intelligent batch translation to optimize API calls[1]
- Uses token estimation algorithms for cost optimization[1]

**Qwen Japanese Translator**:
- OpenRouter API integration with custom rate limiting (20/min, 50/day)[1]
- Specialized for Japanese translation with cultural context awareness[1]

**Helsinki-NLP Transformer**:
- Local transformer model using MarianMT architecture[1]
- Offline processing capability with PyTorch backend[1]

**Google Translate API**:
- Fallback translation service with gtx client integration[1]
- No API key required for basic functionality[1]

#### 3. User Interface (`app/main.py`)
- **Streamlit Framework**: Professional web interface with custom CSS theming[1]
- **Session Management**: Persistent translation results with UUID-based file tracking[1]
- **Real-time Processing**: Live status updates and progress monitoring[1]
- **API Key Management**: Secure credential handling with encrypted storage[1]

## 🔧 Advanced Technical Details

### Layout Preservation Algorithm

The system implements a sophisticated layout preservation mechanism:

1. **Rectangle Extraction**: Identifies all geometric shapes and content boundaries[1]
2. **Bounding Box Analysis**: Calculates intersection relationships between text spans and layout elements[1]
3. **Content Clustering**: Groups related text elements using distance-based algorithms (configurable threshold)[1]
4. **Dynamic Resizing**: Adapts container dimensions based on translated text length to prevent overflow[1]

### Text Processing Pipeline

```python
# Pseudo-code for text processing flow
spans = extract_text_spans_with_coordinates(pdf_page)
clusters = group_nearby_spans(spans, distance_threshold=20)
translated_clusters = batch_translate(clusters)
reconstructed_layout = adapt_bounding_boxes(translated_clusters)
```

### Table Processing Architecture

- **Detection**: Uses Camelot's lattice-based algorithm to identify table structures[1]
- **Extraction**: Preserves cell relationships and hierarchical data[1]
- **Translation**: Maintains table formatting while translating content[1]
- **Rendering**: Converts to images using ReportLab with custom styling for seamless integration[1]

### Memory Management

The system implements intelligent memory management:
- **Garbage Collection**: Strategic cleanup of PDF objects and image buffers[1]
- **Temporary File Handling**: Secure cleanup of intermediate processing files[1]
- **Resource Optimization**: Efficient handling of large documents with streaming processing[1]

## 📋 Dependencies & Libraries

### Core PDF Processing
- **PyMuPDF (fitz)**: Advanced PDF manipulation and text extraction with coordinate precision[1]
- **Spire.PDF**: Professional-grade image extraction and metadata processing[1]
- **ReportLab**: PDF generation with custom font support and advanced styling[1]
- **Camelot-py**: Machine learning-based table detection and extraction[1]

### AI/ML Components
- **Transformers**: Hugging Face library for Helsinki-NLP model integration[1]
- **PyTorch**: Backend deep learning framework for transformer operations[1]
- **Google Generative AI**: Official Google Gemini API client[1]
- **SentencePiece**: Tokenization for multilingual text processing[1]

### UI/Backend
- **Streamlit**: Modern web framework for interactive applications[1]
- **Pillow (PIL)**: Image processing and manipulation[1]
- **NumPy**: Numerical computing for geometric calculations[1]
- **Matplotlib**: Data visualization support[1]

## 🛠️ Project Structure

```
polyglot-pdf/
│
├── app/                        # Streamlit web interface
│   └── main.py                 # Main application with UI logic
│
├── pdf_processing/             # Core PDF processing engine
│   ├── __init__.py
│   └── pdf_translate.py        # Main translation pipeline
│
├── translation_models/         # AI model implementations
│   ├── __init__.py
│   └── translators.py          # Model classes and API integrations
│
├── .devcontainer/              # Development environment configuration
├── test-manuals/               # Sample documents for testing
├── outputs/                    # Generated files (gitignored)
├── NotoSansJP-*.ttf           # Japanese font files for proper rendering
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Version control exclusions
```

## ⚙️ Setup & Configuration

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/saurabhhbansal/polyglot-pdf.git
cd polyglot-pdf

# Install Python dependencies
pip install -r requirements.txt
```

### 2. System Dependencies (Optional)
For full functionality, install:
- **Ghostscript**: Required for Camelot table processing[1]
- **Tkinter**: GUI backend for certain operations[1]

### 3. Font Installation
The project includes Noto Sans JP fonts for proper Japanese character rendering:
- `NotoSansJP-Regular.ttf`: Standard Japanese text[1]
- `NotoSansJP-Bold.ttf`: Bold Japanese text[1]

### 4. API Configuration
Configure your API keys through the web interface:
- **Google Gemini**: For advanced AI translation[1]
- **OpenRouter**: For Qwen model access[1]
- **Note**: Google Translate works without API keys[1]

### 5. Launch Application
```bash
streamlit run app/main.py
```

## 🔄 Translation Workflow

### Document Processing Steps
1. **Upload**: PDF file validation and temporary storage[1]
2. **Analysis**: Content extraction (text, images, tables) with coordinate mapping[1]
3. **Translation**: Multi-model processing with rate limiting and error handling[1]
4. **Reconstruction**: Layout-aware PDF generation with preserved formatting[1]
5. **Download**: Secure file delivery with session management[1]

### Algorithm Optimization
- **Batch Processing**: Minimizes API calls through intelligent text grouping[1]
- **Caching**: Translation results stored for session persistence[1]
- **Error Recovery**: Graceful handling of API failures and malformed content[1]
- **Performance Monitoring**: Real-time processing metrics and bottleneck identification[1]

## 🚨 Technical Considerations

### Performance Optimization
- **Memory Efficient**: Streaming processing for large documents[1]
- **Concurrent Processing**: Multi-threaded operations where applicable[1]
- **Resource Management**: Automatic cleanup of temporary resources[1]

### Quality Assurance
- **Format Validation**: Pre-processing checks for optimal PDF compatibility[1]
- **Translation Accuracy**: Multiple model comparison for quality optimization[1]
- **Layout Integrity**: Geometric validation of reconstructed documents[1]

### Security Features
- **API Key Protection**: Secure credential handling without persistent storage[1]
- **File Isolation**: Temporary processing in isolated directories[1]
- **Session Security**: UUID-based file tracking prevents cross-user access[1]

## 📊 Usage Guidelines

### Optimal Results
- Use high-quality PDFs with selectable text for best translation accuracy[1]
- Documents with complex layouts benefit from the advanced bounding box algorithms[1]
- Large documents are processed efficiently through the streaming pipeline[1]

### Model Selection
- **Google Gemini**: Best for context-aware translations and complex content[1]
- **Qwen**: Specialized for Japanese with cultural nuance understanding[1]
- **Helsinki-NLP**: Fast offline processing for standard translations[1]
- **Google Translate**: Reliable fallback option with broad language support[1]

## 🌐 Live Demo

Experience PolyglotPDF: [polyglot-pdf.streamlit.app](https://polyglot-pdf.streamlit.app)

## 📄 License

MIT License - see LICENSE file for details.

**Note**: Output files and large dependencies are automatically excluded from version control. The system is designed for production deployment with professional-grade document processing capabilities.[1]
