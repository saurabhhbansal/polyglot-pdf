# PolyglotPDF

A professional multilingual PDF translation tool that preserves document formatting while translating content using state-of-the-art AI models. Built with advanced PDF processing pipelines and intelligent layout preservation algorithms.

![App Screenshot](https://github.com/user-attachments/assets/55ced4b9-b54d-4a5c-88f8-70ca3423d21c)

- **Multi-Model AI Translation**: Supports Google Gemini 2.0 Flash, Qwen 3 models, Helsinki-NLP transformers, and Google Translate with dynamic model selection
- **Intelligent Layout Preservation**: Advanced bounding box adaptation and geometric calculations to maintain document structure
- **Complex Document Processing**: Handles tables, images, hyperlinks, and multi-column layouts with precision
- **Professional Web Interface**: Built with Streamlit featuring real-time progress tracking, session management, and batch processing
- **Comprehensive Language Support**: Currently supports Japanese and Spanish with extensible architecture for additional languages

## 🏗️ Technical Architecture

### Core Processing Pipeline

The PolyglotPDF system employs a sophisticated multi-stage processing pipeline:

```
PDF Input → Text/Image/Table Extraction → Content Analysis → Translation → Layout Reconstruction → PDF Output
```

### Key Components

#### 1. PDF Processing Engine (`pdf_processing/pdf_translate.py`)
- **Text Extraction**: Uses PyMuPDF (fitz) for precise text span extraction with bounding box coordinates
- **Image Processing**: Leverages Spire.PDF library for image extraction with position metadata
- **Table Detection**: Implements Camelot-py with lattice-based table recognition algorithms
- **Layout Analysis**: Advanced rectangle detection and clustering algorithms for content grouping

#### 2. Translation Models (`translation_models/translators.py`)
**Google Gemini Translator**:
- Implements sophisticated rate limiting with class-level shared limiters (30 RPM, 1M TPM, 200 RPD)
- Features intelligent batch translation to optimize API calls
- Uses token estimation algorithms for cost optimization

**Qwen Japanese Translator**:
- OpenRouter API integration with custom rate limiting (20/min, 50/day)
- Specialized for Japanese translation with cultural context awareness

**Helsinki-NLP Transformer**:
- Local transformer model using MarianMT architecture
- Offline processing capability with PyTorch backend

**Google Translate API**:
- Fallback translation service with gtx client integration
- No API key required for basic functionality

#### 3. User Interface (`app/main.py`)
- **Streamlit Framework**: Professional web interface with custom CSS theming
- **Session Management**: Persistent translation results with UUID-based file tracking
- **Real-time Processing**: Live status updates and progress monitoring
- **API Key Management**: Secure credential handling with encrypted storage

## 🔧 Advanced Technical Details

### Layout Preservation Algorithm

The system implements a sophisticated layout preservation mechanism:

1. **Rectangle Extraction**: Identifies all geometric shapes and content boundaries
2. **Bounding Box Analysis**: Calculates intersection relationships between text spans and layout elements
3. **Content Clustering**: Groups related text elements using distance-based algorithms (configurable threshold)
4. **Dynamic Resizing**: Adapts container dimensions based on translated text length to prevent overflow

### Text Processing Pipeline

```python
# Pseudo-code for text processing flow
spans = extract_text_spans_with_coordinates(pdf_page)
clusters = group_nearby_spans(spans, distance_threshold=20)
translated_clusters = batch_translate(clusters)
reconstructed_layout = adapt_bounding_boxes(translated_clusters)
```

### Table Processing Architecture

- **Detection**: Uses Camelot's lattice-based algorithm to identify table structures
- **Extraction**: Preserves cell relationships and hierarchical data
- **Translation**: Maintains table formatting while translating content
- **Rendering**: Converts to images using ReportLab with custom styling for seamless integration

### Memory Management

The system implements intelligent memory management:
- **Garbage Collection**: Strategic cleanup of PDF objects and image buffers
- **Temporary File Handling**: Secure cleanup of intermediate processing files
- **Resource Optimization**: Efficient handling of large documents with streaming processing

## 📋 Dependencies & Libraries

### Core PDF Processing
- **PyMuPDF (fitz)**: Advanced PDF manipulation and text extraction with coordinate precision
- **Spire.PDF**: Professional-grade image extraction and metadata processing
- **ReportLab**: PDF generation with custom font support and advanced styling
- **Camelot-py**: Machine learning-based table detection and extraction

### AI/ML Components
- **Transformers**: Hugging Face library for Helsinki-NLP model integration
- **PyTorch**: Backend deep learning framework for transformer operations
- **Google Generative AI**: Official Google Gemini API client
- **SentencePiece**: Tokenization for multilingual text processing

### UI/Backend
- **Streamlit**: Modern web framework for interactive applications
- **Pillow (PIL)**: Image processing and manipulation
- **NumPy**: Numerical computing for geometric calculations
- **Matplotlib**: Data visualization support

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
- **Ghostscript**: Required for Camelot table processing
- **Tkinter**: GUI backend for certain operations

### 3. Font Installation
The project includes Noto Sans JP fonts for proper Japanese character rendering:
- `NotoSansJP-Regular.ttf`: Standard Japanese text
- `NotoSansJP-Bold.ttf`: Bold Japanese text

### 4. API Configuration
Configure your API keys through the web interface:
- **Google Gemini**: For advanced AI translation
- **OpenRouter**: For Qwen model access
- **Note**: Google Translate works without API keys

### 5. Launch Application
```bash
streamlit run app/main.py
```

## 🔄 Translation Workflow

### Document Processing Steps
1. **Upload**: PDF file validation and temporary storage
2. **Analysis**: Content extraction (text, images, tables) with coordinate mapping
3. **Translation**: Multi-model processing with rate limiting and error handling
4. **Reconstruction**: Layout-aware PDF generation with preserved formatting
5. **Download**: Secure file delivery with session management

### Algorithm Optimization
- **Batch Processing**: Minimizes API calls through intelligent text grouping
- **Caching**: Translation results stored for session persistence
- **Error Recovery**: Graceful handling of API failures and malformed content
- **Performance Monitoring**: Real-time processing metrics and bottleneck identification

## 🚨 Technical Considerations

### Performance Optimization
- **Memory Efficient**: Streaming processing for large documents
- **Concurrent Processing**: Multi-threaded operations where applicable
- **Resource Management**: Automatic cleanup of temporary resources

### Quality Assurance
- **Format Validation**: Pre-processing checks for optimal PDF compatibility
- **Translation Accuracy**: Multiple model comparison for quality optimization
- **Layout Integrity**: Geometric validation of reconstructed documents

### Security Features
- **API Key Protection**: Secure credential handling without persistent storage
- **File Isolation**: Temporary processing in isolated directories
- **Session Security**: UUID-based file tracking prevents cross-user access

## 📊 Usage Guidelines

### Optimal Results
- Use high-quality PDFs with selectable text for best translation accuracy
- Documents with complex layouts benefit from the advanced bounding box algorithms
- Large documents are processed efficiently through the streaming pipeline

### Model Selection
- **Google Gemini**: Best for context-aware translations and complex content
- **Qwen**: Specialized for Japanese with cultural nuance understanding
- **Helsinki-NLP**: Fast offline processing for standard translations
- **Google Translate**: Reliable fallback option with broad language support

## 🌐 Live Demo

Experience PolyglotPDF: [polyglot-pdf.streamlit.app](https://polyglot-pdf.streamlit.app)

## 📄 License

MIT License - see LICENSE file for details.

**Note**: Output files and large dependencies are automatically excluded from version control. The system is designed for production deployment with professional-grade document processing capabilities.
