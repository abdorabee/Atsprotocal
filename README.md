# ATS Resume Analyzer MVP

A web-based application that analyzes resumes for ATS (Applicant Tracking System) compatibility and provides actionable improvement suggestions.

## Features

- **Resume Upload**: Support for PDF and text file formats
- **Job Description Analysis**: Optional job description input for targeted keyword analysis
- **Comprehensive Analysis**:
  - Keyword matching with job requirements
  - Employment gap detection
  - Quantified achievements assessment
  - Action verb strength evaluation
  - Resume length and formatting checks
  - Contact information validation
- **Scoring System**: Overall ATS compatibility score (0-100)
- **Actionable Suggestions**: Specific recommendations for improvement

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```
2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)
3. **Upload your resume** using the sidebar file uploader
4. **Optionally paste a job description** for targeted analysis
5. **Click "Analyze Resume"** to get instant feedback

## What Gets Analyzed

### Strengths Detection
- Good keyword match with job requirements
- Appropriate resume length
- Quantified achievements present
- Professional contact information

### Weakness Detection
- Missing keywords from job description
- Employment gaps (>6 months)
- Lack of quantified achievements
- Weak action verbs (e.g., "responsible for" instead of "led")
- Resume too long/short
- Missing contact information

### Improvement Suggestions
- Specific keywords to add
- How to quantify achievements
- Stronger action verbs to use
- Formatting recommendations
- Gap explanation strategies

## Technical Details

### Architecture
- **Frontend**: Streamlit for interactive web UI
- **PDF Parsing**: pdfplumber for text extraction
- **NLP**: spaCy for entity recognition and text processing
- **Text Analysis**: NLTK for tokenization and keyword extraction

### File Structure
```
Atsprotocal/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Security & Privacy
- No data storage - all processing is done in-memory
- Files are processed locally on your machine
- No external API calls for sensitive data

## Limitations (MVP)

- English language only
- Basic date parsing for employment gaps
- Heuristic-based section detection
- Generic keyword database when no job description provided
- Text-only analysis (no visual formatting assessment)

## Future Enhancements

- Machine learning-based resume section parsing
- Industry-specific keyword databases
- Visual formatting analysis
- Multi-language support
- Integration with job boards for real-time keyword updates
- Resume template suggestions
- Batch processing for multiple resumes

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**:
   The app will automatically download required NLTK data on first run.

3. **PDF parsing issues**:
   - Ensure PDF contains selectable text (not just images)
   - Try converting to text format if issues persist

4. **Large file uploads**:
   - File size limit is 5MB
   - Consider compressing or converting large PDFs

### System Requirements
- Python 3.7+
- 2GB RAM minimum
- Internet connection for initial setup (downloading models)

## Contributing

This is an MVP designed for quick prototyping. For production use, consider:
- Adding comprehensive testing
- Implementing proper error logging
- Adding user authentication
- Database integration for analytics
- API development for integration with other tools

## License

This project is for educational and demonstration purposes.
