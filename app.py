"""
ATS Resume Analyzer MVP
A web-based application to analyze resumes for ATS compatibility and provide improvement suggestions.

To run this application:
1. Install dependencies: pip install streamlit pdfplumber spacy nltk
2. Download spaCy model: python -m spacy download en_core_web_sm
3. Run the app: streamlit run app.py
"""

import streamlit as st
import pdfplumber
import spacy
import nltk
import re
import io
from collections import Counter
from datetime import datetime
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

class ResumeAnalyzer:
    def __init__(self):
        self.nlp = load_spacy_model()
        self.stop_words = set(stopwords.words('english'))
        
        # Common ATS keywords for fallback when no job description is provided
        self.default_keywords = {
            'technical': ['python', 'java', 'sql', 'javascript', 'html', 'css', 'react', 'node.js', 
                         'aws', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'api', 'database'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 
                           'analytical', 'creative', 'adaptable', 'organized', 'detail-oriented'],
            'business': ['project management', 'budget', 'revenue', 'sales', 'marketing', 
                        'strategy', 'operations', 'customer service', 'stakeholder']
        }
        
        # Weak action verbs to flag
        self.weak_verbs = ['responsible for', 'duties included', 'worked on', 'helped with', 
                          'assisted', 'participated in', 'involved in', 'tasked with']
        
        # Strong action verbs to suggest
        self.strong_verbs = ['led', 'managed', 'developed', 'implemented', 'achieved', 'improved',
                           'optimized', 'created', 'designed', 'executed', 'delivered', 'increased']

    def parse_resume(self, file_content, file_type):
        """Parse resume from PDF or text file"""
        try:
            if file_type == 'pdf':
                return self._parse_pdf(file_content)
            else:
                return file_content.decode('utf-8')
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")

    def _parse_pdf(self, pdf_content):
        """Extract text from PDF using pdfplumber"""
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,()@]', '', text)
        return text.strip()

    def extract_sections(self, text):
        """Extract key sections from resume using heuristics"""
        sections = {
            'contact': '',
            'summary': '',
            'skills': '',
            'experience': '',
            'education': ''
        }
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        lines = text.split('\n')
        
        current_section = None
        section_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if any(keyword in line_lower for keyword in ['email', 'phone', '@']):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'contact'
                section_content = [line]
            elif any(keyword in line_lower for keyword in ['summary', 'objective', 'profile']):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'summary'
                section_content = []
            elif any(keyword in line_lower for keyword in ['skills', 'technical skills', 'competencies']):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'skills'
                section_content = []
            elif any(keyword in line_lower for keyword in ['experience', 'employment', 'work history', 'professional experience']):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'experience'
                section_content = []
            elif any(keyword in line_lower for keyword in ['education', 'academic', 'qualifications']):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'education'
                section_content = []
            else:
                if current_section:
                    section_content.append(line)
        
        # Add the last section
        if current_section:
            sections[current_section] = '\n'.join(section_content)
        
        return sections

    def extract_keywords(self, text, job_description=None):
        """Extract keywords from text"""
        if not self.nlp:
            return []
            
        # Use job description keywords if provided, otherwise use defaults
        if job_description:
            target_keywords = self._extract_job_keywords(job_description)
        else:
            target_keywords = []
            for category in self.default_keywords.values():
                target_keywords.extend(category)
        
        # Extract keywords from resume
        doc = self.nlp(text.lower())
        resume_keywords = []
        
        for token in doc:
            if (token.text not in self.stop_words and 
                not token.is_punct and 
                len(token.text) > 2 and
                token.text.isalpha()):
                resume_keywords.append(token.text)
        
        # Also extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE']:
                resume_keywords.append(ent.text.lower())
        
        return list(set(resume_keywords)), target_keywords

    def _extract_job_keywords(self, job_description):
        """Extract important keywords from job description"""
        if not self.nlp:
            return []
            
        doc = self.nlp(job_description.lower())
        keywords = []
        
        for token in doc:
            if (token.text not in self.stop_words and 
                not token.is_punct and 
                len(token.text) > 2 and
                (token.pos_ in ['NOUN', 'ADJ'] or token.ent_type_)):
                keywords.append(token.text)
        
        # Extract named entities
        for ent in doc.ents:
            keywords.append(ent.text.lower())
        
        return list(set(keywords))

    def detect_employment_gaps(self, experience_text):
        """Detect employment gaps in experience section"""
        # Extract dates using regex
        date_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\b\d{1,2}/\d{4}|\b\d{4}\b'
        dates = re.findall(date_pattern, experience_text.lower())
        
        gaps = []
        if len(dates) >= 2:
            # Simple gap detection - this is a basic implementation
            # In a production system, you'd want more sophisticated date parsing
            years = []
            for date in dates:
                year_match = re.search(r'\d{4}', date)
                if year_match:
                    years.append(int(year_match.group()))
            
            if years:
                years.sort()
                for i in range(len(years) - 1):
                    if years[i+1] - years[i] > 1:
                        gaps.append(f"Gap detected between {years[i]} and {years[i+1]}")
        
        return gaps

    def check_quantification(self, experience_text):
        """Check for quantified achievements in experience"""
        # Look for numbers, percentages, dollar amounts
        quantification_patterns = [
            r'\d+%',  # percentages
            r'\$\d+',  # dollar amounts
            r'\d+\s*(million|thousand|k|m)',  # large numbers
            r'\d+\s*(people|employees|team members|clients|customers)',  # team sizes
            r'increased.*\d+',  # increased by X
            r'reduced.*\d+',  # reduced by X
            r'improved.*\d+'   # improved by X
        ]
        
        sentences = sent_tokenize(experience_text)
        quantified_sentences = []
        unquantified_sentences = []
        
        for sentence in sentences:
            if any(re.search(pattern, sentence.lower()) for pattern in quantification_patterns):
                quantified_sentences.append(sentence.strip())
            elif len(sentence.strip()) > 20:  # Ignore very short sentences
                unquantified_sentences.append(sentence.strip())
        
        return quantified_sentences, unquantified_sentences

    def detect_weak_language(self, text):
        """Detect weak action verbs and passive voice"""
        weak_phrases = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for weak_verb in self.weak_verbs:
                if weak_verb in sentence_lower:
                    weak_phrases.append(sentence.strip())
                    break
        
        return weak_phrases

    def analyze_resume(self, resume_text, job_description=None):
        """Main analysis function"""
        analysis_results = {
            'strengths': [],
            'weaknesses': [],
            'detailed_weaknesses': [],  # New: detailed breakdown
            'suggestions': [],
            'score': 0
        }
        
        # Clean text
        clean_text = self.clean_text(resume_text)
        
        # Extract sections
        sections = self.extract_sections(clean_text)
        
        # Extract keywords
        resume_keywords, target_keywords = self.extract_keywords(clean_text, job_description)
        
        # Keyword analysis
        if target_keywords:
            matched_keywords = set(resume_keywords) & set(target_keywords)
            match_percentage = len(matched_keywords) / len(target_keywords) * 100
            
            if match_percentage < 70:
                missing_keywords = set(target_keywords) - set(resume_keywords)
                analysis_results['weaknesses'].append(f"Low keyword match ({match_percentage:.1f}%)")
                analysis_results['detailed_weaknesses'].append({
                    'category': 'Keywords',
                    'issue': f'Missing {len(missing_keywords)} important keywords',
                    'details': f'Your resume matches only {match_percentage:.1f}% of job requirements',
                    'missing_items': list(missing_keywords)[:15],
                    'found_items': list(matched_keywords)[:10],
                    'severity': 'High' if match_percentage < 50 else 'Medium'
                })
                analysis_results['suggestions'].append(
                    f"Add relevant keywords to your resume: {', '.join(list(missing_keywords)[:10])}"
                )
            else:
                analysis_results['strengths'].append(f"Good keyword match ({match_percentage:.1f}%)")
        
        # Employment gaps
        gaps = self.detect_employment_gaps(sections['experience'])
        if gaps:
            analysis_results['weaknesses'].extend(gaps)
            analysis_results['detailed_weaknesses'].append({
                'category': 'Employment History',
                'issue': f'Employment gaps detected ({len(gaps)} gap(s))',
                'details': 'Gaps in employment history may raise questions for recruiters',
                'gap_details': gaps,
                'severity': 'Medium',
                'recommendation': 'Explain activities during gaps (education, projects, volunteering)'
            })
            analysis_results['suggestions'].append(
                "Address employment gaps by explaining what you did during those periods (education, freelancing, volunteering, etc.)"
            )
        
        # Quantification check
        quantified, unquantified = self.check_quantification(sections['experience'])
        if len(unquantified) > len(quantified):
            analysis_results['weaknesses'].append("Lack of quantified achievements")
            analysis_results['detailed_weaknesses'].append({
                'category': 'Achievement Quantification',
                'issue': f'{len(unquantified)} unquantified statements found',
                'details': f'Only {len(quantified)} out of {len(quantified) + len(unquantified)} statements include metrics',
                'unquantified_examples': unquantified[:5],
                'quantified_examples': quantified[:3] if quantified else [],
                'severity': 'High' if len(quantified) == 0 else 'Medium',
                'improvement_examples': [
                    "'Managed team' → 'Managed team of 8 developers'",
                    "'Increased sales' → 'Increased sales by 25% over 6 months'",
                    "'Improved efficiency' → 'Reduced processing time by 40%'"
                ]
            })
            analysis_results['suggestions'].append(
                "Add numbers and metrics to your achievements. Example: 'Managed team' → 'Managed team of 5, increasing productivity by 20%'"
            )
        else:
            analysis_results['strengths'].append("Good use of quantified achievements")
        
        # Weak language detection
        weak_phrases = self.detect_weak_language(sections['experience'])
        if weak_phrases:
            analysis_results['weaknesses'].append("Weak action verbs detected")
            analysis_results['detailed_weaknesses'].append({
                'category': 'Language Strength',
                'issue': f'{len(weak_phrases)} weak phrases found',
                'details': 'Weak language reduces impact and shows passive involvement',
                'weak_examples': weak_phrases[:5],
                'strong_alternatives': self.strong_verbs[:8],
                'severity': 'Medium',
                'replacements': [
                    "'Responsible for managing' → 'Led and managed'",
                    "'Helped with projects' → 'Delivered 3 key projects'",
                    "'Worked on team' → 'Collaborated with cross-functional team'"
                ]
            })
            analysis_results['suggestions'].append(
                f"Replace weak phrases with strong action verbs. Use words like: {', '.join(self.strong_verbs[:5])}"
            )
        
        # Resume length check
        word_count = len(clean_text.split())
        if word_count < 200:
            analysis_results['weaknesses'].append("Resume too short")
            analysis_results['detailed_weaknesses'].append({
                'category': 'Resume Length',
                'issue': f'Resume is too short ({word_count} words)',
                'details': 'Short resumes may not provide enough information about your qualifications',
                'current_length': f'{word_count} words',
                'recommended_length': '300-600 words (1-2 pages)',
                'severity': 'Medium',
                'areas_to_expand': ['Work experience details', 'Achievement descriptions', 'Skills elaboration', 'Project outcomes']
            })
            analysis_results['suggestions'].append("Expand your resume with more details about your experience and achievements")
        elif word_count > 800:
            analysis_results['weaknesses'].append("Resume too long")
            analysis_results['detailed_weaknesses'].append({
                'category': 'Resume Length',
                'issue': f'Resume is too long ({word_count} words)',
                'details': 'Long resumes may lose recruiter attention and fail ATS parsing',
                'current_length': f'{word_count} words',
                'recommended_length': '300-600 words (1-2 pages)',
                'severity': 'Medium',
                'areas_to_condense': ['Remove outdated experience', 'Combine similar roles', 'Shorten bullet points', 'Remove irrelevant skills']
            })
            analysis_results['suggestions'].append("Condense your resume to 1-2 pages, focusing on most relevant experience")
        else:
            analysis_results['strengths'].append("Appropriate resume length")
        
        # Contact information check
        email_found = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', clean_text)
        phone_found = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', clean_text)
        
        missing_contact = []
        if not email_found:
            missing_contact.append('email address')
        if not phone_found:
            missing_contact.append('phone number')
            
        if missing_contact:
            analysis_results['weaknesses'].append(f"Missing contact information: {', '.join(missing_contact)}")
            analysis_results['detailed_weaknesses'].append({
                'category': 'Contact Information',
                'issue': f'Missing {len(missing_contact)} contact detail(s)',
                'details': 'Complete contact information is essential for recruiters to reach you',
                'missing_items': missing_contact,
                'severity': 'High',
                'required_items': ['Professional email', 'Phone number', 'LinkedIn profile (recommended)', 'City, State']
            })
            analysis_results['suggestions'].append("Include complete contact information: professional email, phone number, and LinkedIn profile")
        
        # Calculate overall score
        total_checks = 6  # Number of different checks we perform
        weaknesses_count = len([w for w in analysis_results['weaknesses'] if not w.startswith('Gap detected')])
        analysis_results['score'] = max(0, (total_checks - weaknesses_count) / total_checks * 100)
        
        return analysis_results

def main():
    st.set_page_config(
        page_title="Resume ATS Analyzer MVP",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Resume ATS Analyzer MVP")
    st.markdown("Upload your resume and get instant feedback on ATS compatibility and improvement suggestions.")
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    if analyzer.nlp is None:
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'txt'],
            help="Upload a PDF or text file (max 5MB)"
        )
        
        st.header("💼 Job Description (Optional)")
        job_description = st.text_area(
            "Paste the job description here to get targeted keyword analysis",
            height=150,
            help="If provided, the analyzer will check for job-specific keywords"
        )
    
    # Main content area
    if uploaded_file is not None:
        # File size check
        if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
            st.error("File size too large. Please upload a file smaller than 5MB.")
            return
        
        try:
            # Read file content
            file_content = uploaded_file.read()
            file_type = 'pdf' if uploaded_file.type == 'application/pdf' else 'txt'
            
            # Parse resume
            with st.spinner("Parsing resume..."):
                resume_text = analyzer.parse_resume(file_content, file_type)
            
            if not resume_text.strip():
                st.error("No text could be extracted from the file. Please check if the file is valid.")
                return
            
            # Analyze button
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing resume..."):
                    results = analyzer.analyze_resume(resume_text, job_description)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    # Score display
                    score = results['score']
                    if score >= 80:
                        score_color = "green"
                        score_emoji = "🟢"
                    elif score >= 60:
                        score_color = "orange"
                        score_emoji = "🟡"
                    else:
                        score_color = "red"
                        score_emoji = "🔴"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 2px solid {score_color}; border-radius: 10px;">
                        <h2>{score_emoji} Overall Score</h2>
                        <h1 style="color: {score_color};">{score:.0f}/100</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col1:
                    # Strengths
                    if results['strengths']:
                        st.success("✅ **Strengths Found**")
                        for strength in results['strengths']:
                            st.write(f"• {strength}")
                    
                    # Weaknesses with detailed breakdown
                    if results['weaknesses']:
                        st.error("⚠️ **Areas for Improvement**")
                        
                        # Show detailed weaknesses if available
                        if results.get('detailed_weaknesses'):
                            for detail in results['detailed_weaknesses']:
                                severity_color = {
                                    'High': '🔴',
                                    'Medium': '🟡', 
                                    'Low': '🟢'
                                }.get(detail['severity'], '⚪')
                                
                                with st.expander(f"{severity_color} {detail['category']}: {detail['issue']}", expanded=True):
                                    st.write(f"**Issue:** {detail['details']}")
                                    
                                    # Show specific examples based on category
                                    if detail['category'] == 'Keywords' and 'missing_items' in detail:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**❌ Missing Keywords:**")
                                            for keyword in detail['missing_items'][:10]:
                                                st.write(f"• {keyword}")
                                        with col2:
                                            if detail['found_items']:
                                                st.write("**✅ Found Keywords:**")
                                                for keyword in detail['found_items'][:8]:
                                                    st.write(f"• {keyword}")
                                    
                                    elif detail['category'] == 'Achievement Quantification':
                                        if detail.get('unquantified_examples'):
                                            st.write("**📝 Statements needing quantification:**")
                                            for example in detail['unquantified_examples']:
                                                st.write(f"• {example[:100]}...")
                                        
                                        if detail.get('improvement_examples'):
                                            st.write("**💡 How to improve:**")
                                            for example in detail['improvement_examples']:
                                                st.write(f"• {example}")
                                    
                                    elif detail['category'] == 'Language Strength':
                                        if detail.get('weak_examples'):
                                            st.write("**❌ Weak phrases found:**")
                                            for example in detail['weak_examples']:
                                                st.write(f"• {example[:80]}...")
                                        
                                        if detail.get('replacements'):
                                            st.write("**✅ Suggested improvements:**")
                                            for replacement in detail['replacements']:
                                                st.write(f"• {replacement}")
                                    
                                    elif detail['category'] == 'Employment History':
                                        if detail.get('gap_details'):
                                            st.write("**📅 Gaps detected:**")
                                            for gap in detail['gap_details']:
                                                st.write(f"• {gap}")
                                        if detail.get('recommendation'):
                                            st.info(f"💡 **Recommendation:** {detail['recommendation']}")
                                    
                                    elif detail['category'] == 'Resume Length':
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Current Length", detail.get('current_length', 'Unknown'))
                                        with col2:
                                            st.metric("Recommended", detail.get('recommended_length', '300-600 words'))
                                        
                                        if detail.get('areas_to_expand'):
                                            st.write("**📈 Areas to expand:**")
                                            for area in detail['areas_to_expand']:
                                                st.write(f"• {area}")
                                        elif detail.get('areas_to_condense'):
                                            st.write("**📉 Areas to condense:**")
                                            for area in detail['areas_to_condense']:
                                                st.write(f"• {area}")
                                    
                                    elif detail['category'] == 'Contact Information':
                                        if detail.get('missing_items'):
                                            st.write("**❌ Missing information:**")
                                            for item in detail['missing_items']:
                                                st.write(f"• {item.title()}")
                                        
                                        if detail.get('required_items'):
                                            st.write("**✅ Complete contact info should include:**")
                                            for item in detail['required_items']:
                                                st.write(f"• {item}")
                        else:
                            # Fallback to simple list if no detailed breakdown
                            for weakness in results['weaknesses']:
                                st.write(f"• {weakness}")
                    
                    # Suggestions
                    if results['suggestions']:
                        st.info("💡 **Improvement Suggestions**")
                        for i, suggestion in enumerate(results['suggestions'], 1):
                            st.write(f"{i}. {suggestion}")
                
                # Detailed breakdown
                with st.expander("📊 Detailed Analysis"):
                    st.subheader("Resume Sections Detected")
                    sections = analyzer.extract_sections(resume_text)
                    
                    for section_name, section_content in sections.items():
                        if section_content.strip():
                            st.write(f"**{section_name.title()}:** ✅")
                        else:
                            st.write(f"**{section_name.title()}:** ❌ Not detected")
                    
                    st.subheader("Resume Preview")
                    st.text_area("Extracted Text", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200)
        
        except Exception as e:
            st.error(f"An error occurred while processing your resume: {str(e)}")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload your resume using the sidebar to get started.")
        
        st.markdown("""
        ### How it works:
        1. **Upload** your resume (PDF or text format)
        2. **Optionally** paste a job description for targeted analysis
        3. **Click** "Analyze Resume" to get instant feedback
        4. **Review** strengths, weaknesses, and improvement suggestions
        
        ### What we analyze:
        - ✅ Keyword matching with job requirements
        - ✅ Employment gaps detection
        - ✅ Quantified achievements
        - ✅ Action verb strength
        - ✅ Resume length and formatting
        - ✅ Contact information completeness
        
        ### Future enhancements:
        - ML-based section parsing
        - Industry-specific keyword databases
        - ATS formatting score
        - Skills gap analysis
        - Resume template suggestions
        """)

if __name__ == "__main__":
    main()
