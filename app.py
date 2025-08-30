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
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data without checking if it exists first
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# punkt_tab is included in punkt for newer versions

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
        
        # Resume validation patterns
        self.resume_indicators = {
            'section_headers': [
                'experience', 'work experience', 'professional experience', 'employment', 'work history',
                'education', 'academic background', 'qualifications', 'degrees', 'academic',
                'skills', 'technical skills', 'core competencies', 'abilities', 'expertise',
                'summary', 'profile', 'objective', 'about me', 'career summary', 'overview',
                'contact', 'contact information', 'personal details', 'personal information',
                'projects', 'achievements', 'accomplishments', 'awards', 'honors',
                'certifications', 'licenses', 'training', 'courses', 'professional development',
                'languages', 'interests', 'hobbies', 'volunteer', 'references'
            ],
            'contact_patterns': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
                r'\b(?:linkedin\.com|github\.com|portfolio)\b',  # Professional links
            ],
            'date_patterns': [
                r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
                r'\b\d{1,2}/\d{4}\b',
                r'\b\d{4}\s*-\s*\d{4}\b',
                r'\b\d{4}\s*-\s*present\b',
                r'\bpresent\b'
            ],
            'action_verbs': [
                'managed', 'led', 'developed', 'created', 'implemented', 'designed',
                'achieved', 'improved', 'increased', 'reduced', 'optimized',
                'coordinated', 'supervised', 'executed', 'delivered', 'built',
                'organized', 'planned', 'analyzed', 'collaborated', 'maintained',
                'established', 'initiated', 'facilitated', 'supported', 'assisted',
                'worked', 'performed', 'completed', 'handled', 'processed'
            ],
            'degree_indicators': [
                'bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma',
                'certificate', 'certification', 'university', 'college', 'school',
                'b.s.', 'm.s.', 'b.a.', 'm.a.', 'mba', 'gpa', 'graduate',
                'undergraduate', 'major', 'minor', 'honors', 'cum laude',
                'institute', 'academy', 'high school', 'secondary'
            ]
        }
        
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

    def validate_resume(self, text):
        """Validate if the uploaded document is actually a resume"""
        text_lower = text.lower()
        validation_score = 0
        validation_details = {
            'is_resume': False,
            'confidence': 0,
            'found_indicators': [],
            'missing_indicators': [],
            'suggestions': []
        }
        
        # Check for section headers (30 points max)
        section_score = 0
        found_sections = []
        for section in self.resume_indicators['section_headers']:
            if section in text_lower:
                section_score += 1.5
                found_sections.append(section)
                if section_score >= 30:
                    break
        validation_score += min(section_score, 30)
        
        # Check for contact information (20 points max)
        contact_score = 0
        found_contacts = []
        for pattern in self.resume_indicators['contact_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                contact_score += 7
                found_contacts.append("contact info")
        validation_score += min(contact_score, 20)
        
        # Check for date patterns (15 points max)
        date_score = 0
        found_dates = []
        for pattern in self.resume_indicators['date_patterns']:
            matches = re.findall(pattern, text_lower)
            if matches:
                date_score += len(matches) * 2
                found_dates.extend(matches[:3])  # Limit to first 3
        validation_score += min(date_score, 15)
        
        # Check for action verbs (20 points max)
        action_score = 0
        found_actions = []
        for verb in self.resume_indicators['action_verbs']:
            if verb in text_lower:
                action_score += 0.8
                found_actions.append(verb)
        validation_score += min(action_score, 20)
        
        # Check for education indicators (15 points max)
        education_score = 0
        found_education = []
        for indicator in self.resume_indicators['degree_indicators']:
            if indicator in text_lower:
                education_score += 1
                found_education.append(indicator)
        validation_score += min(education_score, 15)
        
        # Calculate confidence percentage
        confidence = min(validation_score, 100)
        
        # Add bonus points for common resume patterns
        bonus_score = 0
        
        # Check for name patterns (likely at the top)
        lines = text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            if len(line.split()) == 2 and line.replace(' ', '').isalpha() and len(line) > 5:
                bonus_score += 5
                break
        
        # Check for job titles
        job_titles = ['manager', 'engineer', 'developer', 'analyst', 'coordinator', 
                     'specialist', 'director', 'assistant', 'associate', 'consultant',
                     'administrator', 'supervisor', 'lead', 'senior', 'junior']
        for title in job_titles:
            if title in text_lower:
                bonus_score += 2
                break
        
        # Check for company indicators
        company_words = ['company', 'corporation', 'inc', 'llc', 'ltd', 'group', 'solutions']
        for word in company_words:
            if word in text_lower:
                bonus_score += 2
                break
        
        validation_score += min(bonus_score, 15)
        confidence = min(validation_score, 100)
        
        # Lower threshold for resume detection
        is_resume = confidence >= 45
        
        # Populate validation details
        validation_details['is_resume'] = is_resume
        validation_details['confidence'] = confidence
        validation_details['found_indicators'] = {
            'sections': found_sections[:5],
            'contact': found_contacts,
            'dates': found_dates[:3],
            'action_verbs': found_actions[:5],
            'education': found_education[:3]
        }
        
        # Generate suggestions for improvement
        if confidence < 45:
            suggestions = []
            if section_score < 15:
                suggestions.append("Add clear section headers like 'Experience', 'Education', 'Skills'")
            if contact_score < 10:
                suggestions.append("Include complete contact information (email, phone number)")
            if date_score < 6:
                suggestions.append("Add employment dates and education timeline")
            if action_score < 8:
                suggestions.append("Use strong action verbs to describe your experience")
            if education_score < 5:
                suggestions.append("Include education background and qualifications")
            
            validation_details['suggestions'] = suggestions
        
        return validation_details

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
        page_title="Resume ATS Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for Material Design styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .score-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .score-number {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .score-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Success/Error/Info cards */
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(79, 172, 254, 0.2);
    }
    
    .error-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(250, 112, 154, 0.2);
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(168, 237, 234, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Upload area styling */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed #e2e8f0;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
    }
    
    /* Severity indicators */
    .severity-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .severity-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .severity-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 Resume ATS Analyzer</h1>
        <p class="main-subtitle">AI-powered resume optimization for modern job seekers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    if analyzer.nlp is None:
        st.stop()
    
    # Initialize session state for wizard steps
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Progress indicator
    progress_steps = ["📤 Upload Resume", "💼 Job Details", "🔍 Analysis", "📊 Results"]
    
    # Create progress bar
    col1, col2, col3, col4 = st.columns(4)
    for i, (col, step_name) in enumerate(zip([col1, col2, col3, col4], progress_steps), 1):
        with col:
            if i < st.session_state.step:
                # Completed step
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                           color: white; border-radius: 8px; margin-bottom: 1rem; font-weight: 600;">
                    ✅ {step_name}
                </div>
                """, unsafe_allow_html=True)
            elif i == st.session_state.step:
                # Current step
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; border-radius: 8px; margin-bottom: 1rem; font-weight: 600;">
                    {step_name}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Future step
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f1f5f9; 
                           color: #64748b; border-radius: 8px; margin-bottom: 1rem; font-weight: 500;">
                    {step_name}
                </div>
                """, unsafe_allow_html=True)
    
    # Step-based content
    if st.session_state.step == 1:
        # Step 1: Upload Resume
        col1, col2, col3 = st.columns([1, 2, 1])
    
        with col2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center;">
                <h2 style="color: #2d3748; margin-top: 0; font-weight: 600; margin-bottom: 1rem;">📤 Upload Your Resume</h2>
                <p style="color: #718096; font-size: 1rem; margin-bottom: 2rem;">Drag and drop your resume or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced file uploader with drag-and-drop styling
            uploaded_file = st.file_uploader(
                "Choose your resume file",
                type=['pdf', 'txt'],
                help="Upload a PDF or text file (max 5MB)"
            )
            
            # Real-time file validation and preview
            if uploaded_file is not None:
                # File validation
                file_size_mb = uploaded_file.size / (1024 * 1024)
                file_type = uploaded_file.type
                
                # Show file info
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("File Size", f"{file_size_mb:.1f} MB")
                with col_info2:
                    st.metric("File Type", "PDF" if file_type == "application/pdf" else "Text")
                
                # File size check
                if file_size_mb > 5:
                    st.error("❌ File size too large. Please upload a file smaller than 5MB.")
                    st.stop()
                
                # Parse and preview resume
                try:
                    with st.spinner("📄 Extracting text from your resume..."):
                        file_content = uploaded_file.read()
                        file_type_str = 'pdf' if uploaded_file.type == 'application/pdf' else 'txt'
                        resume_text = analyzer.parse_resume(file_content, file_type_str)
                        st.session_state.resume_text = resume_text
                    
                    if resume_text.strip():
                        # Validate if it's actually a resume
                        with st.spinner("🔍 Validating resume content..."):
                            validation = analyzer.validate_resume(resume_text)
                        
                        # Display validation results
                        if validation['is_resume']:
                            st.success(f"✅ Resume validated! Confidence: {validation['confidence']:.0f}%")
                            
                            # Show what was found
                            if validation['found_indicators']:
                                with st.expander("🔍 Validation Details", expanded=False):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if validation['found_indicators']['sections']:
                                            st.write("**📋 Sections Found:**")
                                            for section in validation['found_indicators']['sections']:
                                                st.write(f"• {section.title()}")
                                        
                                        if validation['found_indicators']['action_verbs']:
                                            st.write("**💪 Action Verbs Found:**")
                                            for verb in validation['found_indicators']['action_verbs']:
                                                st.write(f"• {verb}")
                                    
                                    with col2:
                                        if validation['found_indicators']['contact']:
                                            st.write("**📞 Contact Info:** ✅")
                                        
                                        if validation['found_indicators']['dates']:
                                            st.write("**📅 Dates Found:**")
                                            for date in validation['found_indicators']['dates']:
                                                st.write(f"• {date}")
                                        
                                        if validation['found_indicators']['education']:
                                            st.write("**🎓 Education Keywords:**")
                                            for edu in validation['found_indicators']['education']:
                                                st.write(f"• {edu}")
                            
                            # Show preview
                            with st.expander("📖 Resume Preview", expanded=False):
                                preview_text = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
                                st.text_area("Extracted Text Preview", preview_text, height=150, disabled=True)
                                st.info(f"Total words extracted: {len(resume_text.split())}")
                            
                            # Next step button
                            if st.button("➡️ Continue to Job Details", type="primary", use_container_width=True):
                                st.session_state.step = 2
                                st.rerun()
                        
                        else:
                            # Not a valid resume
                            st.error(f"⚠️ This doesn't appear to be a resume (Confidence: {validation['confidence']:.0f}%)")
                            
                            st.markdown("""
                            <div style="background: #fef2f2; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ef4444; margin: 1rem 0;">
                                <h4 style="color: #dc2626; margin-top: 0;">❌ Document Validation Failed</h4>
                                <p style="color: #7f1d1d; margin-bottom: 0;">The uploaded file doesn't contain typical resume elements. Please upload a proper resume document.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show what's missing and suggestions
                            if validation['suggestions']:
                                st.write("**💡 To make this a valid resume, consider adding:**")
                                for suggestion in validation['suggestions']:
                                    st.write(f"• {suggestion}")
                            
                            # Show what was found (if any)
                            if any(validation['found_indicators'].values()):
                                with st.expander("🔍 What We Found", expanded=False):
                                    found_something = False
                                    for category, items in validation['found_indicators'].items():
                                        if items:
                                            found_something = True
                                            st.write(f"**{category.title()}:** {', '.join(map(str, items[:3]))}")
                                    
                                    if not found_something:
                                        st.write("No typical resume elements were detected.")
                            
                            # Option to continue anyway
                            st.warning("⚠️ You can still continue, but the analysis may not be accurate for non-resume documents.")
                            
                            col_force1, col_force2 = st.columns(2)
                            with col_force1:
                                if st.button("🔄 Upload Different File", use_container_width=True):
                                    st.session_state.resume_text = None
                                    st.rerun()
                            
                            with col_force2:
                                if st.button("➡️ Continue Anyway", type="secondary", use_container_width=True):
                                    st.session_state.step = 2
                                    st.rerun()
                    
                    else:
                        st.error("❌ No text could be extracted from the file. Please check if the file is valid.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            
            else:
                # Show feature cards when no file uploaded
                st.markdown("### ✨ What We Analyze")
                
                feature_col1, feature_col2 = st.columns(2)
                with feature_col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                        <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-size: 0.9rem;">Keywords</h4>
                        <p style="color: #718096; font-size: 0.8rem;">Job matching</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with feature_col2:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                        <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-size: 0.9rem;">Achievements</h4>
                        <p style="color: #718096; font-size: 0.8rem;">Quantification</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                feature_col3, feature_col4 = st.columns(2)
                with feature_col3:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💪</div>
                        <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-size: 0.9rem;">Language</h4>
                        <p style="color: #718096; font-size: 0.8rem;">Action verbs</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with feature_col4:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📏</div>
                        <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-size: 0.9rem;">Format</h4>
                        <p style="color: #718096; font-size: 0.8rem;">ATS friendly</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif st.session_state.step == 2:
        # Step 2: Job Description
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center;">
                <h2 style="color: #2d3748; margin-top: 0; font-weight: 600; margin-bottom: 1rem;">💼 Job Description</h2>
                <p style="color: #718096; font-size: 1rem; margin-bottom: 2rem;">Add job description for targeted keyword analysis (optional)</p>
            </div>
            """, unsafe_allow_html=True)
            
            job_description = st.text_area(
                "Paste the job description here",
                height=200,
                help="If provided, the analyzer will check for job-specific keywords",
                placeholder="Paste the job description here to get targeted keyword analysis..."
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⬅️ Back to Upload", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
            
            with col_btn2:
                if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
                    st.session_state.job_description = job_description
                    st.session_state.step = 3
                    st.rerun()
    
    elif st.session_state.step == 3:
        # Step 3: Analysis
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center;">
                <h2 style="color: #2d3748; margin-top: 0; font-weight: 600; margin-bottom: 1rem;">🔍 Analyzing Your Resume</h2>
                <p style="color: #718096; font-size: 1rem; margin-bottom: 2rem;">Our AI is analyzing your resume for ATS compatibility...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate analysis steps with progress
            import time
            
            steps = [
                ("📄 Parsing resume content...", 20),
                ("🔍 Extracting keywords...", 40),
                ("📊 Analyzing achievements...", 60),
                ("💬 Checking language strength...", 80),
                ("📋 Generating recommendations...", 100)
            ]
            
            for step_text, progress in steps:
                status_text.text(step_text)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            # Perform actual analysis
            job_desc = getattr(st.session_state, 'job_description', '')
            results = analyzer.analyze_resume(st.session_state.resume_text, job_desc)
            st.session_state.analysis_results = results
            
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            st.session_state.step = 4
            st.rerun()
    
    elif st.session_state.step == 4:
        # Step 4: Results with tabs
        results = st.session_state.analysis_results
        
        # Results header
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center;">
            <h2 style="color: #2d3748; margin-top: 0; font-weight: 600; margin-bottom: 1rem;">📊 Analysis Results</h2>
            <p style="color: #718096; font-size: 1rem; margin-bottom: 0;">Your resume has been analyzed for ATS compatibility</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Score display
        score = results['score']
        if score >= 80:
            score_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
            score_emoji = "🎉"
            score_text = "Excellent"
        elif score >= 60:
            score_gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
            score_emoji = "👍"
            score_text = "Good"
        else:
            score_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
            score_emoji = "⚠️"
            score_text = "Needs Work"
        
        st.markdown(f"""
        <div class="score-card" style="background: {score_gradient}; margin-bottom: 2rem;">
            <div class="score-label">{score_emoji} ATS Compatibility Score</div>
            <div class="score-number">{score:.0f}</div>
            <div class="score-label">{score_text} • Out of 100</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabbed interface for results
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "⚠️ Issues", "💡 Suggestions", "📊 Details"])
        
        with tab1:
            # Overview tab
            col1, col2 = st.columns(2)
            
            with col1:
                if results['strengths']:
                    st.markdown("""
                    <div class="success-card">
                        <h3 style="margin-top: 0; font-weight: 600;">✅ Strengths Found</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for strength in results['strengths']:
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <span style="color: #065f46; font-weight: 500;">• {strength}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if results['weaknesses']:
                    st.markdown("""
                    <div class="error-card">
                        <h3 style="margin-top: 0; font-weight: 600;">⚠️ Areas to Improve</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for weakness in results['weaknesses'][:3]:  # Show top 3
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #ef4444; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <span style="color: #dc2626; font-weight: 500;">• {weakness}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            # Issues tab - detailed weaknesses
            if results.get('detailed_weaknesses'):
                for detail in results['detailed_weaknesses']:
                    severity_badge = {
                        'High': '<span class="severity-high">🔴 HIGH PRIORITY</span>',
                        'Medium': '<span class="severity-medium">🟡 MEDIUM</span>', 
                        'Low': '<span class="severity-low">🟢 LOW</span>'
                    }.get(detail['severity'], '<span class="severity-medium">⚪ UNKNOWN</span>')
                    
                    with st.expander(f"{detail['category']}: {detail['issue']}", expanded=True):
                        st.markdown(severity_badge, unsafe_allow_html=True)
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
        
        with tab3:
            # Suggestions tab
            if results['suggestions']:
                st.markdown("""
                <div class="info-card">
                    <h3 style="margin-top: 0; font-weight: 600; color: #2d3748;">💡 Action Plan</h3>
                    <p style="margin-bottom: 0; color: #4a5568;">Priority improvements to boost your ATS score</p>
                </div>
                """, unsafe_allow_html=True)
                
                for i, suggestion in enumerate(results['suggestions'], 1):
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #3182ce; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <span style="color: #2b6cb0; font-weight: 500;">{i}. {suggestion}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab4:
            # Details tab
            st.subheader("Resume Sections Detected")
            sections = analyzer.extract_sections(st.session_state.resume_text)
            
            for section_name, section_content in sections.items():
                if section_content.strip():
                    st.write(f"**{section_name.title()}:** ✅")
                else:
                    st.write(f"**{section_name.title()}:** ❌ Not detected")
            
            st.subheader("Resume Preview")
            st.text_area("Extracted Text", st.session_state.resume_text[:1000] + "..." if len(st.session_state.resume_text) > 1000 else st.session_state.resume_text, height=200)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Analyze Another Resume", use_container_width=True):
                # Reset session state
                st.session_state.step = 1
                st.session_state.resume_text = None
                st.session_state.analysis_results = None
                st.rerun()
        
        with col2:
            if st.button("📥 Download Report", use_container_width=True):
                st.info("📥 Download feature coming soon!")
        
        with col3:
            if st.button("📤 Share Results", use_container_width=True):
                st.info("📤 Share feature coming soon!")
    
    # Developer credit footer
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; margin-top: 3rem; text-align: center; color: white;">
        <h3 style="margin-top: 0; font-weight: 600; color: white;">👨‍💻 Built by Abdel-Rahaman Rabee</h3>
        <p style="margin-bottom: 1rem; opacity: 0.9; font-size: 1.1rem;">Software Engineer</p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">📧</span>
                <a href="mailto:abdelrahamanrabee1134@gmail.com" style="color: white; text-decoration: none; font-weight: 500;">abdelrahamanrabee1134@gmail.com</a>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">🐙</span>
                <a href="https://github.com/abdorabee" target="_blank" style="color: white; text-decoration: none; font-weight: 500;">github.com/abdorabee</a>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">
                Trust the process
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
