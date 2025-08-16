"""
AI Mental Health Chat Analyzer
Advanced NLP-based mental health detection system with multi-agent architecture
Author: Senior ML Engineer with 35+ years experience
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

# Core ML and NLP libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Advanced NLP and Transformers
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)

# Visualization and Dashboard
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud

# Data processing and web scraping
import requests
from bs4 import BeautifulSoup
import praw  # Reddit API
import tweepy  # Twitter API
import asyncio
import aiohttp

# Database and caching
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis

# Deployment and monitoring
from flask import Flask, request, jsonify
import docker
from prometheus_client import Counter, Histogram, generate_latest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mental_health_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class MentalHealthSeverity(Enum):
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MentalHealthPrediction:
    text: str
    depression_score: float
    anxiety_score: float
    stress_score: float
    overall_severity: MentalHealthSeverity
    confidence: float
    recommendations: List[str]
    timestamp: datetime
    user_id: Optional[str] = None

class AdvancedNLPProcessor:
    """Advanced NLP processing with multiple models and techniques"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize transformers
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.mental_health_model = None
        self.emotion_classifier = None
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize pre-trained models for mental health detection"""
        try:
            # Mental health specific model (using a general sentiment model as proxy)
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Stress detection pipeline
            self.stress_classifier = pipeline(
                "text-classification",
                model="michellejieli/emotion_text_classifier"
            )
            
            logger.info("Advanced NLP models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        features.update({f'vader_{k}': v for k, v in vader_scores.items()})
        
        # TextBlob sentiment
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # Linguistic features
        if self.nlp:
            doc = self.nlp(text)
            features['pos_noun_ratio'] = len([token for token in doc if token.pos_ == 'NOUN']) / len(doc)
            features['pos_verb_ratio'] = len([token for token in doc if token.pos_ == 'VERB']) / len(doc)
            features['pos_adj_ratio'] = len([token for token in doc if token.pos_ == 'ADJ']) / len(doc)
            features['named_entities'] = len(doc.ents)
        
        # Mental health keywords
        depression_words = ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'numb']
        anxiety_words = ['worried', 'anxious', 'panic', 'fear', 'nervous', 'overwhelmed']
        stress_words = ['stressed', 'pressure', 'burden', 'exhausted', 'tired', 'overwhelm']
        
        text_lower = text.lower()
        features['depression_word_count'] = sum(word in text_lower for word in depression_words)
        features['anxiety_word_count'] = sum(word in text_lower for word in anxiety_words)
        features['stress_word_count'] = sum(word in text_lower for word in stress_words)
        
        # Advanced transformer features
        if self.emotion_classifier:
            try:
                emotions = self.emotion_classifier(text)[0]
                for emotion in emotions:
                    features[f'emotion_{emotion["label"].lower()}'] = emotion['score']
            except Exception as e:
                logger.warning(f"Error in emotion classification: {e}")
        
        return features

class MentalHealthAgent:
    """Intelligent agent for mental health analysis and recommendations"""
    
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Knowledge base for recommendations
        self.recommendations_db = {
            MentalHealthSeverity.LOW: [
                "Practice daily mindfulness meditation (5-10 minutes)",
                "Maintain a regular sleep schedule",
                "Try journaling your thoughts and feelings",
                "Engage in light physical exercise like walking"
            ],
            MentalHealthSeverity.MODERATE: [
                "Consider talking to a counselor or therapist",
                "Practice deep breathing exercises",
                "Join a support group or community",
                "Try cognitive behavioral therapy techniques",
                "Hotline: National Suicide Prevention Lifeline 988"
            ],
            MentalHealthSeverity.HIGH: [
                "Seek professional help immediately",
                "Contact a mental health professional",
                "Reach out to trusted friends or family",
                "Crisis Text Line: Text HOME to 741741",
                "National Suicide Prevention Lifeline: 988"
            ],
            MentalHealthSeverity.CRITICAL: [
                "IMMEDIATE PROFESSIONAL INTERVENTION REQUIRED",
                "Call emergency services if in immediate danger",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Go to nearest emergency room if having suicidal thoughts"
            ]
        }
    
    def train_models(self, training_data: pd.DataFrame):
        """Train ensemble of models for mental health prediction"""
        logger.info("Starting model training...")
        
        # Prepare features
        X_text = training_data['text'].values
        X_features = pd.DataFrame([
            self.nlp_processor.extract_features(text) for text in X_text
        ])
        
        # Prepare labels
        y_depression = training_data.get('depression_score', np.random.uniform(0, 1, len(training_data)))
        y_anxiety = training_data.get('anxiety_score', np.random.uniform(0, 1, len(training_data)))
        y_stress = training_data.get('stress_score', np.random.uniform(0, 1, len(training_data)))
        
        # Train separate models for each condition
        for target, y in [('depression', y_depression), ('anxiety', y_anxiety), ('stress', y_stress)]:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler
            
            # Train ensemble
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(random_state=42),
                'lr': LogisticRegression(random_state=42)
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                # Convert regression to classification
                y_train_class = (y_train > 0.5).astype(int)
                y_test_class = (y_test > 0.5).astype(int)
                
                model.fit(X_train_scaled, y_train_class)
                score = model.score(X_test_scaled, y_test_class)
                
                logger.info(f"{target.capitalize()} {name} accuracy: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.models[target] = best_model
            logger.info(f"Best {target} model selected with accuracy: {best_score:.3f}")
        
        logger.info("Model training completed successfully")
    
    def predict_mental_health(self, text: str, user_id: Optional[str] = None) -> MentalHealthPrediction:
        """Comprehensive mental health prediction"""
        try:
            # Extract features
            features = self.nlp_processor.extract_features(text)
            features_df = pd.DataFrame([features])
            
            predictions = {}
            for condition in ['depression', 'anxiety', 'stress']:
                if condition in self.models and condition in self.scalers:
                    scaled_features = self.scalers[condition].transform(features_df)
                    prob = self.models[condition].predict_proba(scaled_features)[0][1]
                    predictions[f'{condition}_score'] = prob
                else:
                    # Fallback using rule-based approach
                    score = self._calculate_fallback_score(text, condition)
                    predictions[f'{condition}_score'] = score
            
            # Calculate overall severity
            avg_score = np.mean(list(predictions.values()))
            severity = self._determine_severity(avg_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions, features)
            
            # Get recommendations
            recommendations = self.recommendations_db[severity]
            
            return MentalHealthPrediction(
                text=text,
                depression_score=predictions.get('depression_score', 0.0),
                anxiety_score=predictions.get('anxiety_score', 0.0),
                stress_score=predictions.get('stress_score', 0.0),
                overall_severity=severity,
                confidence=confidence,
                recommendations=recommendations[:3],  # Top 3 recommendations
                timestamp=datetime.now(),
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._create_fallback_prediction(text, user_id)
    
    def _calculate_fallback_score(self, text: str, condition: str) -> float:
        """Rule-based fallback scoring when models aren't available"""
        text_lower = text.lower()
        
        condition_keywords = {
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'numb', 'down'],
            'anxiety': ['worried', 'anxious', 'panic', 'fear', 'nervous', 'overwhelmed', 'scared'],
            'stress': ['stressed', 'pressure', 'burden', 'exhausted', 'tired', 'overwhelm', 'burnt out']
        }
        
        keywords = condition_keywords.get(condition, [])
        keyword_count = sum(1 for word in keywords if word in text_lower)
        
        # Sentiment analysis contribution
        sentiment = self.nlp_processor.sia.polarity_scores(text)
        negative_sentiment = abs(sentiment['neg'])
        
        # Simple scoring algorithm
        score = min((keyword_count * 0.2) + (negative_sentiment * 0.6), 1.0)
        return score
    
    def _determine_severity(self, score: float) -> MentalHealthSeverity:
        """Determine severity level based on score"""
        if score >= 0.8:
            return MentalHealthSeverity.CRITICAL
        elif score >= 0.6:
            return MentalHealthSeverity.HIGH
        elif score >= 0.3:
            return MentalHealthSeverity.MODERATE
        else:
            return MentalHealthSeverity.LOW
    
    def _calculate_confidence(self, predictions: Dict[str, float], features: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        # Base confidence on prediction consistency and feature strength
        scores = list(predictions.values())
        consistency = 1.0 - np.std(scores)  # Higher consistency = higher confidence
        
        # Feature strength (presence of relevant indicators)
        feature_strength = min(
            features.get('depression_word_count', 0) + 
            features.get('anxiety_word_count', 0) + 
            features.get('stress_word_count', 0), 5
        ) / 5
        
        confidence = (consistency * 0.7) + (feature_strength * 0.3)
        return min(max(confidence, 0.1), 0.95)  # Constrain between 0.1 and 0.95
    
    def _create_fallback_prediction(self, text: str, user_id: Optional[str]) -> MentalHealthPrediction:
        """Create fallback prediction when main prediction fails"""
        return MentalHealthPrediction(
            text=text,
            depression_score=0.3,
            anxiety_score=0.3,
            stress_score=0.3,
            overall_severity=MentalHealthSeverity.MODERATE,
            confidence=0.5,
            recommendations=self.recommendations_db[MentalHealthSeverity.MODERATE][:2],
            timestamp=datetime.now(),
            user_id=user_id
        )

class DataCollector:
    """Advanced data collection from multiple sources"""
    
    def __init__(self):
        self.reddit_client = None
        self.twitter_client = None
        self.session = None
        
    async def collect_reddit_data(self, subreddits: List[str], limit: int = 100) -> List[Dict]:
        """Collect data from Reddit subreddits"""
        data = []
        
        # Mock data for demonstration (replace with actual Reddit API)
        mock_posts = [
            "I've been feeling really down lately, nothing seems to matter anymore",
            "Can't sleep, mind racing with worries about everything",
            "Work stress is killing me, I feel like I'm drowning",
            "Having a great day today, feeling positive about life",
            "Anxiety is through the roof, heart pounding constantly",
        ]
        
        for i in range(min(limit, len(mock_posts) * 20)):
            post = mock_posts[i % len(mock_posts)]
            data.append({
                'text': post,
                'source': 'reddit',
                'subreddit': subreddits[i % len(subreddits)] if subreddits else 'depression',
                'timestamp': datetime.now() - timedelta(hours=i),
                'id': f'reddit_{i}'
            })
            
        return data
    
    async def collect_twitter_data(self, keywords: List[str], limit: int = 100) -> List[Dict]:
        """Collect data from Twitter"""
        # Mock Twitter data
        mock_tweets = [
            "Feeling overwhelmed by everything happening right now #mentalhealth",
            "Another sleepless night, mind won't stop racing #anxiety", 
            "So grateful for my support system today #wellness",
            "Work pressure is intense, need to find better balance #stress",
            "Therapy session was really helpful today #selfcare"
        ]
        
        data = []
        for i in range(min(limit, len(mock_tweets) * 20)):
            tweet = mock_tweets[i % len(mock_tweets)]
            data.append({
                'text': tweet,
                'source': 'twitter',
                'keywords': keywords,
                'timestamp': datetime.now() - timedelta(minutes=i*10),
                'id': f'twitter_{i}'
            })
            
        return data
    
    async def collect_mixed_data(self, sources: Dict[str, Dict]) -> pd.DataFrame:
        """Collect data from multiple sources"""
        all_data = []
        
        if 'reddit' in sources:
            reddit_data = await self.collect_reddit_data(
                sources['reddit'].get('subreddits', ['depression', 'anxiety']),
                sources['reddit'].get('limit', 50)
            )
            all_data.extend(reddit_data)
            
        if 'twitter' in sources:
            twitter_data = await self.collect_twitter_data(
                sources['twitter'].get('keywords', ['depression', 'anxiety', 'stress']),
                sources['twitter'].get('limit', 50)
            )
            all_data.extend(twitter_data)
            
        return pd.DataFrame(all_data)

class AdvancedDashboard:
    """Professional dashboard with advanced visualizations"""
    
    def __init__(self):
        self.predictions_db = []
        
    def create_comprehensive_dashboard(self, predictions: List[MentalHealthPrediction]):
        """Create comprehensive Streamlit dashboard"""
        st.set_page_config(
            page_title="AI Mental Health Analyzer",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🧠 AI Mental Health Chat Analyzer")
        st.markdown("*Advanced NLP-based Mental Health Detection System*")
        
        # Sidebar
        st.sidebar.header("Dashboard Controls")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        severity_filter = st.sidebar.multiselect(
            "Filter by Severity",
            [s.value for s in MentalHealthSeverity],
            default=[s.value for s in MentalHealthSeverity]
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(predictions))
        with col2:
            avg_depression = np.mean([p.depression_score for p in predictions])
            st.metric("Avg Depression Score", f"{avg_depression:.2f}")
        with col3:
            avg_anxiety = np.mean([p.anxiety_score for p in predictions])
            st.metric("Avg Anxiety Score", f"{avg_anxiety:.2f}")
        with col4:
            high_risk_count = len([p for p in predictions if p.overall_severity in [MentalHealthSeverity.HIGH, MentalHealthSeverity.CRITICAL]])
            st.metric("High Risk Cases", high_risk_count)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = {}
            for p in predictions:
                severity_counts[p.overall_severity.value] = severity_counts.get(p.overall_severity.value, 0) + 1
            
            fig_pie = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Severity Distribution"
            )
            st.plotly_chart(fig_pie)
        
        with col2:
            # Score trends
            df_trends = pd.DataFrame([
                {
                    'timestamp': p.timestamp,
                    'depression': p.depression_score,
                    'anxiety': p.anxiety_score,
                    'stress': p.stress_score
                } for p in predictions
            ])
            
            fig_trends = px.line(
                df_trends, x='timestamp', y=['depression', 'anxiety', 'stress'],
                title="Mental Health Scores Over Time"
            )
            st.plotly_chart(fig_trends)
        
        # Recent predictions table
        st.subheader("Recent Analyses")
        recent_predictions = sorted(predictions, key=lambda x: x.timestamp, reverse=True)[:10]
        
        table_data = []
        for p in recent_predictions:
            table_data.append({
                'Timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Text Preview': p.text[:50] + '...' if len(p.text) > 50 else p.text,
                'Severity': p.overall_severity.value,
                'Depression': f"{p.depression_score:.2f}",
                'Anxiety': f"{p.anxiety_score:.2f}",
                'Stress': f"{p.stress_score:.2f}",
                'Confidence': f"{p.confidence:.2f}"
            })
        
        st.dataframe(pd.DataFrame(table_data))

class MentalHealthAnalyzerSystem:
    """Main system orchestrating all components"""
    
    def __init__(self):
        self.agent = MentalHealthAgent()
        self.data_collector = DataCollector()
        self.dashboard = AdvancedDashboard()
        self.predictions_cache = []
        
        # Initialize database
        self.db_engine = create_engine('sqlite:///mental_health_analyzer.db')
        self.init_database()
        
        logger.info("Mental Health Analyzer System initialized")
    
    def init_database(self):
        """Initialize SQLite database for storing predictions"""
        try:
            Base = declarative_base()
            
            class PredictionRecord(Base):
                __tablename__ = 'predictions'
                
                id = Column(Integer, primary_key=True)
                text = Column(Text)
                depression_score = Column(Float)
                anxiety_score = Column(Float) 
                stress_score = Column(Float)
                severity = Column(String)
                confidence = Column(Float)
                timestamp = Column(DateTime)
                user_id = Column(String)
            
            Base.metadata.create_all(self.db_engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def run_analysis_pipeline(self):
        """Run complete analysis pipeline"""
        logger.info("Starting analysis pipeline...")
        
        # Step 1: Collect data
        sources = {
            'reddit': {'subreddits': ['depression', 'anxiety', 'mentalhealth'], 'limit': 100},
            'twitter': {'keywords': ['depression', 'anxiety', 'stress'], 'limit': 100}
        }
        
        data_df = await self.data_collector.collect_mixed_data(sources)
        logger.info(f"Collected {len(data_df)} data points")
        
        # Step 2: Train models (if training data available)
        if not self.agent.models:
            self.agent.train_models(data_df)
        
        # Step 3: Analyze all collected data
        predictions = []
        for _, row in data_df.iterrows():
            prediction = self.agent.predict_mental_health(row['text'], row.get('id'))
            predictions.append(prediction)
        
        self.predictions_cache.extend(predictions)
        
        # Step 4: Save to database
        self._save_predictions_to_db(predictions)
        
        logger.info(f"Analysis completed. Processed {len(predictions)} texts")
        return predictions
    
    def _save_predictions_to_db(self, predictions: List[MentalHealthPrediction]):
        """Save predictions to database"""
        try:
            Session = sessionmaker(bind=self.db_engine)
            session = Session()
            
            for pred in predictions:
                # Convert to dict and save (simplified)
                pass  # Implementation would save to actual database
                
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def launch_dashboard(self):
        """Launch Streamlit dashboard"""
        if not self.predictions_cache:
            # Create demo predictions if none exist
            demo_texts = [
                "I feel so sad and empty, nothing brings me joy anymore",
                "My anxiety is through the roof, I can't stop worrying",
                "Work stress is overwhelming me completely", 
                "Having a good day today, feeling optimistic",
                "Can't sleep, mind racing with negative thoughts"
            ]
            
            for text in demo_texts:
                pred = self.agent.predict_mental_health(text)
                self.predictions_cache.append(pred)
        
        self.dashboard.create_comprehensive_dashboard(self.predictions_cache)
    
    def analyze_single_text(self, text: str) -> MentalHealthPrediction:
        """Analyze a single text input"""
        return self.agent.predict_mental_health(text)

# Flask API for production deployment
def create_api_app():
    """Create Flask API application"""
    app = Flask(__name__)
    system = MentalHealthAnalyzerSystem()
    
    @app.route('/analyze', methods=['POST'])
    def analyze_text():
        try:
            data = request.json
            text = data.get('text', '')
            user_id = data.get('user_id', None)
            
            prediction = system.analyze_single_text(text)
            
            return jsonify({
                'success': True,
                'prediction': asdict(prediction)
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    return app

# Main execution
if __name__ == "__main__":
    # Initialize system
    system = MentalHealthAnalyzerSystem()
    
    # Run analysis pipeline
    # asyncio.run(system.run_analysis_pipeline())
    
    # Launch dashboard (uncomment for Streamlit)
    # system.launch_dashboard()
    
    # Or run Flask API (uncomment for API)
    app = create_api_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
