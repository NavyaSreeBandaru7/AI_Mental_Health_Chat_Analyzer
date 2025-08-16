import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Advanced ML imports
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, pipeline, BertTokenizer, BertForSequenceClassification
)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import joblib

# NLP and text processing
import spacy
from textstat import flesch_reading_ease, dale_chall_readability_score
import re
from collections import Counter

# Async processing
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

class AgentSpecialty(Enum):
    DEPRESSION_SPECIALIST = "depression_specialist"
    ANXIETY_SPECIALIST = "anxiety_specialist"
    CRISIS_SPECIALIST = "crisis_specialist"
    LINGUISTIC_ANALYST = "linguistic_analyst"
    BEHAVIORAL_ANALYST = "behavioral_analyst"
    RECOMMENDATION_ENGINE = "recommendation_engine"

@dataclass
class AgentAnalysis:
    agent_id: str
    specialty: AgentSpecialty
    confidence: float
    risk_score: float
    key_indicators: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    processing_time: float

class BaseAgent(ABC):
    """Abstract base class for all mental health agents"""
    
    def __init__(self, agent_id: str, specialty: AgentSpecialty):
        self.agent_id = agent_id
        self.specialty = specialty
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self):
        """Initialize agent models and resources"""
        pass
    
    @abstractmethod
    async def analyze(self, text: str, context: Dict[str, Any] = None) -> AgentAnalysis:
        """Perform specialized analysis on text"""
        pass
    
    def _extract_base_features(self, text: str) -> Dict[str, float]:
        """Extract common features used by multiple agents"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Readability scores
        features['flesch_score'] = flesch_reading_ease(text)
        features['dale_chall_score'] = dale_chall_readability_score(text)
        
        # Emotional indicators
        negative_words = ['sad', 'depressed', 'anxious', 'worried', 'hopeless', 
                         'worthless', 'empty', 'overwhelmed', 'stressed']
        features['negative_word_count'] = sum(1 for word in negative_words if word in text.lower())
        
        return features

class DepressionSpecialistAgent(BaseAgent):
    """Specialized agent for depression detection and analysis"""
    
    def __init__(self):
        super().__init__("depression_specialist_001", AgentSpecialty.DEPRESSION_SPECIALIST)
        self.depression_keywords = [
            'sad', 'sadness', 'depressed', 'depression', 'hopeless', 'hopelessness',
            'worthless', 'worthlessness', 'empty', 'emptiness', 'numb', 'numbness',
            'lonely', 'loneliness', 'isolated', 'isolation', 'meaningless',
            'tired', 'exhausted', 'fatigue', 'sleep', 'insomnia', 'appetite'
        ]
        
    async def initialize(self):
        """Initialize depression-specific models"""
        try:
            # Load specialized depression detection model
            self.tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "mental/mental-bert-base-uncased"
            )
            
            # Fallback to general sentiment model if specialized model unavailable
            if self.model is None:
                self.depression_classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            
            self.is_initialized = True
            logger.info(f"Depression specialist agent {self.agent_id} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize depression agent: {e}")
            self.depression_classifier = None
    
    async def analyze(self, text: str, context: Dict[str, Any] = None) -> AgentAnalysis:
        """Analyze text for depression indicators"""
        start_time = datetime.now()
        
        try:
            # Extract base features
            features = self._extract_base_features(text)
            
            # Depression-specific analysis
            risk_score = await self._calculate_depression_risk(text, features)
            confidence = await self._calculate_confidence(text, features, risk_score)
            key_indicators = self._identify_depression_indicators(text)
            recommendations = self._generate_depression_recommendations(risk_score)
            
            # Advanced linguistic analysis
            linguistic_features = await self._analyze_depression_linguistics(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentAnalysis(
                agent_id=self.agent_id,
                specialty=self.specialty,
                confidence=confidence,
                risk_score=risk_score,
                key_indicators=key_indicators,
                recommendations=recommendations,
                metadata={
                    'features': features,
                    'linguistic_features': linguistic_features,
                    'model_used': 'depression_specialist_v2.1'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Depression analysis failed: {e}")
            return self._create_fallback_analysis(text, start_time)
    
    async def _calculate_depression_risk(self, text: str, features: Dict[str, float]) -> float:
        """Calculate depression risk score using multiple indicators"""
        scores = []
        
        # Keyword-based scoring
        keyword_score = sum(1 for keyword in self.depression_keywords if keyword in text.lower())
        normalized_keyword_score = min(keyword_score / 10, 1.0)
        scores.append(normalized_keyword_score)
        
        # Sentiment analysis
        if self.depression_classifier:
            try:
                sentiment_result = self.depression_classifier(text)[0]
                if sentiment_result['label'] == 'NEGATIVE':
                    sentiment_score = sentiment_result['score']
                else:
                    sentiment_score = 1 - sentiment_result['score']
                scores.append(sentiment_score)
            except:
                scores.append(0.5)  # Neutral if failed
        
        # Linguistic patterns associated with depression
        linguistic_score = self._analyze_depression_patterns(text)
        scores.append(linguistic_score)
        
        # Feature-based scoring
        feature_score = self._calculate_feature_based_score(features)
        scores.append(feature_score)
        
        # Weighted average
        weights = [0.3, 0.3, 0.25, 0.15]
        final_score = np.average(scores, weights=weights)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _analyze_depression_patterns(self, text: str) -> float:
        """Analyze linguistic patterns associated with depression"""
        score = 0.0
        text_lower = text.lower()
        
        # Negative self-talk patterns
        negative_self_patterns = [
            r'\bi am (worthless|useless|failure|burden)',
            r'\bi (hate|despise) myself',
            r'\bi (can\'t|cannot) do anything',
            r'nothing (matters|works|helps)',
            r'(everything|life) is (pointless|meaningless)'
        ]
        
        for pattern in negative_self_patterns:
            if re.search(pattern, text_lower):
                score += 0.2
        
        # Hopelessness indicators
        hopelessness_patterns = [
            r'(no|never) (hope|point|reason)',
            r'(give|giving) up',
            r'(can\'t|cannot) go on',
            r'(things|life) will never (get better|improve)',
            r'no way (out|forward)',
            r'(stuck|trapped) forever'
        ]
        
        for pattern in hopelessness_patterns:
            if re.search(pattern, text_lower):
                score += 0.25
        
        # Sleep and energy patterns
        sleep_energy_patterns = [
            r'(can\'t|cannot) sleep',
            r'(tired|exhausted) all the time',
            r'no energy',
            r'(sleeping|sleep) too much'
        ]
        
        for pattern in sleep_energy_patterns:
            if re.search(pattern, text_lower):
                score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_feature_based_score(self, features: Dict[str, float]) -> float:
        """Calculate risk based on text features"""
        score = 0.0
        
        # Long texts might indicate rumination
        if features['word_count'] > 100:
            score += 0.1
        
        # Very short texts might indicate apathy
        if features['word_count'] < 10:
            score += 0.2
        
        # Low readability might indicate cognitive issues
        if features['flesch_score'] < 30:
            score += 0.15
        
        # High negative word density
        negative_density = features['negative_word_count'] / max(features['word_count'], 1)
        score += min(negative_density * 2, 0.5)
        
        return min(score, 1.0)
    
    async def _calculate_confidence(self, text: str, features: Dict[str, float], risk_score: float) -> float:
        """Calculate confidence in the depression assessment"""
        confidence_factors = []
        
        # Text length factor
        text_length_factor = min(features['word_count'] / 50, 1.0)
        confidence_factors.append(text_length_factor)
        
        # Keyword presence factor
        keyword_presence = min(features['negative_word_count'] / 5, 1.0)
        confidence_factors.append(keyword_presence)
        
        # Model agreement factor (if multiple models used)
        model_agreement = 0.8  # Simulated
        confidence_factors.append(model_agreement)
        
        # Risk score extremity (more confident in extreme scores)
        risk_extremity = abs(risk_score - 0.5) * 2
        confidence_factors.append(risk_extremity)
        
        return np.mean(confidence_factors)
    
    def _identify_depression_indicators(self, text: str) -> List[str]:
        """Identify specific depression indicators in text"""
        indicators = []
        text_lower = text.lower()
        
        indicator_patterns = {
            'Hopelessness': [r'no hope', r'hopeless', r'pointless', r'meaningless'],
            'Worthlessness': [r'worthless', r'useless', r'failure', r'burden'],
            'Sleep Issues': [r'can\'t sleep', r'insomnia', r'sleeping too much'],
            'Energy Loss': [r'no energy', r'exhausted', r'tired all the time'],
            'Isolation': [r'alone', r'lonely', r'isolated', r'no friends'],
            'Suicidal Ideation': [r'want to die', r'end it all', r'suicide', r'kill myself']
        }
        
        for indicator, patterns in indicator_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    indicators.append(indicator)
                    break
        
        return indicators
    
    def _generate_depression_recommendations(self, risk_score: float) -> List[str]:
        """Generate depression-specific recommendations"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "URGENT: Please contact a mental health professional immediately",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Consider going to the nearest emergency room"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Strongly recommend speaking with a therapist or counselor",
                "Contact your primary care physician",
                "Consider cognitive behavioral therapy (CBT)",
                "Reach out to trusted friends or family members"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Consider speaking with a mental health professional",
                "Try mindfulness meditation or yoga",
                "Maintain regular exercise routine",
                "Ensure adequate sleep (7-9 hours per night)"
            ])
        else:
            recommendations.extend([
                "Continue healthy lifestyle habits",
                "Stay connected with friends and family",
                "Practice stress management techniques",
                "Monitor mood changes"
            ])
        
        return recommendations
    
    async def _analyze_depression_linguistics(self, text: str) -> Dict[str, float]:
        """Advanced linguistic analysis for depression markers"""
        features = {}
        
        # First-person pronoun usage (associated with depression)
        first_person_pronouns = len(re.findall(r'\b(i|me|my|myself)\b', text.lower()))
        features['first_person_ratio'] = first_person_pronouns / max(len(text.split()), 1)
        
        # Past tense usage (rumination indicator)
        past_tense_verbs = len(re.findall(r'\b\w+ed\b', text.lower()))
        features['past_tense_ratio'] = past_tense_verbs / max(len(text.split()), 1)
        
        # Absolute words (black-and-white thinking)
        absolute_words = ['always', 'never', 'everything', 'nothing', 'all', 'none']
        absolute_count = sum(1 for word in absolute_words if word in text.lower())
        features['absolute_words_ratio'] = absolute_count / max(len(text.split()), 1)
        
        return features
    
    def _create_fallback_analysis(self, text: str, start_time: datetime) -> AgentAnalysis:
        """Create fallback analysis when main analysis fails"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentAnalysis(
            agent_id=self.agent_id,
            specialty=self.specialty,
            confidence=0.3,
            risk_score=0.5,
            key_indicators=['Analysis Failed'],
            recommendations=['Please retry analysis or contact support'],
            metadata={'error': 'Fallback analysis used'},
            processing_time=processing_time
        )

class AnxietySpecialistAgent(BaseAgent):
    """Specialized agent for anxiety detection and analysis"""
    
    def __init__(self):
        super().__init__("anxiety_specialist_001", AgentSpecialty.ANXIETY_SPECIALIST)
        self.anxiety_keywords = [
            'worried', 'worry', 'anxious', 'anxiety', 'panic', 'fear', 'scared',
            'nervous', 'overwhelmed', 'stressed', 'tension', 'restless',
            'racing thoughts', 'heart pounding', 'sweating', 'trembling'
        ]
    
    async def initialize(self):
        """Initialize anxiety-specific models"""
        try:
            self.anxiety_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            self.is_initialized = True
            logger.info(f"Anxiety specialist agent {self.agent_id} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize anxiety agent: {e}")
            self.anxiety_classifier = None
    
    async def analyze(self, text: str, context: Dict[str, Any] = None) -> AgentAnalysis:
        """Analyze text for anxiety indicators"""
        start_time = datetime.now()
        
        try:
            features = self._extract_base_features(text)
            risk_score = await self._calculate_anxiety_risk(text, features)
            confidence = await self._calculate_anxiety_confidence(text, features, risk_score)
            key_indicators = self._identify_anxiety_indicators(text)
            recommendations = self._generate_anxiety_recommendations(risk_score)
            
            # Anxiety-specific features
            anxiety_features = self._extract_anxiety_features(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentAnalysis(
                agent_id=self.agent_id,
                specialty=self.specialty,
                confidence=confidence,
                risk_score=risk_score,
                key_indicators=key_indicators,
                recommendations=recommendations,
                metadata={
                    'features': features,
                    'anxiety_features': anxiety_features,
                    'model_used': 'anxiety_specialist_v1.8'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Anxiety analysis failed: {e}")
            return self._create_fallback_analysis(text, start_time)
    
    async def _calculate_anxiety_risk(self, text: str, features: Dict[str, float]) -> float:
        """Calculate anxiety risk score"""
        scores = []
        
        # Keyword-based scoring
        keyword_score = sum(1 for keyword in self.anxiety_keywords if keyword in text.lower())
        normalized_keyword_score = min(keyword_score / 8, 1.0)
        scores.append(normalized_keyword_score)
        
        # Physical symptom patterns
        physical_symptoms = self._detect_physical_anxiety_symptoms(text)
        scores.append(physical_symptoms)
        
        # Cognitive patterns
        cognitive_patterns = self._detect_anxiety_cognitive_patterns(text)
        scores.append(cognitive_patterns)
        
        # Urgency and catastrophizing
        urgency_score = self._detect_urgency_patterns(text)
        scores.append(urgency_score)
        
        # Weighted average
        weights = [0.25, 0.25, 0.3, 0.2]
        final_score = np.average(scores, weights=weights)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _detect_physical_anxiety_symptoms(self, text: str) -> float:
        """Detect physical symptoms of anxiety"""
        physical_symptoms = [
            'heart racing', 'heart pounding', 'sweating', 'trembling', 'shaking',
            'shortness of breath', 'chest pain', 'nausea', 'dizziness',
            'hot flashes', 'cold sweats', 'muscle tension'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for symptom in physical_symptoms:
            if symptom in text_lower:
                score += 0.15
        
        return min(score, 1.0)
    
    def _detect_anxiety_cognitive_patterns(self, text: str) -> float:
        """Detect cognitive patterns associated with anxiety"""
        score = 0.0
        text_lower = text.lower()
        
        # Catastrophizing patterns
        catastrophizing = [
            r'what if', r'worst case', r'something terrible',
            r'going to happen', r'can\'t stop thinking'
        ]
        
        for pattern in catastrophizing:
            if re.search(pattern, text_lower):
                score += 0.2
        
        # Worry patterns
        worry_patterns = [
            r'worried about', r'can\'t stop worrying',
            r'mind racing', r'overthinking'
        ]
        
        for pattern in worry_patterns:
            if re.search(pattern, text_lower):
                score += 0.25
        
        return min(score, 1.0)
    
    def _detect_urgency_patterns(self, text: str) -> float:
        """Detect urgency and panic-related patterns"""
        urgency_indicators = [
            '!!!', 'urgent', 'emergency', 'panic', 'right now',
            'immediately', 'can\'t wait', 'need help now'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        # Count exclamation marks
        exclamation_count = text.count('!')
        score += min(exclamation_count * 0.1, 0.3)
        
        # Check urgency words
        for indicator in urgency_indicators:
            if indicator in text_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _extract_anxiety_features(self, text: str) -> Dict[str, float]:
        """Extract anxiety-specific linguistic features"""
        features = {}
        
        # Question marks (uncertainty/worry)
        features['question_ratio'] = text.count('?') / max(len(text.split()), 1)
        
        # Future tense (anticipatory anxiety)
        future_words = ['will', 'going to', 'might', 'could', 'may']
        future_count = sum(1 for word in future_words if word in text.lower())
        features['future_tense_ratio'] = future_count / max(len(text.split()), 1)
        
        # Uncertainty words
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'uncertain', 'unsure']
        uncertainty_count = sum(1 for word in uncertainty_words if word in text.lower())
        features['uncertainty_ratio'] = uncertainty_count / max(len(text.split()), 1)
        
        return features
    
    def _identify_anxiety_indicators(self, text: str) -> List[str]:
        """Identify specific anxiety indicators"""
        indicators = []
        text_lower = text.lower()
        
        indicator_patterns = {
            'Physical Symptoms': [r'heart (racing|pounding)', r'sweating', r'trembling'],
            'Catastrophizing': [r'what if', r'worst case', r'something terrible'],
            'Worry Thoughts': [r'worried', r'can\'t stop thinking', r'mind racing'],
            'Panic': [r'panic', r'panic attack', r'can\'t breathe'],
            'Avoidance': [r'avoiding', r'can\'t face', r'too scared to']
        }
        
        for indicator, patterns in indicator_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    indicators.append(indicator)
                    break
        
        return indicators
    
    def _generate_anxiety_recommendations(self, risk_score: float) -> List[str]:
        """Generate anxiety-specific recommendations"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "Consider immediate professional help for severe anxiety",
                "Practice emergency grounding techniques (5-4-3-2-1 method)",
                "Crisis Text Line: Text HOME to 741741",
                "Contact your healthcare provider"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Consider therapy, especially Cognitive Behavioral Therapy (CBT)",
                "Practice deep breathing exercises",
                "Try progressive muscle relaxation",
                "Limit caffeine intake"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Practice mindfulness meditation",
                "Regular exercise can help reduce anxiety",
                "Maintain a consistent sleep schedule",
                "Consider keeping an anxiety journal"
            ])
        else:
            recommendations.extend([
                "Continue stress management practices",
                "Maintain healthy lifestyle habits",
                "Practice relaxation techniques",
                "Stay connected with support network"
            ])
        
        return recommendations
    
    async def _calculate_anxiety_confidence(self, text: str, features: Dict[str, float], risk_score: float) -> float:
        """Calculate confidence in anxiety assessment"""
        confidence_factors = []
        
        # Text adequacy
        text_adequacy = min(features['word_count'] / 30, 1.0)
        confidence_factors.append(text_adequacy)
        
        # Anxiety keyword density
        keyword_density = min(features['negative_word_count'] / 3, 1.0)
        confidence_factors.append(keyword_density)
        
        # Pattern consistency
        pattern_consistency = 0.85  # Simulated
        confidence_factors.append(pattern_consistency)
        
        return np.mean(confidence_factors)

class CrisisSpecialistAgent(BaseAgent):
    """Specialized agent for crisis detection and immediate intervention"""
    
    def __init__(self):
        super().__init__("crisis_specialist_001", AgentSpecialty.CRISIS_SPECIALIST)
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'want to die', 'better off dead',
            'hurt myself', 'self harm', 'cut myself', 'overdose', 'jump off',
            'can\'t go on', 'no point living', 'goodbye forever'
        ]
        
    async def initialize(self):
        """Initialize crisis detection models"""
        try:
            # Load suicide risk detection model
            self.crisis_classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model"  # Proxy for harmful content
            )
            self.is_initialized = True
            logger.info(f"Crisis specialist agent {self.agent_id} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize crisis agent: {e}")
            self.crisis_classifier = None
    
    async def analyze(self, text: str, context: Dict[str, Any] = None) -> AgentAnalysis:
        """Analyze text for crisis indicators requiring immediate intervention"""
        start_time = datetime.now()
        
        try:
            features = self._extract_base_features(text)
            risk_score = await self._calculate_crisis_risk(text, features)
            confidence = await self._calculate_crisis_confidence(text, risk_score)
            key_indicators = self._identify_crisis_indicators(text)
            recommendations = self._generate_crisis_recommendations(risk_score)
            
            # Crisis-specific analysis
            crisis_features = self._extract_crisis_features(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log high-risk cases immediately
            if risk_score >= 0.7:
                logger.critical(f"HIGH CRISIS RISK DETECTED: Score {risk_score:.3f} - {text[:100]}...")
            
            return AgentAnalysis(
                agent_id=self.agent_id,
                specialty=self.specialty,
                confidence=confidence,
                risk_score=risk_score,
                key_indicators=key_indicators,
                recommendations=recommendations,
                metadata={
                    'features': features,
                    'crisis_features': crisis_features,
                    'requires_immediate_attention': risk_score >= 0.7,
                    'model_used': 'crisis_detection_v3.2'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Crisis analysis failed: {e}")
            return self._create_fallback_analysis(text, start_time)
    
    async def _calculate_crisis_risk(self, text: str, features: Dict[str, float]) -> float:
        """Calculate crisis risk with maximum sensitivity"""
        scores = []
        
        # Direct suicidal ideation
        direct_indicators = self._detect_direct_suicidal_ideation(text)
        scores.append(direct_indicators)
        
        # Indirect indicators
        indirect_indicators = self._detect_indirect_crisis_signs(text)
        scores.append(indirect_indicators)
        
        # Plan and means assessment
        plan_means = self._assess_plan_and_means(text)
        scores.append(plan_means)
        
        # Hopelessness and finality
        hopelessness = self._assess_hopelessness_finality(text)
        scores.append(hopelessness)
        
        # Take maximum score for crisis (err on side of caution)
        final_score = max(scores)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _detect_direct_suicidal_ideation(self, text: str) -> float:
        """Detect direct expressions of suicidal thoughts"""
        direct_indicators = [
            r'want to (die|kill myself)',
            r'going to (kill myself|end it)',
            r'suicide|suicidal',
            r'better off dead',
            r'don\'t want to (live|be here)',
            r'end (it all|my life)'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for pattern in direct_indicators:
            if re.search(pattern, text_lower):
                score = 1.0  # Maximum score for direct ideation
                break
        
        return score
    
    def _detect_indirect_crisis_signs(self, text: str) -> float:
        """Detect indirect signs of crisis"""
        indirect_signs = [
            r'can\'t (take|handle) it anymore',
            r'too much to (bear|handle)',
            r'no way out',
            r'trapped|stuck forever',
            r'goodbye (everyone|world)',
            r'won\'t be around'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for pattern in indirect_signs:
            if re.search(pattern, text_lower):
                score += 0.3
        
        return min(score, 0.8)  # Cap at 0.8 for indirect signs
    
    def _assess_plan_and_means(self, text: str) -> float:
        """Assess presence of plan and means"""
        plan_indicators = [
            r'(pills|rope|gun|bridge|knife)',
            r'figured out how',
            r'have a plan',
            r'know how to',
            r'found a way'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for pattern in plan_indicators:
            if re.search(pattern, text_lower):
                score = 0.9  # Very high score for plan/means
                break
        
        return score
    
    def _assess_hopelessness_finality(self, text: str) -> float:
        """Assess level of hopelessness and finality"""
        hopelessness_patterns = [
            r'no hope',
            r'never (get better|improve)',
            r'always be this way',
            r'final (decision|goodbye)',
            r'made up my mind'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for pattern in hopelessness_patterns:
            if re.search(pattern, text_lower):
                score += 0.2
        
        return min(score, 0.7)
    
    def _extract_crisis_features(self, text: str) -> Dict[str, float]:
        """Extract crisis-specific features"""
        features = {}
        
        # Finality words
        finality_words = ['final', 'last', 'goodbye', 'forever', 'never again']
        finality_count = sum(1 for word in finality_words if word in text.lower())
        features['finality_ratio'] = finality_count / max(len(text.split()), 1)
        
        # Plan specificity
        specific_methods = ['pills', 'rope', 'gun', 'jump', 'overdose', 'cut']
        method_count = sum(1 for method in specific_methods if method in text.lower())
        features['method_specificity'] = method_count / max(len(text.split()), 1)
        
        # Time urgency
        time_urgency = ['tonight', 'today', 'now', 'soon', 'this week']
        urgency_count = sum(1 for word in time_urgency if word in text.lower())
        features['time_urgency'] = urgency_count / max(len(text.split()), 1)
        
        return features
    
    def _identify_crisis_indicators(self, text: str) -> List[str]:
        """Identify specific crisis indicators"""
        indicators = []
        text_lower = text.lower()
        
        crisis_patterns = {
            'Suicidal Ideation': [r'suicide', r'kill myself', r'want to die'],
            'Self-Harm Intent': [r'hurt myself', r'cut myself', r'self harm'],
            'Plan/Method': [r'pills', r'rope', r'gun', r'jump off'],
            'Finality': [r'goodbye', r'final', r'last time', r'forever'],
            'Hopelessness': [r'no hope', r'no point', r'can\'t go on'],
            'Isolation': [r'no one cares', r'alone', r'nobody understands']
        }
        
        for indicator, patterns in crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    indicators.append(indicator)
                    break
        
        return indicators
    
    def _generate_crisis_recommendations(self, risk_score: float) -> List[str]:
        """Generate crisis-specific immediate intervention recommendations"""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.extend([
                "🚨 IMMEDIATE CRISIS INTERVENTION REQUIRED 🚨",
                "Call 911 if in immediate danger",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Go to nearest emergency room immediately",
                "Do not leave person alone if possible",
                "Remove any potential means of self-harm"
            ])
        elif risk_score >= 0.5:
            recommendations.extend([
                "High risk detected - urgent professional help needed",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Contact mental health professional immediately",
                "Reach out to trusted friend or family member",
                "Create safety plan with professional"
            ])
        elif risk_score >= 0.3:
            recommendations.extend([
                "Moderate risk - professional assessment recommended",
                "Contact mental health professional",
                "National Suicide Prevention Lifeline: 988",
                "Develop coping strategies with therapist",
                "Maintain close contact with support system"
            ])
        else:
            recommendations.extend([
                "Continue monitoring mental health",
                "Maintain connection with support network",
                "Know crisis resources: 988 for emergencies",
                "Practice self-care and stress management"
            ])
        
        return recommendations
    
    async def _calculate_crisis_confidence(self, text: str, risk_score: float) -> float:
        """Calculate confidence in crisis assessment"""
        # For crisis detection, we err on the side of caution
        # Higher sensitivity, accept some false positives
        
        base_confidence = 0.8  # Start with high confidence
        
        # Adjust based on text length
        if len(text.split()) < 5:
            base_confidence *= 0.7  # Lower confidence for very short texts
        
        # Adjust based on risk score
        if risk_score >= 0.8:
            base_confidence = 0.95  # Very confident in high-risk cases
        elif risk_score <= 0.2:
            base_confidence = 0.6   # Lower confidence in low-risk cases
        
        return base_confidence

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents for comprehensive analysis"""
    
    def __init__(self):
        self.agents = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize_agents(self):
        """Initialize all specialized agents"""
        agent_classes = [
            DepressionSpecialistAgent,
            AnxietySpecialistAgent,
            CrisisSpecialistAgent
        ]
        
        for agent_class in agent_classes:
            try:
                agent = agent_class()
                await agent.initialize()
                self.agents[agent.specialty] = agent
                logger.info(f"Initialized agent: {agent.specialty.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {agent_class.__name__}: {e}")
    
    async def analyze_comprehensive(self, text: str, context: Dict[str, Any] = None) -> Dict[str, AgentAnalysis]:
        """Run comprehensive analysis using all agents"""
        if not self.agents:
            await self.initialize_agents()
        
        # Run all agents concurrently
        tasks = []
        for specialty, agent in self.agents.items():
            task = asyncio.create_task(agent.analyze(text, context))
            tasks.append((specialty, task))
        
        results = {}
        for specialty, task in tasks:
            try:
                result = await task
                results[specialty] = result
            except Exception as e:
                logger.error(f"Agent {specialty.value} failed: {e}")
        
        return results
    
    def synthesize_analysis(self, agent_results: Dict[AgentSpecialty, AgentAnalysis]) -> Dict[str, Any]:
        """Synthesize results from multiple agents into final assessment"""
        synthesis = {
            'overall_risk_score': 0.0,
            'confidence': 0.0,
            'primary_concerns': [],
            'all_recommendations': [],
            'requires_immediate_attention': False,
            'agent_consensus': {},
            'processing_metrics': {}
        }
        
        if not agent_results:
            return synthesis
        
        # Calculate overall risk (weighted by specialty importance)
        specialty_weights = {
            AgentSpecialty.CRISIS_SPECIALIST: 0.4,
            AgentSpecialty.DEPRESSION_SPECIALIST: 0.3,
            AgentSpecialty.ANXIETY_SPECIALIST: 0.3
        }
        
        weighted_scores = []
        total_weight = 0
        
        for specialty, analysis in agent_results.items():
            weight = specialty_weights.get(specialty, 0.2)
            weighted_scores.append(analysis.risk_score * weight)
            total_weight += weight
            
            # Collect recommendations
            synthesis['all_recommendations'].extend(analysis.recommendations)
            
            # Check for immediate attention requirements
            if (specialty == AgentSpecialty.CRISIS_SPECIALIST and 
                analysis.risk_score >= 0.7):
                synthesis['requires_immediate_attention'] = True
            
            # Collect key indicators as primary concerns
            synthesis['primary_concerns'].extend(analysis.key_indicators)
        
        # Calculate overall scores
        if total_weight > 0:
            synthesis['overall_risk_score'] = sum(weighted_scores) / total_weight
        
        # Calculate consensus confidence
        confidences = [analysis.confidence for analysis in agent_results.values()]
        synthesis['confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Remove duplicate recommendations and concerns
        synthesis['all_recommendations'] = list(set(synthesis['all_recommendations']))
        synthesis['primary_concerns'] = list(set(synthesis['primary_concerns']))
        
        # Agent consensus metrics
        risk_scores = [analysis.risk_score for analysis in agent_results.values()]
        synthesis['agent_consensus'] = {
            'risk_score_std': np.std(risk_scores) if len(risk_scores) > 1 else 0.0,
            'risk_score_range': max(risk_scores) - min(risk_scores) if risk_scores else 0.0,
            'agreement_level': 'high' if np.std(risk_scores) < 0.2 else 'moderate' if np.std(risk_scores) < 0.4 else 'low'
        }
        
        # Processing metrics
        processing_times = [analysis.processing_time for analysis in agent_results.values()]
        synthesis['processing_metrics'] = {
            'total_processing_time': sum(processing_times),
            'avg_processing_time': np.mean(processing_times),
            'agents_used': len(agent_results)
        }
        
        return synthesis

# Example usage and testing functions
async def test_multi_agent_system():
    """Test the multi-agent system with sample texts"""
    orchestrator = MultiAgentOrchestrator()
    await orchestrator.initialize_agents()
    
    test_cases = [
        {
            'text': "I feel really sad and hopeless, nothing seems to matter anymore",
            'expected_primary': 'depression'
        },
        {
            'text': "I'm so anxious about everything, my heart is racing and I can't stop worrying",
            'expected_primary': 'anxiety'
        },
        {
            'text': "I want to end it all, I have pills and I'm going to take them tonight",
            'expected_primary': 'crisis'
        },
        {
            'text': "Having a good day today, feeling optimistic about the future",
            'expected_primary': 'none'
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: {case['text']}")
        
        agent_results = await orchestrator.analyze_comprehensive(case['text'])
        synthesis = orchestrator.synthesize_analysis(agent_results)
        
        print(f"Overall Risk Score: {synthesis['overall_risk_score']:.3f}")
        print(f"Confidence: {synthesis['confidence']:.3f}")
        print(f"Immediate Attention Required: {synthesis['requires_immediate_attention']}")
        print(f"Primary Concerns: {synthesis['primary_concerns']}")
        print(f"Agent Agreement: {synthesis['agent_consensus']['agreement_level']}")
        
        if synthesis['requires_immediate_attention']:
            print("🚨 CRISIS INTERVENTION REQUIRED 🚨")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_multi_agent_system())
            '
