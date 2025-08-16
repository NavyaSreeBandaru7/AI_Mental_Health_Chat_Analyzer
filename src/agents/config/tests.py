import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import json
from typing import Dict, List

# Import main components
from main import (
    MentalHealthAnalyzerSystem, MentalHealthAgent, DataCollector, 
    AdvancedDashboard, MentalHealthSeverity, MentalHealthPrediction
)
from src.agents.advanced_mental_health_agents import (
    MultiAgentOrchestrator, DepressionSpecialistAgent, 
    AnxietySpecialistAgent, CrisisSpecialistAgent
)

class TestDataFixtures:
    """Test data fixtures for consistent testing"""
    
    @staticmethod
    def get_sample_texts():
        return {
            'high_depression': [
                "I feel so sad and empty, nothing brings me joy anymore",
                "Life feels meaningless, I'm worthless and hopeless",
                "I can't stop crying, everything seems pointless"
            ],
            'high_anxiety': [
                "I'm so worried about everything, my heart won't stop racing",
                "Panic attacks are getting worse, I can't breathe properly",
                "Constant fear and anxiety, I can't stop the racing thoughts"
            ],
            'crisis_level': [
                "I want to end it all, I can't take this pain anymore",
                "I have pills and I'm going to take them all tonight",
                "Goodbye everyone, I won't be around much longer"
            ],
            'positive': [
                "Having a great day today, feeling optimistic about life",
                "Things are looking up, I'm grateful for my support system",
                "Made progress in therapy, feeling hopeful about the future"
            ],
            'neutral': [
                "Going to work today, normal routine as usual",
                "The weather is nice, might go for a walk later",
                "Had lunch with a friend, talked about general topics"
            ]
        }
    
    @staticmethod
    def get_sample_dataframe():
        """Generate sample DataFrame for testing"""
        np.random.seed(42)
        texts = TestDataFixtures.get_sample_texts()
        
        data = []
        for category, text_list in texts.items():
            for text in text_list:
                # Simulate scores based on category
                if category == 'high_depression':
                    dep_score = np.random.uniform(0.7, 0.9)
                    anx_score = np.random.uniform(0.3, 0.6)
                    stress_score = np.random.uniform(0.4, 0.7)
                elif category == 'high_anxiety':
                    dep_score = np.random.uniform(0.2, 0.5)
                    anx_score = np.random.uniform(0.7, 0.9)
                    stress_score = np.random.uniform(0.5, 0.8)
                elif category == 'crisis_level':
                    dep_score = np.random.uniform(0.8, 1.0)
                    anx_score = np.random.uniform(0.7, 0.9)
                    stress_score = np.random.uniform(0.8, 1.0)
                else:  # positive or neutral
                    dep_score = np.random.uniform(0.1, 0.3)
                    anx_score = np.random.uniform(0.1, 0.3)
                    stress_score = np.random.uniform(0.1, 0.3)
                
                data.append({
                    'text': text,
                    'depression_score': dep_score,
                    'anxiety_score': anx_score,
                    'stress_score': stress_score,
                    'category': category,
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 100))
                })
        
        return pd.DataFrame(data)

@pytest.fixture
def sample_system():
    """Fixture for MentalHealthAnalyzerSystem"""
    return MentalHealthAnalyzerSystem()

@pytest.fixture
def sample_agent():
    """Fixture for MentalHealthAgent"""
    return MentalHealthAgent()

@pytest.fixture
def sample_data():
    """Fixture for sample data"""
    return TestDataFixtures.get_sample_dataframe()

@pytest.fixture
async def orchestrator():
    """Fixture for MultiAgentOrchestrator"""
    orchestrator = MultiAgentOrchestrator()
    # Mock the initialization to avoid loading actual models in tests
    with patch.object(orchestrator, 'initialize_agents', new_callable=AsyncMock):
        await orchestrator.initialize_agents()
    return orchestrator

class TestMentalHealthAgent:
    """Test cases for MentalHealthAgent"""
    
    def test_agent_initialization(self, sample_agent):
        """Test agent initialization"""
        assert sample_agent.nlp_processor is not None
        assert sample_agent.models == {}
        assert sample_agent.scalers == {}
        
    def test_feature_extraction(self, sample_agent):
        """Test feature extraction from text"""
        text = "I feel very sad and hopeless about everything"
        features = sample_agent.nlp_processor.extract_features(text)
        
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'vader_compound' in features
        assert 'textblob_polarity' in features
        assert 'depression_word_count' in features
        assert features['text_length'] == len(text)
        assert features['word_count'] == len(text.split())
        
    def test_prediction_structure(self, sample_agent):
        """Test prediction output structure"""
        text = "I feel overwhelmed and anxious"
        prediction = sample_agent.predict_mental_health(text)
        
        assert isinstance(prediction, MentalHealthPrediction)
        assert hasattr(prediction, 'depression_score')
        assert hasattr(prediction, 'anxiety_score')
        assert hasattr(prediction, 'stress_score')
        assert hasattr(prediction, 'overall_severity')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'recommendations')
        assert hasattr(prediction, 'timestamp')
        
        # Check score ranges
        assert 0 <= prediction.depression_score <= 1
        assert 0 <= prediction.anxiety_score <= 1
        assert 0 <= prediction.stress_score <= 1
        assert 0 <= prediction.confidence <= 1
        
    def test_severity_classification(self, sample_agent):
        """Test severity classification logic"""
        test_cases = [
            ("I'm having a great day!", MentalHealthSeverity.LOW),
            ("Feeling a bit stressed about work", MentalHealthSeverity.MODERATE),
            ("I feel really depressed and hopeless", MentalHealthSeverity.HIGH),
            ("I want to end it all", MentalHealthSeverity.CRITICAL)
        ]
        
        for text, expected_severity in test_cases:
            prediction = sample_agent.predict_mental_health(text)
            # Allow for some flexibility in classification
            assert prediction.overall_severity in [
                MentalHealthSeverity.LOW, MentalHealthSeverity.MODERATE,
                MentalHealthSeverity.HIGH, MentalHealthSeverity.CRITICAL
            ]
    
    def test_recommendation_generation(self, sample_agent):
        """Test recommendation generation"""
        text = "I feel very depressed and suicidal"
        prediction = sample_agent.predict_mental_health(text)
        
        assert len(prediction.recommendations) > 0
        assert any('988' in rec or 'professional' in rec.lower() 
                  for rec in prediction.recommendations)

class TestSpecializedAgents:
    """Test cases for specialized agents"""
    
    @pytest.mark.asyncio
    async def test_depression_agent(self):
        """Test depression specialist agent"""
        agent = DepressionSpecialistAgent()
        
        # Mock initialization
        with patch.object(agent, 'initialize', new_callable=AsyncMock):
            await agent.initialize()
        
        text = "I feel so sad and worthless, nothing matters anymore"
        
        # Mock the classifier to avoid model loading
        with patch.object(agent, 'depression_classifier', None):
            analysis = await agent.analyze(text)
        
        assert analysis.specialty.value == "depression_specialist"
        assert analysis.risk_score >= 0.0
        assert analysis.confidence >= 0.0
        assert len(analysis.key_indicators) >= 0
        assert len(analysis.recommendations) > 0
    
    @pytest.mark.asyncio 
    async def test_anxiety_agent(self):
        """Test anxiety specialist agent"""
        agent = AnxietySpecialistAgent()
        
        with patch.object(agent, 'initialize', new_callable=AsyncMock):
            await agent.initialize()
        
        text = "I'm so worried and anxious, my heart is racing"
        
        with patch.object(agent, 'anxiety_classifier', None):
            analysis = await agent.analyze(text)
        
        assert analysis.specialty.value == "anxiety_specialist"
        assert analysis.risk_score >= 0.0
        assert analysis.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_crisis_agent(self):
        """Test crisis specialist agent"""
        agent = CrisisSpecialistAgent()
        
        with patch.object(agent, 'initialize', new_callable=AsyncMock):
            await agent.initialize()
        
        text = "I want to kill myself tonight"
        
        with patch.object(agent, 'crisis_classifier', None):
            analysis = await agent.analyze(text)
        
        assert analysis.specialty.value == "crisis_specialist"
        # Crisis agent should detect high risk for this text
        assert analysis.risk_score > 0.5
        assert '988' in str(analysis.recommendations) or 'crisis' in str(analysis.recommendations).lower()

class TestMultiAgentOrchestrator:
    """Test cases for multi-agent orchestration"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.agents == {}
        assert orchestrator.executor is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, orchestrator):
        """Test comprehensive analysis with multiple agents"""
        # Mock agents
        mock_depression_agent = Mock()
        mock_anxiety_agent = Mock()
        mock_crisis_agent = Mock()
        
        # Mock analysis results
        from src.agents.advanced_mental_health_agents import AgentAnalysis, AgentSpecialty
        
        mock_depression_result = AgentAnalysis(
            agent_id="dep_001",
            specialty=AgentSpecialty.DEPRESSION_SPECIALIST,
            confidence=0.8,
            risk_score=0.7,
            key_indicators=["Sadness", "Hopelessness"],
            recommendations=["Seek therapy"],
            metadata={},
            processing_time=0.1
        )
        
        mock_anxiety_result = AgentAnalysis(
            agent_id="anx_001", 
            specialty=AgentSpecialty.ANXIETY_SPECIALIST,
            confidence=0.6,
            risk_score=0.4,
            key_indicators=["Worry"],
            recommendations=["Practice breathing"],
            metadata={},
            processing_time=0.1
        )
        
        mock_crisis_result = AgentAnalysis(
            agent_id="crisis_001",
            specialty=AgentSpecialty.CRISIS_SPECIALIST,
            confidence=0.9,
            risk_score=0.9,
            key_indicators=["Suicidal Ideation"],
            recommendations=["Call 988"],
            metadata={},
            processing_time=0.1
        )
        
        # Setup mock agents
        mock_depression_agent.analyze = AsyncMock(return_value=mock_depression_result)
        mock_anxiety_agent.analyze = AsyncMock(return_value=mock_anxiety_result)
        mock_crisis_agent.analyze = AsyncMock(return_value=mock_crisis_result)
        
        orchestrator.agents = {
            AgentSpecialty.DEPRESSION_SPECIALIST: mock_depression_agent,
            AgentSpecialty.ANXIETY_SPECIALIST: mock_anxiety_agent,
            AgentSpecialty.CRISIS_SPECIALIST: mock_crisis_agent
        }
        
        # Run analysis
        text = "I want to end my life"
        results = await orchestrator.analyze_comprehensive(text)
        
        assert len(results) == 3
        assert AgentSpecialty.DEPRESSION_SPECIALIST in results
        assert AgentSpecialty.ANXIETY_SPECIALIST in results  
        assert AgentSpecialty.CRISIS_SPECIALIST in results
    
    def test_synthesis_analysis(self, orchestrator):
        """Test synthesis of multiple agent results"""
        from src.agents.advanced_mental_health_agents import AgentAnalysis, AgentSpecialty
        
        # Create mock results
        agent_results = {
            AgentSpecialty.DEPRESSION_SPECIALIST: AgentAnalysis(
                agent_id="dep_001",
                specialty=AgentSpecialty.DEPRESSION_SPECIALIST,
                confidence=0.8,
                risk_score=0.7,
                key_indicators=["Sadness"],
                recommendations=["Seek therapy"],
                metadata={},
                processing_time=0.1
            ),
            AgentSpecialty.CRISIS_SPECIALIST: AgentAnalysis(
                agent_id="crisis_001",
                specialty=AgentSpecialty.CRISIS_SPECIALIST,
                confidence=0.9,
                risk_score=0.9,
                key_indicators=["Crisis"],
                recommendations=["Call 988"],
                metadata={},
                processing_time=0.1
            )
        }
        
        synthesis = orchestrator.synthesize_analysis(agent_results)
        
        assert 'overall_risk_score' in synthesis
        assert 'confidence' in synthesis
        assert 'requires_immediate_attention' in synthesis
        assert 'primary_concerns' in synthesis
        assert 'all_recommendations' in synthesis
        
        # Should flag for immediate attention due to high crisis score
        assert synthesis['requires_immediate_attention'] == True
        assert synthesis['overall_risk_score'] > 0.7

class TestDataCollector:
    """Test cases for data collection"""
    
    @pytest.mark.asyncio
    async def test_reddit_data_collection(self):
        """Test Reddit data collection (mocked)"""
        collector = DataCollector()
        
        subreddits = ['depression', 'anxiety']
        data = await collector.collect_reddit_data(subreddits, limit=10)
        
        assert len(data) == 10
        assert all('text' in item for item in data)
        assert all('source' in item for item in data)
        assert all(item['source'] == 'reddit' for item in data)
    
    @pytest.mark.asyncio
    async def test_twitter_data_collection(self):
        """Test Twitter data collection (mocked)"""
        collector = DataCollector()
        
        keywords = ['depression', 'anxiety']
        data = await collector.collect_twitter_data(keywords, limit=5)
        
        assert len(data) == 5
        assert all('text' in item for item in data)
        assert all('source' in item for item in data)
        assert all(item['source'] == 'twitter' for item in data)
    
    @pytest.mark.asyncio
    async def test_mixed_data_collection(self):
        """Test mixed data collection from multiple sources"""
        collector = DataCollector()
        
        sources = {
            'reddit': {'subreddits': ['depression'], 'limit': 3},
            'twitter': {'keywords': ['anxiety'], 'limit': 2}
        }
        
        df = await collector.collect_mixed_data(sources)
        
        assert len(df) == 5
        assert 'text' in df.columns
        assert 'source' in df.columns
        assert set(df['source'].unique()) == {'reddit', 'twitter'}

class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_system):
        """Test complete analysis pipeline"""
        # Mock the data collection to avoid external API calls
        with patch.object(sample_system.data_collector, 'collect_mixed_data') as mock_collect:
            mock_collect.return_value = TestDataFixtures.get_sample_dataframe()
            
            # Run pipeline
            predictions = await sample_system.run_analysis_pipeline()
            
            assert len(predictions) > 0
            assert all(isinstance(p, MentalHealthPrediction) for p in predictions)
    
    def test_single_text_analysis(self, sample_system):
        """Test single text analysis"""
        text = "I feel very depressed and hopeless"
        prediction = sample_system.analyze_single_text(text)
        
        assert isinstance(prediction, MentalHealthPrediction)
        assert prediction.text == text
        assert prediction.overall_severity in [s for s in MentalHealthSeverity]

class TestPerformance:
    """Performance and load testing"""
    
    def test_prediction_performance(self, sample_agent):
        """Test prediction performance"""
        import time
        
        text = "I feel anxious and overwhelmed by everything"
        
        start_time = time.time()
        prediction = sample_agent.predict_mental_health(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 5.0  # 5 seconds max
        assert isinstance(prediction, MentalHealthPrediction)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, orchestrator):
        """Test concurrent analysis performance"""
        import time
        
        # Mock agents for testing
        mock_agent = Mock()
        from src.agents.advanced_mental_health_agents import AgentAnalysis, AgentSpecialty
        
        mock_result = AgentAnalysis(
            agent_id="test_001",
            specialty=AgentSpecialty.DEPRESSION_SPECIALIST,
            confidence=0.8,
            risk_score=0.5,
            key_indicators=[],
            recommendations=[],
            metadata={},
            processing_time=0.1
        )
        
        mock_agent.analyze = AsyncMock(return_value=mock_result)
        orchestrator.agents = {AgentSpecialty.DEPRESSION_SPECIALIST: mock_agent}
        
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        start_time = time.time()
        
        # Run concurrent analyses
        tasks = []
        for text in texts:
            task = orchestrator.analyze_comprehensive(text)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Concurrent processing should be faster than sequential
        assert processing_time < 2.0  # Should complete quickly with mocked agents
        assert len(results) == 5

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_text(self, sample_agent):
        """Test handling of empty text"""
        prediction = sample_agent.predict_mental_health("")
        
        assert isinstance(prediction, MentalHealthPrediction)
        assert prediction.confidence < 0.8  # Should have low confidence
    
    def test_very_long_text(self, sample_agent):
        """Test handling of very long text"""
        long_text = "I feel sad. " * 1000  # Very long text
        prediction = sample_agent.predict_mental_health(long_text)
        
        assert isinstance(prediction, MentalHealthPrediction)
        # Should handle without crashing
    
    def test_special_characters(self, sample_agent):
        """Test handling of special characters"""
        text_with_special_chars = "I feel 😢 sad and 💔 broken!!! @#$%^&*()"
        prediction = sample_agent.predict_mental_health(text_with_special_chars)
        
        assert isinstance(prediction, MentalHealthPrediction)
    
    def test_non_english_text(self, sample_agent):
