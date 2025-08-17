import pytest
import torch
from unittest.mock import patch, MagicMock

# Mock the transformers library to avoid loading actual models during testing
@patch('src.models.qwen_sentiment_model.QwenForSequenceClassification')
@patch('src.models.qwen_sentiment_model.AutoTokenizer')
def test_polymorphic_sentiment_analyzer_initialization(mock_tokenizer, mock_model_class):
    """Test initialization of PolymorphicSentimentAnalyzer"""
    # Mock the model and tokenizer
    mock_model_instance = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    
    # Import after patching
    from src.models.qwen_sentiment_model import PolymorphicSentimentAnalyzer
    
    # Create an instance
    analyzer = PolymorphicSentimentAnalyzer()
    
    # Assert that the model was initialized
    assert analyzer is not None
    mock_model_class.from_pretrained.assert_called_once()

def test_polymorphic_feature_extractor():
    """Test polymorphic feature extractor"""
    from src.models.polymorphic_features import PolymorphicFeatureExtractor
    
    # Create an instance
    extractor = PolymorphicFeatureExtractor()
    
    # Assert that the extractor was initialized
    assert extractor is not None

@patch('src.data.preprocessing.AutoTokenizer')
def test_text_preprocessing(mock_tokenizer_class):
    """Test text preprocessing functions"""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    # Import after patching
    from src.data.preprocessing import clean_text
    
    # Test cleaning function
    cleaned = clean_text("Hello, World! 123")
    assert isinstance(cleaned, str)

if __name__ == "__main__":
    pytest.main([__file__])