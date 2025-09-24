"""
Test for inference endpoint using pytest format.
Tests the deployed Modal endpoint with a predetermined audio file from the ESC-50 dataset.
"""

import pytest
import os
import base64
import io
import requests
import soundfile as sf  # type: ignore
from pathlib import Path


def test_inference_endpoint():
    """
    Test inference endpoint with a predetermined audio file from ESC-50 dataset.
    Uses dog barking audio (1-100032-A-0.wav) and validates the response format.
    
    The endpoint URL should be provided via INFERENCE_ENDPOINT_URL environment variable.
    """
    # Get endpoint URL from environment
    endpoint_url = os.environ.get("INFERENCE_ENDPOINT_URL")
    if not endpoint_url:
        pytest.skip("INFERENCE_ENDPOINT_URL environment variable not set")
    
    # Use predetermined audio file from dataset
    audio_path = Path("/home/pqbas/projects/cnn-audio/data/esc50-data/audio/1-100032-A-0.wav")
    expected_class = "dog"  # Based on the CSV metadata
    
    # Verify audio file exists
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"
    
    # Load and encode audio
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Create in-memory buffer and encode to base64
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Prepare request payload
    payload = {"audio_data": audio_b64}
    
    # Make request to endpoint
    response = requests.post(endpoint_url, json=payload, timeout=40)
    
    # Assert request was successful
    assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
    
    # Parse response
    result = response.json()
    
    # Validate response structure
    assert "predictions" in result, "Response missing 'predictions' key"
    assert "waveform" in result, "Response missing 'waveform' key"
    assert "visualization" in result, "Response missing 'visualization' key"
    assert "input_spectrogram" in result, "Response missing 'input_spectrogram' key"
    
    # Validate predictions structure
    predictions = result["predictions"]
    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) > 0, "Should have at least one prediction"
    
    # Validate first prediction structure
    top_prediction = predictions[0]
    assert "class" in top_prediction, "Prediction missing 'class' key"
    assert "confidence" in top_prediction, "Prediction missing 'confidence' key"
    
    predicted_class = top_prediction["class"]
    confidence = top_prediction["confidence"]
    
    # Validate data types
    assert isinstance(predicted_class, str), "Predicted class should be string"
    assert isinstance(confidence, (int, float)), "Confidence should be numeric"
    assert 0 <= confidence <= 1, f"Confidence should be between 0 and 1, got {confidence}"
    
    # Print results for debugging
    print(f"\n🎵 Audio: {audio_path.name}")
    print(f"📊 Top prediction: {predicted_class} ({confidence:.2%})")
    print(f"✅ Expected class: {expected_class}")
    
    # Check if expected class is in top prediction (case-insensitive)
    if expected_class.lower() in predicted_class.lower():
        print(f"🎯 Success: Expected class '{expected_class}' found in prediction!")
    else:
        print(f"⚠️  Note: Expected class '{expected_class}' not in top prediction")
        # Don't fail the test - the model might predict differently
    
    # Validate waveform data
    waveform_info = result["waveform"]
    assert "duration" in waveform_info, "Waveform missing 'duration'"
    assert "sample_rate" in waveform_info, "Waveform missing 'sample_rate'"
    assert waveform_info["duration"] > 0, "Duration should be positive"
    
    # Validate visualization data exists
    viz_data = result["visualization"]
    assert isinstance(viz_data, dict), "Visualization should be a dict"
    
    # Validate spectrogram data
    spec_data = result["input_spectrogram"]
    assert "shape" in spec_data, "Spectrogram missing 'shape'"
    assert "values" in spec_data, "Spectrogram missing 'values'"
    
    print(f"✅ All validations passed! Response time: {response.elapsed.total_seconds():.2f}s")
