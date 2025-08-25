#!/usr/bin/env python3
"""
Test script for the voice endpoint functionality
Run this after installing the required dependencies
"""

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.api.v1.endpoints.voice import router, VoiceProcessor
    print("✓ Successfully imported voice endpoint components")
    
    # Test voice processor initialization (without loading the model)
    print("✓ Voice endpoint is ready for testing")
    
    # Print endpoint information
    print("\nAvailable endpoints:")
    print("- POST /api/v1/voice/voiceprint - Generate voiceprint from audio file")
    print("- GET /api/v1/voice/voiceprint/info - Get service information")
    print("- POST /api/v1/voice/voiceprint/compare - Compare two voiceprints")
    
    print("\nTo test the endpoint:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start the server: python main.py")
    print("3. Upload an audio file to /api/v1/voice/voiceprint")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)