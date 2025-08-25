# Voice Endpoint Documentation

## Overview
The voice endpoint provides voiceprint (speaker embedding) generation using SpeechBrain's ECAPA-TDNN model. It's optimized for CPU processing and can generate 192-dimensional voice embeddings for speaker identification and verification.

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start the Service
```bash
python main.py
```

The service will be available at `http://localhost:8000`

## API Endpoints

### 1. Generate Voiceprint
**POST** `/api/v1/voice/voiceprint`

Generate a voiceprint (speaker embedding) from an audio file.

**Parameters:**
- `audio_file`: Audio file (multipart/form-data)
  - Supported formats: WAV, MP3, FLAC, M4A, OGG, AAC
  - Minimum duration: 1 second (recommended: 3+ seconds)

**Response:**
```json
{
  "success": true,
  "voiceprint": [0.1234, -0.5678, ...],  // 192-dimensional vector
  "dimension": 192,
  "duration": 5.23,
  "sample_rate": 16000,
  "message": "Voiceprint successfully generated from audio.wav"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/voice/voiceprint" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@path/to/audio.wav"
```

### 2. Service Information
**GET** `/api/v1/voice/voiceprint/info`

Get information about the voiceprint generation service.

**Response:**
```json
{
  "service": "Voiceprint Generation",
  "model": "SpeechBrain ECAPA-TDNN",
  "model_source": "speechbrain/spkrec-ecapa-voxceleb",
  "embedding_dimension": 192,
  "supported_formats": ["WAV", "MP3", "FLAC", "M4A", "OGG", "AAC"],
  "processing": "CPU-optimized",
  "sample_rate": "16kHz (auto-resampled)",
  "minimum_duration": "1.0 seconds"
}
```

### 3. Compare Voiceprints
**POST** `/api/v1/voice/voiceprint/compare`

Compare two voiceprints and return similarity score.

**Request Body:**
```json
{
  "voiceprint1": [0.1234, -0.5678, ...],  // 192-dimensional vector
  "voiceprint2": [0.2345, -0.6789, ...]   // 192-dimensional vector
}
```

**Response:**
```json
{
  "similarity": 0.8534,
  "distance": 0.1466,
  "match_threshold_high": 0.8,
  "match_threshold_medium": 0.6,
  "interpretation": "High match"
}
```

## Features

### CPU-Optimized Processing
- Uses CPU-only inference for better compatibility
- Optimized for single-threaded performance
- No GPU required

### Audio Preprocessing
- Automatic resampling to 16kHz using fast kaiser algorithm
- Smart mono conversion (handles both channel formats)
- Audio normalization to [-1, 1] range
- Silence trimming from beginning and end
- Minimum duration padding (1 second)
- Enhanced format compatibility (soundfile + librosa fallback)

### Model Details
- **Model**: SpeechBrain ECAPA-TDNN
- **Source**: speechbrain/spkrec-ecapa-voxceleb
- **Embedding Dimension**: 192
- **Input**: 16kHz mono audio
- **Architecture**: Enhanced ECAPA-TDNN with attention

### Similarity Scoring
- **Algorithm**: Cosine similarity
- **Range**: -1 to 1 (higher = more similar)
- **Thresholds**:
  - High match: > 0.8
  - Medium match: > 0.6
  - Low match: > 0.4
  - No match: ≤ 0.4

## Usage Examples

### Python Client
```python
import requests

# Generate voiceprint
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/voice/voiceprint',
        files={'audio_file': f}
    )
    result = response.json()
    voiceprint = result['voiceprint']

# Compare voiceprints
compare_response = requests.post(
    'http://localhost:8000/api/v1/voice/voiceprint/compare',
    json={
        'voiceprint1': voiceprint1,
        'voiceprint2': voiceprint2
    }
)
similarity = compare_response.json()['similarity']
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

// Generate voiceprint
const form = new FormData();
form.append('audio_file', fs.createReadStream('audio.wav'));

const response = await fetch('http://localhost:8000/api/v1/voice/voiceprint', {
    method: 'POST',
    body: form
});

const result = await response.json();
const voiceprint = result.voiceprint;
```

## Error Handling

The API returns appropriate HTTP status codes:
- **200**: Success
- **400**: Bad request (invalid file format, wrong parameters)
- **500**: Internal server error (model initialization, processing errors)

Common error scenarios:
- Unsupported audio format
- Corrupted audio file
- Audio too short (< 1 second)
- Model initialization failure

## Performance Notes

- **Processing Time**: ~1-3 seconds per audio file on modern CPU
- **Memory Usage**: ~500MB-1GB during processing
- **File Size Limits**: Depends on FastAPI configuration
- **Concurrent Requests**: Limited by CPU cores and memory

## Integration with Existing Pipeline

The voice endpoint can be used alongside the existing audio pipeline:
- Extract voiceprints during speaker diarization
- Store voiceprints in vector database for similarity search
- Use for speaker verification in real-time applications