from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import tempfile
import os
import torch
import numpy as np
import librosa
import soundfile as sf
import logging
import warnings
import uuid
from app.services.voiceprint_service import voice_processor
from app.services.voiceprint_service import VoiceprintResponse
from app.services.voiceprint_qdrant_service import voiceprint_qdrant_hybrid_service

# Suppress audio library warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message=".*audioread instead.*")

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

@router.post("/voiceprint", response_model=VoiceprintResponse)
async def generate_voiceprint(audio_file: UploadFile = File(...), userId: str = Form(...), name: str = Form(...)):
    """
    Generate a voiceprint (speaker embedding) from an audio file using SpeechBrain ECAPA-TDNN on CPU.
    
    Args:
        audio_file: Audio file (WAV, MP3, FLAC, etc.)
        
    Returns:
        VoiceprintResponse with the extracted voiceprint vector
    """
    
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        # Also accept common file extensions even if MIME type is not set correctly
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file (WAV, MP3, FLAC, M4A, OGG, AAC)"
            )
    
    # Create temporary file to store uploaded audio
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio file with better error handling and format support
        try:
            # Determine file extension for format-specific loading
            file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ''
            
            # Try different loading methods based on format
            audio_data = None
            sample_rate = None
            
            # For WAV and FLAC, try soundfile first (no warnings)
            if file_ext in ['.wav', '.flac']:
                try:
                    audio_data, sample_rate = sf.read(temp_file_path, always_2d=False)
                except Exception:
                    pass
            
            # If soundfile failed or for other formats, use librosa with warning suppression
            if audio_data is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=False)
            
            # Calculate duration
            if len(audio_data.shape) == 1:
                duration = len(audio_data) / sample_rate
            else:
                duration = len(audio_data) / sample_rate if audio_data.shape[0] < audio_data.shape[1] else len(audio_data[0]) / sample_rate
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio file. Please ensure it's a valid audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC. Error: {str(e)}"
            )
        
        # Preprocess audio
        preprocessed_audio = voice_processor.preprocess_audio(audio_data, sample_rate)
        
        # Extract voiceprint
        voiceprint = voice_processor.extract_voiceprint(preprocessed_audio)
        
        # Convert numpy array to list for JSON serialization
        voiceprint_list = voiceprint.tolist()
        
        logger.info(f"Successfully generated voiceprint for {audio_file.filename}: "
                   f"dimension={len(voiceprint_list)}, duration={duration:.2f}s")
        
        voiceprint_qdrant_hybrid_service.upsert_voice(
            document_id=uuid.uuid4().hex,
            voiceprint=voiceprint_list,
            userId=userId,
            name=name,
            tags=["voiceprint"],
            metadata={"duration": duration}
        )

        return VoiceprintResponse(
            success=True,
            voiceprint=voiceprint_list,
            dimension=len(voiceprint_list),
            duration=round(duration, 2),
            sample_rate=16000,  # After preprocessing
            message=f"Voiceprint successfully generated from {audio_file.filename}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing audio file {audio_file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing audio: {str(e)}"
        )
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


@router.post("/voiceprint/indentify", response_model=List[dict[str, Any]])
async def generate_voiceprint(audio_file: UploadFile = File(...)):
    """
    Generate a voiceprint (speaker embedding) from an audio file using SpeechBrain ECAPA-TDNN on CPU.
    
    Args:
        audio_file: Audio file (WAV, MP3, FLAC, etc.)
        
    Returns:
        VoiceprintResponse with the extracted voiceprint vector
    """
    
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        # Also accept common file extensions even if MIME type is not set correctly
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file (WAV, MP3, FLAC, M4A, OGG, AAC)"
            )
    
    # Create temporary file to store uploaded audio
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio file with better error handling and format support
        try:
            # Determine file extension for format-specific loading
            file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ''
            
            # Try different loading methods based on format
            audio_data = None
            sample_rate = None
            
            # For WAV and FLAC, try soundfile first (no warnings)
            if file_ext in ['.wav', '.flac']:
                try:
                    audio_data, sample_rate = sf.read(temp_file_path, always_2d=False)
                except Exception:
                    pass
            
            # If soundfile failed or for other formats, use librosa with warning suppression
            if audio_data is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=False)
            
            # Calculate duration
            if len(audio_data.shape) == 1:
                duration = len(audio_data) / sample_rate
            else:
                duration = len(audio_data) / sample_rate if audio_data.shape[0] < audio_data.shape[1] else len(audio_data[0]) / sample_rate
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio file. Please ensure it's a valid audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC. Error: {str(e)}"
            )
        
        # Preprocess audio
        preprocessed_audio = voice_processor.preprocess_audio(audio_data, sample_rate)
        
        # Extract voiceprint
        voiceprint = voice_processor.extract_voiceprint(preprocessed_audio)
        
        # Convert numpy array to list for JSON serialization
        voiceprint_list = voiceprint.tolist()
        
        print(f"Voiceprint: {voiceprint_list}")

        logger.info(f"Successfully generated voiceprint for {audio_file.filename}: "
                   f"dimension={len(voiceprint_list)}, duration={duration:.2f}s")
        
        voice_match = voiceprint_qdrant_hybrid_service.search_voice_by_voiceprint(
            voiceprint=voiceprint_list,
            limit=2,
            score_threshold=0.1,
            name=None,
            user_id=None,
            tags=None
        )

        print(f"Voiceprint search: {voice_match}")
        return voice_match        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing audio file {audio_file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing audio: {str(e)}"
        )
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


@router.get("/voiceprint/info")
async def voiceprint_info():
    """
    Get information about the voiceprint generation service
    
    Returns:
        Service information and capabilities
    """
    return {
        "service": "Voiceprint Generation",
        "model": "SpeechBrain ECAPA-TDNN",
        "model_source": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dimension": 192,
        "supported_formats": ["WAV", "MP3", "FLAC", "M4A", "OGG", "AAC"],
        "processing": "CPU-optimized",
        "sample_rate": "16kHz (auto-resampled)",
        "minimum_duration": "1.0 seconds",
        "description": "Generates speaker embeddings/voiceprints for voice identification and verification"
    }

@router.post("/voiceprint/compares")
async def compare_voiceprints(voiceprint1: List[float], voiceprint2: List[float]):
    """
    Compare two voiceprints and return similarity score
    
    Args:
        voiceprint1: First voiceprint vector
        voiceprint2: Second voiceprint vector
        
    Returns:
        Similarity score between the two voiceprints (cosine similarity)
    """
    try:
        if len(voiceprint1) != len(voiceprint2):
            raise HTTPException(
                status_code=400,
                detail=f"Voiceprint dimensions don't match: {len(voiceprint1)} vs {len(voiceprint2)}"
            )
        
        # Convert to numpy arrays
        v1 = np.array(voiceprint1, dtype=np.float32)
        v2 = np.array(voiceprint2, dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_v1 * norm_v2)
        
        # Convert to Python float for JSON serialization
        similarity = float(similarity)
        
        return {
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "match_threshold_high": 0.8,  # High confidence threshold
            "match_threshold_medium": 0.6,  # Medium confidence threshold
            "interpretation": (
                "High match" if similarity > 0.8 else
                "Medium match" if similarity > 0.6 else
                "Low match" if similarity > 0.4 else
                "No match"
            )
        }
        
    except Exception as e:
        logger.error(f"Error comparing voiceprints: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare voiceprints: {str(e)}"
        )