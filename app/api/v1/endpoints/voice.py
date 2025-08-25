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
import torchaudio
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
        # Get file extension for proper format recognition
        file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else '.tmp'
        
        # Save uploaded file to temporary location with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio file with format-specific handling
        try:
            if file_ext in ['.m4a', '.mp4', '.aac']:
                # Use librosa for M4A/MP4/AAC files as torchaudio may not support them
                audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=True)
                signal = torch.from_numpy(audio_data).unsqueeze(0)
                fs = sample_rate
            else:
                # Use torchaudio for other formats
                signal, fs = torchaudio.load(temp_file_path)
        except Exception as e:
            # Fallback to librosa for all formats
            audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=True)
            signal = torch.from_numpy(audio_data).unsqueeze(0)
            fs = sample_rate

        # Calculate duration
        duration = signal.shape[1] / fs
        
        # Extract voiceprint
        voiceprint = voice_processor.extract_voiceprint(signal)
        
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
        
    # except HTTPException:
    #     # Re-raise HTTP exceptions
    #     raise
        
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
        # Get file extension for proper format recognition
        file_ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else '.tmp'
        
        # Save uploaded file to temporary location with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio file with format-specific handling
        try:
            if file_ext in ['.m4a', '.mp4', '.aac']:
                # Use librosa for M4A/MP4/AAC files as torchaudio may not support them
                audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=True)
                signal = torch.from_numpy(audio_data).unsqueeze(0)
                fs = sample_rate
            else:
                # Use torchaudio for other formats
                signal, fs = torchaudio.load(temp_file_path)
        except Exception as e:
            # Fallback to librosa for all formats
            audio_data, sample_rate = librosa.load(temp_file_path, sr=None, mono=True)
            signal = torch.from_numpy(audio_data).unsqueeze(0)
            fs = sample_rate

        # Calculate duration
        duration = signal.shape[1] / fs
        
        # Extract voiceprint
        voiceprint = voice_processor.extract_voiceprint(signal)
        
        # Convert numpy array to list for JSON serialization
        voiceprint_list = voiceprint.tolist()
        
        print(f"Voiceprint: {voiceprint_list}")

        logger.info(f"Successfully generated voiceprint for {audio_file.filename}: "
                   f"dimension={len(voiceprint_list)}, duration={duration:.2f}s")
        
        voice_match = voiceprint_qdrant_hybrid_service.search_voice_by_voiceprint(
            voiceprint=voiceprint_list,
            limit=1,
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
