import warnings

# Suppress speechbrain deprecation warnings before imports
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")

from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import HTTPException
from speechbrain.inference.speaker import EncoderClassifier
import torch
import numpy as np
import librosa
import logging

# Configure logging
logger = logging.getLogger(__name__)

class VoiceprintResponse(BaseModel):
    """Response model for voiceprint generation"""
    success: bool
    voiceprint: List[float]
    dimension: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    message: Optional[str] = None

class VoiceProcessor:
    """Voice processing class for generating voiceprints using SpeechBrain on CPU"""
    
    def __init__(self):
        self.encoder = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SpeechBrain ECAPA-TDNN model for CPU processing"""
        try:
            logger.info("Initializing SpeechBrain ECAPA-TDNN encoder for CPU...")
            
            # Force CPU usage
            device = torch.device('cpu')
            torch.set_num_threads(1)  # Optimize for CPU performance
            
            # Initialize the encoder with CPU device
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            
            logger.info("SpeechBrain ECAPA-TDNN encoder loaded successfully on CPU")
            
        except Exception as e:
            logger.error(f"Failed to initialize SpeechBrain encoder: {e}")
            self.encoder = None
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize voice processing model: {str(e)}"
            )
    
    def preprocess_audio(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Preprocess audio for voiceprint extraction
        
        Args:
            audio_data: Raw audio data as numpy array
            original_sr: Original sample rate
            
        Returns:
            Preprocessed audio data at 16kHz
        """
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                if audio_data.shape[0] < audio_data.shape[1]:
                    # Audio is (channels, samples)
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    # Audio is (samples, channels)  
                    audio_data = np.mean(audio_data, axis=1)
            
            # Ensure we have a 1D array
            audio_data = np.squeeze(audio_data)
            
            # Resample to 16kHz if needed (SpeechBrain expects 16kHz)
            if original_sr != 16000:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=original_sr, 
                        target_sr=16000,
                        res_type='kaiser_fast'  # Faster resampling
                    )
            
            # Normalize audio to [-1, 1] range
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Apply basic audio enhancement
            # Remove silence at the beginning and end
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to preprocess audio: {str(e)}"
            )
    
    def extract_voiceprint(self, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Extract voiceprint/embedding from preprocessed audio
        
        Args:
            audio_data: Preprocessed audio data at 16kHz
            
        Returns:
            Voice embedding as numpy array
        """
        if self.encoder is None:
            raise HTTPException(
                status_code=500,
                detail="Voice processing model not initialized"
            )
        
        try:
            # Ensure minimum duration (SpeechBrain works better with longer segments)
            min_duration = 1.0  # 1 second minimum
            min_samples = int(min_duration * 16000)
            
            # Remove batch dimension if it exists for length check
            if audio_data.dim() > 1:
                audio_flat = audio_data.squeeze()
            else:
                audio_flat = audio_data
                
            if len(audio_flat) < min_samples:
                # Pad with zeros if too short
                padding = min_samples - len(audio_flat)
                audio_flat = torch.nn.functional.pad(audio_flat, (0, padding), mode='constant')
            
            # Add batch dimension
            audio_tensor = audio_flat.unsqueeze(0).float()
            
            # Extract embedding using SpeechBrain on CPU
            with torch.no_grad():
                # Set to CPU explicitly
                audio_tensor = audio_tensor.cpu()
                embedding = self.encoder.encode_batch(audio_tensor)
                
                # Convert to numpy and remove batch dimension
                voiceprint = embedding.squeeze().cpu()
            
            return voiceprint
            
        except Exception as e:
            logger.error(f"Error extracting voiceprint: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract voiceprint: {str(e)}"
            )

# Initialize voice processor globally
voice_processor = VoiceProcessor()
