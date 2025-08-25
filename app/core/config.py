from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = Field(default="Intelligence Service", env="PROJECT_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # MongoDB
    MONGODB_URL: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: str = Field(default="intelligence_db", env="DATABASE_NAME")
    
    # Qdrant
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="voiceprints-v1", env="QDRANT_COLLECTION_NAME")
    VECTOR_DIMENSION: int = Field(default=192, env="VECTOR_DIMENSION")
    
    # LLM Providers
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_LLM_MODEL: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(default=None, env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Gemini
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()