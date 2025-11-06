"""
Production Configuration for PADIM (Publicly-Achievable Defensive Impact Model)
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Base configuration class."""

    # Application settings
    APP_NAME = "PADIM"
    VERSION = "0.1.0"
    DEBUG = False

    # Database settings
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/padim.db")
    RAPM_MODELS_PATH = os.getenv("RAPM_MODELS_PATH", "models/rapm_models.pkl")

    # Security settings (minimal for data processing tool)
    SECRET_KEY = os.getenv("SECRET_KEY", "padim-development-key")

    # API settings
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))  # requests per hour
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))  # seconds (longer for NBA API)

    # Monitoring settings
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/padim.log")

    # RAPM Model settings
    RAPM_CACHE_SIZE = int(os.getenv("RAPM_CACHE_SIZE", "10"))
    RAPM_TIMEOUT = int(os.getenv("RAPM_TIMEOUT", "300"))  # seconds (RAPM training can be slow)

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        # Create required directories
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        # Check if database directory exists (don't require database file yet)
        db_dir = Path(cls.DATABASE_PATH).parent
        db_dir.mkdir(exist_ok=True)

        return True

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    RAPM_TIMEOUT = 60  # Faster timeouts for development

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "INFO"
    RAPM_TIMEOUT = 1800  # Longer timeouts for production training

    @classmethod
    def validate(cls) -> bool:
        """Validate production configuration."""
        if not super().validate():
            return False

        # Production-specific validations could be added here
        # For now, just ensure basic setup is valid
        return True

def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()
