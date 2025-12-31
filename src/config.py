"""Configuration management for the trading agent."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the trading agent."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Inject environment variables
        if 'claude' in config:
            config['claude']['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if 'data' not in config:
            config['data'] = {}
        config['data']['alpha_vantage_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')

        return config

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            'data/cache',
            'results',
            'logs'
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def trading(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get('trading', {})

    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})

    @property
    def strategies(self) -> Dict[str, Any]:
        """Get strategies configuration."""
        return self.config.get('strategies', {})

    @property
    def risk(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.config.get('risk', {})

    @property
    def backtest(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config.get('backtest', {})

    @property
    def claude(self) -> Dict[str, Any]:
        """Get Claude agent configuration."""
        return self.config.get('claude', {})

    @property
    def metrics(self) -> list:
        """Get performance metrics to track."""
        return self.config.get('metrics', [])

    def save(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = Path(path) if path else self.config_path

        # Remove API keys before saving
        config_copy = self.config.copy()
        if 'claude' in config_copy and 'api_key' in config_copy['claude']:
            del config_copy['claude']['api_key']
        if 'data' in config_copy and 'alpha_vantage_key' in config_copy['data']:
            del config_copy['data']['alpha_vantage_key']

        with open(save_path, 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file."""
    return Config(config_path)
