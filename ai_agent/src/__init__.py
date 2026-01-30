try:
    from src.dqn_agent import DQNAgent
    from src.dqn_model import DuelingDQN
    from src.connect_four_env import ConnectFourEnv
except ImportError:
    from .dqn_agent import DQNAgent
    from .dqn_model import DuelingDQN
    from .connect_four_env import ConnectFourEnv

__version__ = "1.0.0"
__all__ = ['DQNAgent', 'DuelingDQN', 'ConnectFourEnv']
