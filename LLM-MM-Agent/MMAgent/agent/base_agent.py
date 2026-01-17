from abc import ABC, abstractmethod
import logging


class BaseAgent(ABC):

    def __init__(self, llm, logger_manager=None):
        """
        Initialize base agent with optional logger manager.

        Args:
            llm: Language model instance
            logger_manager: Optional MMExperimentLogger instance for proper error logging
        """
        self.llm = llm
        self.logger_manager = logger_manager

        # CRITICAL FIX: Use logger_manager if available, otherwise fall back to standard logger
        if logger_manager:
            self.logger = logger_manager.get_logger('main')
        else:
            # Fallback to standard Python logger (won't log to errors.log)
            self.logger = logging.getLogger(f'mmagent.{self.__class__.__name__}')
