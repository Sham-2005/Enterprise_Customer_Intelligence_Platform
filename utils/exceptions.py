"""
Custom Exception Hierarchy for Enterprise Customer Intelligence Platform (ECIP).
Follows standard OOP exception design for distinct diagnostic handling.
"""

class ECIPBaseException(Exception):
    """Base exception class for all ECIP errors."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"[{self.__class__.__name__}] {self.message} | Details: {self.details}"
        return f"[{self.__class__.__name__}] {self.message}"


class ConfigurationError(ECIPBaseException):
    """Raised when system configuration loading or validation fails."""
    pass


class DataIngestionError(ECIPBaseException):
    """Raised when raw dataset loading or parsing fails."""
    pass


class DataValidationError(ECIPBaseException):
    """Raised when data schema validation fails."""
    pass


class FeatureEngineeringError(ECIPBaseException):
    """Raised when feature transformation or aggregation fails."""
    pass


class ModelTrainingError(ECIPBaseException):
    """Raised when machine learning model training fails."""
    pass


class InferenceError(ECIPBaseException):
    """Raised when model inference or prediction fails."""
    pass


class ReportGenerationError(ECIPBaseException):
    """Raised when report export/compilation fails."""
    pass


class APIException(ECIPBaseException):
    """Raised for REST API runtime failures."""
    def __init__(self, message: str, status_code: int = 500, details: str = None):
        super().__init__(message, details)
        self.status_code = status_code
