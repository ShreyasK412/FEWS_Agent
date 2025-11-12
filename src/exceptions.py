"""
Custom exceptions for FEWS system.
"""


class FEWSException(Exception):
    """Base exception for FEWS system."""
    pass


class RegionNotFoundError(FEWSException):
    """Region not found in IPC data."""
    pass


class InsufficientDataError(FEWSException):
    """Insufficient data for analysis."""
    pass


class VectorStoreError(FEWSException):
    """Vector store error."""
    pass


class DomainKnowledgeError(FEWSException):
    """Domain knowledge lookup error."""
    pass


class RetrievalError(FEWSException):
    """Document retrieval error."""
    pass

