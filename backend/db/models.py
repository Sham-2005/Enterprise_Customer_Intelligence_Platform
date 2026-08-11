"""
SQLAlchemy Database ORM Entity Models for ECIP.
Defines tables for User, PredictionLog, AuditLog, and ModelRegistryDB.
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="Business Analyst") # Administrator, Business Analyst, Data Scientist, Manager, Read-Only User
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), index=True)
    customer_id = Column(String(100), index=True)
    prediction_result = Column(Float)
    risk_level_or_tier = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), default="system")
    action = Column(String(100))
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
