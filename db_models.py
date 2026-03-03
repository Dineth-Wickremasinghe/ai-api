from sqlalchemy import Column, BigInteger, Float, String, DateTime
from database import Base

class Prediction(Base):
    __tablename__ = "prediction"

    id                = Column(BigInteger, primary_key=True, autoincrement=True)
    input_features    = Column(String(500))
    prediction_result = Column(Float)
    actual_result     = Column(Float)
    created_at        = Column(DateTime)

    fabric_type = Column(String)
    fabric_pattern = Column(String)
    cutting_method = Column(String)
    operator_experience = Column(Float)
    pattern_complexity = Column(Float)
    marker_loss_pct = Column(Float)