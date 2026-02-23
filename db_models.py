from sqlalchemy import Column, BigInteger, Float, String, DateTime
from database import Base

class Prediction(Base):
    __tablename__ = "prediction"

    id                = Column(BigInteger, primary_key=True, autoincrement=True)
    input_features    = Column(String(500))
    prediction_result = Column(Float)
    actual_result     = Column(Float)
    created_at        = Column(DateTime)