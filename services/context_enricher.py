# services/context_enricher.py

from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ContextEnricher:
    """Enriches RAG context with ML model predictions"""

    def __init__(self):
        self.vector_store = None
        self.model = None
        self.fabric_type_map = {}
        self.fabric_pattern_map = {}

        # Try to load vector store
        try:
            from services.vector_store import get_vector_store
            self.vector_store = get_vector_store()
        except Exception as e:
            print(f"Vector store not available: {e}")

        # Try to load model
        try:
            from model import get_model
            self.model = get_model()
        except Exception as e:
            print(f"Model not available: {e}")

        # Load encoding maps
        try:
            import json
            base_dir = os.path.dirname(os.path.dirname(__file__))
            with open(os.path.join(base_dir, "target_encoding_mappings.json"), "r") as f:
                encodings = json.load(f)
            self.fabric_type_map = encodings.get("Fabric_Type", {})
            self.fabric_pattern_map = encodings.get("Fabric_Pattern", {})
        except Exception as e:
            print(f"Encoding maps not available: {e}")

    def enrich_with_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Add ML model prediction to the feature set"""

        # Encode categorical features
        fabric_type_encoded = features.get('fabric_type_encoded')
        if fabric_type_encoded is None:
            fabric_type_encoded = self.fabric_type_map.get(
                features.get('fabric_type', ''), 0
            )

        fabric_pattern_encoded = features.get('fabric_pattern_encoded')
        if fabric_pattern_encoded is None:
            fabric_pattern_encoded = self.fabric_pattern_map.get(
                features.get('fabric_pattern', ''), 0
            )

        # Build feature vector
        feature_vector = [
            float(features.get('pattern_complexity', 5.0)),
            float(features.get('operator_experience', 3.0)),
            float(fabric_pattern_encoded),
            float(features.get('cutting_method', 0)),
            float(fabric_type_encoded),
            float(features.get('marker_loss_pct', 8.0))
        ]

        # Get prediction from ML model
        if self.model is not None:
            try:
                prediction = float(self.model.predict([feature_vector])[0])
            except:
                prediction = features.get('marker_loss_pct', 8.0)
        else:
            prediction = features.get('marker_loss_pct', 8.0)

        # Determine risk level
        if prediction <= 5.0:
            risk_level = "Low"
        elif prediction <= 10.0:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "prediction": round(prediction, 2),
            "risk_level": risk_level,
            "feature_vector": feature_vector,
            "features": features
        }

    async def retrieve_relevant_context(
            self,
            features: Dict[str, Any],
            prediction_result: Dict[str, Any]
    ) -> List[Any]:
        """Retrieve relevant documents from vector database"""

        if self.vector_store is None:
            return []

        # Build search query
        search_query = f"""
        Fabric: {features.get('fabric_type', 'unknown')}
        Pattern: {features.get('fabric_pattern', 'unknown')}
        Complexity: {features.get('pattern_complexity', 5)}/10
        Predicted waste: {prediction_result['prediction']}% ({prediction_result['risk_level']} risk)
        """

        try:
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
            )
            docs = await retriever.ainvoke(search_query)
            return docs
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def extract_sources(self, documents: List[Any]) -> List[Dict]:
        """Extract source information from retrieved documents"""
        sources = []
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1)
                })
            else:
                sources.append({"source": "Unknown", "page": 1})
        return sources