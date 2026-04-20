# services/rag_service.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from services.vector_store import get_vector_store
from config import settings
from typing import Dict, Any, List
from services.prompt_templates import CuttingRoomPrompts

# ============================================
# BASIC RAG (Document Q&A)
# ============================================

prompts = CuttingRoomPrompts()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always cite which document/page your answer comes from.

Context:
{context}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


async def query_rag(question: str, role: str = None, prediction_data: dict = None) -> dict:
    """Basic RAG query for document Q&A"""

    from google import genai
    from langchain_core.messages import HumanMessage, SystemMessage

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # Retrieve relevant chunks
    retrieved_docs = await retriever.ainvoke(question)
    context = format_docs(retrieved_docs)

    # ========== USE ROLE-BASED PROMPT ==========
    if role:
        prompt = prompts.get_prompt_for_role(
            role=role,
            question=question,
            context_docs=context,
            prediction_data=prediction_data
        )
    else:
        prompt = prompts.default_prompt(question, context)
    # ===========================================

    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always cite which document/page your answer comes from.

Context:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt
    )

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })

    return {
        "answer": response.text,
        "sources": sources
    }


# ============================================
# ENHANCED RAG (Risk Prediction)
# ============================================

# Global instance for enhanced RAG service
_enhanced_rag_service = None


def get_enhanced_rag_service():
    """Get or create the enhanced RAG service instance"""
    global _enhanced_rag_service
    if _enhanced_rag_service is None:
        _enhanced_rag_service = EnhancedRAGService()
    return _enhanced_rag_service


class EnhancedRAGService:
    """Enhanced RAG with specialized prompt engineering for risk prediction"""

    def __init__(self):
        self.client = None
        self.enricher = None
        self.prompts = None
        self.model_name = "gemini-2.5-flash"

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components lazily"""
        try:
            from google import genai
            from config import settings
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception as e:
            print(f"Gemini client not available: {e}")

        try:
            from services.context_enricher import ContextEnricher
            self.enricher = ContextEnricher()
        except Exception as e:
            print(f"Context enricher not available: {e}")

        try:
            from services.prompt_templates import CuttingRoomPrompts
            self.prompts = CuttingRoomPrompts()
        except Exception as e:
            print(f"Prompt templates not available: {e}")

    def _format_context(self, documents: List[Any]) -> str:
        """Convert retrieved documents to readable text"""
        if not documents:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if len(content) > 1000:
                content = content[:1000] + "..."

            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            page = doc.metadata.get('page', 1) if hasattr(doc, 'metadata') else 1

            formatted.append(f"[{i}] Source: {source}, Page: {page}\n{content}\n")

        return "\n".join(formatted)

    async def explain_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Explain why a risk was predicted"""

        if self.enricher is None:
            return {
                "prediction": 0,
                "risk_level": "Unknown",
                "explanation": "Context enricher not available",
                "sources": []
            }

        # Step 1: Get ML prediction
        enriched = self.enricher.enrich_with_prediction(features)

        # Step 2: Retrieve relevant context
        context_docs = await self.enricher.retrieve_relevant_context(features, enriched)
        context_text = self._format_context(context_docs)

        # Step 3: Build prompt
        if self.prompts:
            prompt = self.prompts.risk_explanation_prompt(
                prediction=enriched['prediction'],
                risk_level=enriched['risk_level'],
                context_docs=context_text,
                features=features
            )
        else:
            prompt = f"Explain why {enriched['risk_level']} risk was predicted for waste percentage {enriched['prediction']}%"

        # Step 4: Call Gemini
        explanation = "Explanation not available"
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                explanation = response.text
            except Exception as e:
                explanation = f"Error: {str(e)}"

        # Step 5: Extract sources
        sources = self.enricher.extract_sources(context_docs)

        return {
            "prediction": enriched['prediction'],
            "risk_level": enriched['risk_level'],
            "explanation": explanation,
            "sources": sources
        }

    async def get_recommendations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get actionable cutting recommendations"""

        if self.enricher is None:
            return {
                "prediction": 0,
                "risk_level": "Unknown",
                "immediate_action": "Service unavailable",
                "process_change": "Check configuration",
                "expected_improvement": "Unknown",
                "confidence": "Low",
                "based_on": "Error"
            }

        enriched = self.enricher.enrich_with_prediction(features)
        context_docs = await self.enricher.retrieve_relevant_context(features, enriched)
        context_text = self._format_context(context_docs)

        if self.prompts:
            prompt = self.prompts.actionable_recommendations_prompt(
                prediction=enriched['prediction'],
                risk_level=enriched['risk_level'],
                context_docs=context_text,
                features=features
            )
        else:
            prompt = f"Provide recommendations to reduce waste from {enriched['prediction']}%"

        raw_response = ""
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                raw_response = response.text
            except Exception as e:
                raw_response = f"Error: {str(e)}"

        recommendations = self._parse_recommendations(raw_response, enriched)
        return recommendations

    async def get_alert(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Get one-line alert for dashboard"""

        if self.enricher is None:
            return {"alert": "⚠️ System error: Context enricher not available"}

        enriched = self.enricher.enrich_with_prediction(features)

        if self.prompts:
            prompt = self.prompts.supervisor_alert_prompt(
                prediction=enriched['prediction'],
                risk_level=enriched['risk_level'],
                features=features
            )
        else:
            prompt = f"Create an alert for {enriched['risk_level']} risk with {enriched['prediction']}% waste"

        alert = f"{enriched['risk_level']} risk: {enriched['prediction']}% waste predicted"
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                alert = response.text.strip()
            except Exception as e:
                alert = f"⚠️ ALERT: {enriched['risk_level']} risk ({enriched['prediction']}% waste)"

        return {"alert": alert}

    def _parse_recommendations(self, raw_text: str, enriched: Dict) -> Dict[str, Any]:
        """Parse the structured recommendations from AI response"""

        result = {
            "prediction": enriched['prediction'],
            "risk_level": enriched['risk_level'],
            "immediate_action": "Review cutting parameters",
            "process_change": "Monitor and collect more data",
            "expected_improvement": "2-5% waste reduction",
            "confidence": "Medium",
            "based_on": "General best practices"
        }

        lines = raw_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if "IMMEDIATE ACTION" in line.upper():
                current_section = "immediate_action"
                if ':' in line:
                    result[current_section] = line.split(':', 1)[1].strip()
            elif "PROCESS CHANGE" in line.upper():
                current_section = "process_change"
                if ':' in line:
                    result[current_section] = line.split(':', 1)[1].strip()
            elif "EXPECTED IMPROVEMENT" in line.upper():
                current_section = "expected_improvement"
                if ':' in line:
                    result[current_section] = line.split(':', 1)[1].strip()
            elif "CONFIDENCE" in line.upper():
                current_section = "confidence"
                if ':' in line:
                    result[current_section] = line.split(':', 1)[1].strip()
            elif "BASED ON" in line.upper():
                current_section = "based_on"
                if ':' in line:
                    result[current_section] = line.split(':', 1)[1].strip()
            elif current_section and line and not line.startswith('**'):
                if result[current_section] == line:
                    pass
                elif len(result[current_section]) < len(line):
                    result[current_section] = line

        return result