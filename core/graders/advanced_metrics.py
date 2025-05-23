"""
Advanced grading metrics for OpenEvals
"""
from typing import Any, Dict, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AdvancedMetrics:
    """Container for advanced metric implementations"""
    
    def __init__(self):
        self.embedding_model = None
    
    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the embedding model (lazy loading)"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(model_name)
            except ImportError:
                logger.error("SentenceTransformers not installed - cannot use embedding metrics")
                raise
    
    def llm_as_judge(
        self,
        system_output: str,
        reference: str,
        prompt_template: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use an LLM to judge the quality of the system output.
        
        Args:
            system_output: The output from the system being evaluated
            reference: The reference or ground truth output
            prompt_template: Template for the judge prompt
            model: Which LLM to use as judge
            temperature: Temperature for LLM sampling
            
        Returns:
            Dictionary containing:
            - value: Score between 0 and 1
            - passed: Boolean if threshold is provided
            - details: Full judge response and reasoning
        """
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package required for LLM-as-judge")
            raise
        
        if prompt_template is None:
            prompt_template = (
                "Evaluate the quality of the following response to the reference text.\n\n"
                "Reference: {reference}\n"
                "Response: {system_output}\n\n"
                "Provide a score between 0 and 1 where 1 is perfect and 0 is completely wrong. "
                "Include a brief explanation of your scoring."
            )
        
        prompt = prompt_template.format(
            reference=reference,
            system_output=system_output
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            score = self._extract_score_from_response(content)
            
            return {
                "value": score,
                "passed": None,  # No binary pass/fail unless threshold provided
                "details": {
                    "judge_response": content,
                    "model": model,
                    "prompt": prompt
                }
            }
        except Exception as e:
            logger.error(f"LLM-as-judge failed: {str(e)}")
            return {
                "value": 0.0,
                "passed": False,
                "details": {"error": str(e)}
            }
    
    def embedding_similarity(
        self,
        system_output: str,
        reference: str,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            system_output: The output from the system being evaluated
            reference: The reference or ground truth output
            model_name: Which embedding model to use
            
        Returns:
            Dictionary containing:
            - value: Cosine similarity score between 0 and 1
            - passed: Boolean if threshold is provided
            - details: Embedding model info
        """
        self.load_embedding_model(model_name)
        
        try:
            # Encode both texts
            embeddings = self.embedding_model.encode([system_output, reference])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0]
            
            # Scale to 0-1 range
            normalized_similarity = (similarity + 1) / 2
            
            return {
                "value": normalized_similarity,
                "passed": None,
                "details": {
                    "model": model_name,
                    "raw_similarity": similarity
                }
            }
        except Exception as e:
            logger.error(f"Embedding similarity failed: {str(e)}")
            return {
                "value": 0.0,
                "passed": False,
                "details": {"error": str(e)}
            }
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numeric score from LLM judge response"""
        # Simple pattern matching to find first number between 0 and 1
        import re
        match = re.search(r"\b0?\.\d+\b|\b[01]\b", response)
        if match:
            return float(match.group())
        return 0.0

# Initialize global instance
advanced_metrics = AdvancedMetrics()

# Register the metrics
from openevals.core.graders import register_grader
register_grader("llm_judge", advanced_metrics.llm_as_judge, MetricType.LLM_AS_JUDGE)
register_grader("embedding_sim", advanced_metrics.embedding_similarity, MetricType.SEMANTIC_SIMILARITY) 