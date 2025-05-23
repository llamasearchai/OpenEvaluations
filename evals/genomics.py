"""
OpenEvaluations Genomics Evaluation Module
==========================================

Comprehensive genomics and bioinformatics evaluation suite for AI systems.
Covers variant analysis, gene expression, phylogenetics, and functional genomics.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from ..core.definitions import EvalTask, EvalCase, EvalInput, EvalOutputReference
from ..core.graders import MetricFunction, register_grader
from ..config import EvalSuiteConfig, EvalTaskConfigRef


@dataclass
class GenomicVariant:
    """Represents a genomic variant for evaluation"""
    chromosome: str
    position: int
    reference: str
    alternative: str
    variant_type: str  # SNV, INDEL, CNV, SV
    gene: Optional[str] = None
    transcript: Optional[str] = None
    protein_change: Optional[str] = None
    clinical_significance: Optional[str] = None
    population_frequency: Optional[float] = None


@dataclass
class GeneExpressionData:
    """Gene expression data structure"""
    gene_id: str
    gene_name: str
    expression_level: float
    fold_change: Optional[float] = None
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    sample_type: str = "unknown"


@dataclass
class PhylogeneticTree:
    """Phylogenetic tree structure for evaluation"""
    species: List[str]
    branch_lengths: Dict[str, float]
    topology: str
    support_values: Dict[str, float]
    molecular_clock: bool = False


class GenomicsEvaluationSuite:
    """Comprehensive genomics evaluation suite"""
    
    def __init__(self):
        self.name = "Genomics & Bioinformatics Evaluation"
        self.description = "Comprehensive evaluation of genomics and bioinformatics knowledge"
        self.version = "2.0.0"
        
    def get_variant_interpretation_tasks(self) -> List[EvalTask]:
        """Generate variant interpretation evaluation tasks"""
        tasks = []
        
        # SNV interpretation task
        snv_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze this genomic variant: Chr17:43094406 G>A in BRCA1. "
                           "The variant causes p.Gly1738Arg. Patient is a 35-year-old woman "
                           "with family history of breast cancer. Provide clinical interpretation.",
                    context={
                        "variant": GenomicVariant(
                            chromosome="17",
                            position=43094406,
                            reference="G",
                            alternative="A",
                            variant_type="SNV",
                            gene="BRCA1",
                            protein_change="p.Gly1738Arg",
                            population_frequency=0.0001
                        ),
                        "patient_info": {
                            "age": 35,
                            "sex": "female",
                            "family_history": "breast_cancer"
                        }
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "clinical_significance": "Likely Pathogenic",
                        "disease_association": "Hereditary Breast and Ovarian Cancer",
                        "recommendation": "Genetic counseling and enhanced screening",
                        "mechanism": "Loss of DNA repair function",
                        "inheritance": "Autosomal dominant",
                        "penetrance": "High (70-80% lifetime risk)"
                    }
                ),
                metadata={
                    "category": "clinical_genomics",
                    "difficulty": "expert",
                    "domain": "variant_interpretation"
                }
            ),
            
            EvalCase(
                input=EvalInput(
                    prompt="Interpret this pharmacogenomic variant: CYP2D6*4 (splicing defect). "
                           "Patient prescribed tramadol for post-operative pain. What are the implications?",
                    context={
                        "variant": GenomicVariant(
                            chromosome="22",
                            position=42525841,
                            reference="G",
                            alternative="A",
                            variant_type="SNV",
                            gene="CYP2D6",
                            clinical_significance="Pathogenic"
                        ),
                        "medication": "tramadol",
                        "indication": "post_operative_pain"
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "phenotype": "Poor Metabolizer",
                        "drug_response": "Reduced efficacy",
                        "mechanism": "Deficient CYP2D6 enzyme activity",
                        "recommendation": "Alternative analgesic or dose adjustment",
                        "monitoring": "Enhanced pain assessment"
                    }
                ),
                metadata={
                    "category": "pharmacogenomics",
                    "difficulty": "advanced",
                    "domain": "drug_metabolism"
                }
            )
        ]
        
        snv_task = EvalTask(
            id="genomics_snv_interpretation",
            name="Single Nucleotide Variant Interpretation",
            description="Evaluate AI ability to interpret clinical significance of SNVs",
            test_cases=snv_cases,
            input_format="structured_genomic_data",
            output_format="clinical_interpretation",
            grading_criteria=[
                {
                    "metric": "genomics_variant_accuracy",
                    "weight": 0.4,
                    "parameters": {"focus": "clinical_significance"}
                },
                {
                    "metric": "genomics_mechanism_understanding",
                    "weight": 0.3,
                    "parameters": {"focus": "molecular_mechanism"}
                },
                {
                    "metric": "clinical_recommendation_quality",
                    "weight": 0.3,
                    "parameters": {"focus": "actionable_recommendations"}
                }
            ]
        )
        
        tasks.append(snv_task)
        return tasks
    
    def get_gene_expression_tasks(self) -> List[EvalTask]:
        """Generate gene expression analysis tasks"""
        
        # RNA-seq differential expression analysis
        rnaseq_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze this RNA-seq differential expression data from cancer vs normal tissue. "
                           "Identify key dysregulated pathways and potential therapeutic targets.",
                    context={
                        "expression_data": [
                            GeneExpressionData("ENSG00000141510", "TP53", 2.3, -3.2, 0.001, 0.01, "tumor"),
                            GeneExpressionData("ENSG00000171862", "PTEN", 0.8, -2.1, 0.005, 0.02, "tumor"),
                            GeneExpressionData("ENSG00000136997", "MYC", 8.7, 4.5, 0.0001, 0.001, "tumor"),
                            GeneExpressionData("ENSG00000134086", "VHL", 1.2, -1.8, 0.01, 0.05, "tumor")
                        ],
                        "comparison": "tumor_vs_normal",
                        "tissue_type": "breast_cancer",
                        "sample_size": 100
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "dysregulated_pathways": [
                            "p53 signaling pathway",
                            "PI3K/AKT pathway",
                            "MYC target genes",
                            "Hypoxia response"
                        ],
                        "key_findings": {
                            "tumor_suppressors_down": ["TP53", "PTEN", "VHL"],
                            "oncogenes_up": ["MYC"],
                            "pathway_analysis": "Multiple tumor suppressor pathways disrupted"
                        },
                        "therapeutic_targets": [
                            "PI3K/mTOR inhibitors",
                            "MYC-targeted therapy",
                            "p53 restoration therapy"
                        ],
                        "biological_significance": "Classic tumor suppressor loss and oncogene activation pattern"
                    }
                ),
                metadata={
                    "category": "transcriptomics",
                    "difficulty": "expert",
                    "domain": "cancer_genomics"
                }
            )
        ]
        
        rnaseq_task = EvalTask(
            id="genomics_rnaseq_analysis",
            name="RNA-seq Differential Expression Analysis",
            description="Evaluate interpretation of RNA-seq differential expression data",
            test_cases=rnaseq_cases,
            input_format="expression_matrix",
            output_format="pathway_analysis",
            grading_criteria=[
                {
                    "metric": "pathway_identification_accuracy",
                    "weight": 0.4
                },
                {
                    "metric": "biological_interpretation_quality",
                    "weight": 0.4
                },
                {
                    "metric": "therapeutic_relevance",
                    "weight": 0.2
                }
            ]
        )
        
        return [rnaseq_task]
    
    def get_phylogenetics_tasks(self) -> List[EvalTask]:
        """Generate phylogenetic analysis tasks"""
        
        phylo_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze this phylogenetic tree of coronavirus spike proteins. "
                           "Determine evolutionary relationships and predict functional implications.",
                    context={
                        "tree": PhylogeneticTree(
                            species=["SARS-CoV-2", "SARS-CoV", "MERS-CoV", "HCoV-NL63", "HCoV-229E"],
                            branch_lengths={
                                "SARS-CoV-2": 0.02,
                                "SARS-CoV": 0.05,
                                "MERS-CoV": 0.15,
                                "HCoV-NL63": 0.25,
                                "HCoV-229E": 0.28
                            },
                            topology="((SARS-CoV-2,SARS-CoV),(MERS-CoV,(HCoV-NL63,HCoV-229E)))",
                            support_values={"node1": 0.98, "node2": 0.95, "node3": 0.87},
                            molecular_clock=True
                        ),
                        "gene": "spike_protein",
                        "analysis_type": "maximum_likelihood"
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "evolutionary_relationships": {
                            "closest_relatives": "SARS-CoV-2 and SARS-CoV are sister species",
                            "divergence_time": "Recent divergence, likely decades",
                            "clade_structure": "Two major clades: SARS-like and common cold-like"
                        },
                        "functional_predictions": {
                            "receptor_binding": "SARS-CoV-2 and SARS-CoV likely share ACE2 binding",
                            "pathogenicity": "SARS-like viruses show higher pathogenic potential",
                            "host_range": "Broad mammalian host range for SARS-like clade"
                        },
                        "evolutionary_insights": {
                            "selection_pressure": "Positive selection on receptor binding domain",
                            "recombination": "Evidence of recombination in coronavirus evolution",
                            "molecular_clock": "Consistent evolutionary rate across lineages"
                        }
                    }
                ),
                metadata={
                    "category": "phylogenetics",
                    "difficulty": "expert",
                    "domain": "viral_evolution"
                }
            )
        ]
        
        phylo_task = EvalTask(
            id="genomics_phylogenetic_analysis",
            name="Phylogenetic Analysis and Interpretation",
            description="Evaluate phylogenetic tree interpretation and evolutionary insights",
            test_cases=phylo_cases,
            input_format="phylogenetic_tree",
            output_format="evolutionary_analysis",
            grading_criteria=[
                {
                    "metric": "phylogenetic_interpretation_accuracy",
                    "weight": 0.4
                },
                {
                    "metric": "evolutionary_reasoning_quality",
                    "weight": 0.4
                },
                {
                    "metric": "functional_prediction_accuracy",
                    "weight": 0.2
                }
            ]
        )
        
        return [phylo_task]
    
    def get_functional_genomics_tasks(self) -> List[EvalTask]:
        """Generate functional genomics tasks"""
        
        func_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze this GWAS result for Type 2 Diabetes. The lead SNP rs7903146 "
                           "in TCF7L2 has OR=1.45, p=5e-12. Explain the biological mechanism.",
                    context={
                        "gwas_result": {
                            "snp": "rs7903146",
                            "gene": "TCF7L2",
                            "chromosome": "10",
                            "position": 114748339,
                            "alleles": "C/T",
                            "risk_allele": "T",
                            "odds_ratio": 1.45,
                            "p_value": 5e-12,
                            "beta": 0.372,
                            "maf": 0.28
                        },
                        "trait": "Type 2 Diabetes",
                        "study_size": 50000,
                        "population": "European"
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "biological_mechanism": {
                            "gene_function": "TCF7L2 encodes transcription factor 7-like 2",
                            "pathway": "Wnt signaling pathway",
                            "diabetes_connection": "Regulates insulin secretion and glucose homeostasis",
                            "tissue_expression": "Pancreatic beta cells, liver, intestine"
                        },
                        "functional_consequences": {
                            "molecular_effect": "Altered transcriptional regulation",
                            "cellular_impact": "Reduced insulin secretion",
                            "physiological_outcome": "Impaired glucose tolerance"
                        },
                        "clinical_implications": {
                            "risk_assessment": "29% population carries risk allele",
                            "effect_size": "Moderate effect (OR=1.45)",
                            "therapeutic_relevance": "Target for GLP-1 agonist therapy"
                        },
                        "population_genetics": {
                            "frequency_variation": "Higher frequency in Europeans",
                            "selection_signature": "Possible balancing selection",
                            "linkage_disequilibrium": "Tag SNP for larger haplotype block"
                        }
                    }
                ),
                metadata={
                    "category": "functional_genomics",
                    "difficulty": "expert", 
                    "domain": "gwas_interpretation"
                }
            )
        ]
        
        func_task = EvalTask(
            id="genomics_functional_analysis",
            name="Functional Genomics and GWAS Interpretation",
            description="Evaluate functional genomics analysis and GWAS interpretation",
            test_cases=func_cases,
            input_format="gwas_results",
            output_format="functional_annotation",
            grading_criteria=[
                {
                    "metric": "functional_annotation_accuracy",
                    "weight": 0.3
                },
                {
                    "metric": "pathway_mechanism_understanding",
                    "weight": 0.3
                },
                {
                    "metric": "clinical_translation_quality",
                    "weight": 0.2
                },
                {
                    "metric": "population_genetics_insight",
                    "weight": 0.2
                }
            ]
        )
        
        return [func_task]


# Register genomics-specific graders
@register_grader("genomics_variant_accuracy")
def genomics_variant_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of genomic variant interpretation"""
    try:
        # Parse response for key components
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check clinical significance accuracy
        clinical_score = 0.0
        expected_clinical = expected_content.get("clinical_significance", "").lower()
        if expected_clinical in response_lower:
            clinical_score = 1.0
        elif "pathogenic" in response_lower and "pathogenic" in expected_clinical:
            clinical_score = 0.8
        elif "benign" in response_lower and "benign" in expected_clinical:
            clinical_score = 0.8
        
        # Check mechanism understanding
        mechanism_score = 0.0
        expected_mechanism = expected_content.get("mechanism", "").lower()
        if expected_mechanism and expected_mechanism in response_lower:
            mechanism_score = 1.0
        elif any(term in response_lower for term in ["function", "protein", "activity"]):
            mechanism_score = 0.5
        
        # Check recommendation quality
        recommendation_score = 0.0
        expected_rec = expected_content.get("recommendation", "").lower()
        if expected_rec and any(word in response_lower for word in expected_rec.split()):
            recommendation_score = 1.0
        elif any(term in response_lower for term in ["screening", "counseling", "monitoring"]):
            recommendation_score = 0.7
        
        # Weighted average
        return (clinical_score * 0.4 + mechanism_score * 0.3 + recommendation_score * 0.3)
        
    except Exception:
        return 0.0


@register_grader("pathway_identification_accuracy")
def pathway_identification_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of pathway identification from expression data"""
    try:
        response_lower = response.lower()
        expected_pathways = expected.get("content", {}).get("dysregulated_pathways", [])
        
        # Check for pathway mentions
        pathway_matches = 0
        for pathway in expected_pathways:
            pathway_terms = pathway.lower().split()
            if any(term in response_lower for term in pathway_terms):
                pathway_matches += 1
        
        pathway_score = pathway_matches / len(expected_pathways) if expected_pathways else 0.0
        
        # Check for key gene mentions
        expected_genes = expected.get("content", {}).get("key_findings", {})
        gene_score = 0.0
        
        if "tumor_suppressors_down" in expected_genes:
            ts_genes = expected_genes["tumor_suppressors_down"]
            ts_matches = sum(1 for gene in ts_genes if gene.lower() in response_lower)
            gene_score += (ts_matches / len(ts_genes)) * 0.5
        
        if "oncogenes_up" in expected_genes:
            onc_genes = expected_genes["oncogenes_up"]
            onc_matches = sum(1 for gene in onc_genes if gene.lower() in response_lower)
            gene_score += (onc_matches / len(onc_genes)) * 0.5
        
        return (pathway_score * 0.6 + gene_score * 0.4)
        
    except Exception:
        return 0.0


@register_grader("phylogenetic_interpretation_accuracy")
def phylogenetic_interpretation_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate phylogenetic analysis accuracy"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check evolutionary relationships
        rel_score = 0.0
        if "evolutionary_relationships" in expected_content:
            expected_rel = expected_content["evolutionary_relationships"]
            if "closest_relatives" in expected_rel:
                if "sars" in response_lower and "sister" in response_lower:
                    rel_score += 0.5
            if "divergence_time" in expected_rel:
                if any(term in response_lower for term in ["recent", "decades", "divergence"]):
                    rel_score += 0.5
        
        # Check functional predictions
        func_score = 0.0
        if "functional_predictions" in expected_content:
            expected_func = expected_content["functional_predictions"]
            if "ace2" in response_lower or "receptor" in response_lower:
                func_score += 0.3
            if "pathogen" in response_lower:
                func_score += 0.3
            if "host" in response_lower:
                func_score += 0.4
        
        # Check evolutionary insights
        insight_score = 0.0
        if "evolutionary_insights" in expected_content:
            if any(term in response_lower for term in ["selection", "recombination", "molecular clock"]):
                insight_score = 0.8
        
        return (rel_score * 0.4 + func_score * 0.4 + insight_score * 0.2)
        
    except Exception:
        return 0.0


def create_genomics_evaluation_suite() -> EvalSuiteConfig:
    """Create comprehensive genomics evaluation suite configuration"""
    suite = GenomicsEvaluationSuite()
    
    tasks = []
    tasks.extend(suite.get_variant_interpretation_tasks())
    tasks.extend(suite.get_gene_expression_tasks())
    tasks.extend(suite.get_phylogenetics_tasks())
    tasks.extend(suite.get_functional_genomics_tasks())
    
    # Create task references
    task_refs = [
        EvalTaskConfigRef(
            task_id=task.id,
            weight=1.0 / len(tasks),
            grading_criteria=task.grading_criteria
        )
        for task in tasks
    ]
    
    return EvalSuiteConfig(
        id="genomics_comprehensive",
        name="Comprehensive Genomics & Bioinformatics Evaluation",
        description="Complete evaluation of genomics, variant analysis, gene expression, and phylogenetics",
        category="genomics",
        tasks=task_refs,
        metadata={
            "version": "2.0.0",
            "domains": ["variant_interpretation", "gene_expression", "phylogenetics", "functional_genomics"],
            "difficulty": "expert",
            "estimated_time_minutes": 120
        }
    )


# Export key components
__all__ = [
    "GenomicsEvaluationSuite",
    "GenomicVariant", 
    "GeneExpressionData",
    "PhylogeneticTree",
    "create_genomics_evaluation_suite",
    "genomics_variant_accuracy",
    "pathway_identification_accuracy",
    "phylogenetic_interpretation_accuracy"
] 