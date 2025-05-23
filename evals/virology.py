"""
OpenEvaluations Virology Evaluation Module
==========================================

Comprehensive virology and infectious disease evaluation suite for AI systems.
Covers viral structure, replication, pathogenesis, epidemiology, and vaccine development.

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
class ViralGenome:
    """Represents viral genome characteristics"""
    genome_type: str  # DNA, RNA, ssRNA, dsRNA, ssDNA, dsDNA
    size_bp: int
    segments: int
    polarity: str  # positive, negative, ambisense
    host_integration: bool
    replication_strategy: str


@dataclass
class ViralProtein:
    """Viral protein structure and function"""
    name: str
    function: str
    location: str  # capsid, envelope, internal
    molecular_weight: float
    conservation_level: str  # high, medium, low
    antigenic_sites: List[str]


@dataclass
class EpidemiologicalData:
    """Epidemiological characteristics of viral outbreak"""
    pathogen: str
    basic_reproduction_number: float  # R0
    incubation_period_days: Tuple[int, int]
    infectious_period_days: int
    case_fatality_rate: float
    transmission_route: List[str]
    geographic_distribution: List[str]


@dataclass
class VaccineCandidate:
    """Vaccine development candidate information"""
    antigen_target: str
    vaccine_type: str  # live_attenuated, inactivated, subunit, mRNA, viral_vector
    immunogenicity: str  # high, medium, low
    safety_profile: str
    manufacturing_complexity: str
    storage_requirements: str


class VirologyEvaluationSuite:
    """Comprehensive virology evaluation suite"""
    
    def __init__(self):
        self.name = "Virology & Infectious Disease Evaluation"
        self.description = "Comprehensive evaluation of virology, pathogenesis, and vaccine development"
        self.version = "2.0.0"
        
    def get_viral_structure_tasks(self) -> List[EvalTask]:
        """Generate viral structure and organization tasks"""
        
        structure_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze the structure of SARS-CoV-2. Describe the key structural proteins, "
                           "their functions, and how they contribute to viral pathogenesis.",
                    context={
                        "virus": "SARS-CoV-2",
                        "genome": ViralGenome(
                            genome_type="ssRNA",
                            size_bp=29903,
                            segments=1,
                            polarity="positive",
                            host_integration=False,
                            replication_strategy="RNA-dependent RNA polymerase"
                        ),
                        "structural_proteins": [
                            ViralProtein("Spike", "receptor_binding_fusion", "envelope", 141.2, "medium", ["RBD", "NTD"]),
                            ViralProtein("Nucleocapsid", "RNA_binding_packaging", "internal", 45.6, "high", ["N1", "N2"]),
                            ViralProtein("Membrane", "viral_assembly", "envelope", 25.1, "high", ["M_domain"]),
                            ViralProtein("Envelope", "viral_budding", "envelope", 8.4, "high", ["E_channel"])
                        ]
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "structural_analysis": {
                            "spike_protein": {
                                "function": "ACE2 receptor binding and membrane fusion",
                                "pathogenesis_role": "Primary determinant of host tropism and transmissibility",
                                "antigenic_importance": "Major target for neutralizing antibodies"
                            },
                            "nucleocapsid_protein": {
                                "function": "RNA genome packaging and protection",
                                "pathogenesis_role": "Essential for viral replication and assembly",
                                "diagnostic_importance": "Highly conserved, good diagnostic target"
                            },
                            "membrane_protein": {
                                "function": "Viral envelope structure and assembly",
                                "pathogenesis_role": "Critical for viral morphogenesis"
                            },
                            "envelope_protein": {
                                "function": "Ion channel activity and viral release",
                                "pathogenesis_role": "Involved in viral budding and pathogenicity"
                            }
                        },
                        "pathogenesis_mechanisms": {
                            "cell_entry": "Spike-mediated endocytosis via ACE2 receptor",
                            "tissue_tropism": "Respiratory epithelium, with multi-organ involvement",
                            "immune_evasion": "Glycan shielding of spike protein",
                            "disease_severity": "Cytokine storm and vascular damage"
                        },
                        "therapeutic_targets": [
                            "Spike protein for neutralizing antibodies",
                            "RNA polymerase for antiviral drugs",
                            "Protease for viral replication inhibition"
                        ]
                    }
                ),
                metadata={
                    "category": "viral_structure",
                    "difficulty": "expert",
                    "domain": "coronavirus_biology"
                }
            ),
            
            EvalCase(
                input=EvalInput(
                    prompt="Compare the structure and replication strategies of HIV-1 and Influenza A virus. "
                           "Explain how their different structures lead to different clinical challenges.",
                    context={
                        "viruses": {
                            "HIV-1": ViralGenome(
                                genome_type="ssRNA",
                                size_bp=9719,
                                segments=1,
                                polarity="positive",
                                host_integration=True,
                                replication_strategy="reverse_transcription"
                            ),
                            "Influenza_A": ViralGenome(
                                genome_type="ssRNA",
                                size_bp=13500,
                                segments=8,
                                polarity="negative",
                                host_integration=False,
                                replication_strategy="transcription_replication"
                            )
                        }
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "structural_comparison": {
                            "genome_organization": {
                                "HIV": "Single segment, integrates into host genome",
                                "Influenza": "Eight segments, remains episomal"
                            },
                            "replication_strategy": {
                                "HIV": "Reverse transcription, persistent infection",
                                "Influenza": "Nuclear transcription, acute infection"
                            }
                        },
                        "clinical_implications": {
                            "HIV_challenges": [
                                "Proviral integration makes cure difficult",
                                "High mutation rate leads to drug resistance",
                                "Latent reservoirs persist despite treatment"
                            ],
                            "Influenza_challenges": [
                                "Segmented genome allows reassortment",
                                "Antigenic drift and shift create vaccine challenges",
                                "Seasonal circulation requires annual vaccination"
                            ]
                        },
                        "therapeutic_approaches": {
                            "HIV": "Highly active antiretroviral therapy (HAART)",
                            "Influenza": "Annual vaccination and neuraminidase inhibitors"
                        }
                    }
                ),
                metadata={
                    "category": "comparative_virology",
                    "difficulty": "expert",
                    "domain": "viral_replication"
                }
            )
        ]
        
        structure_task = EvalTask(
            id="virology_structural_analysis",
            name="Viral Structure and Organization Analysis",
            description="Evaluate understanding of viral structure and its relationship to pathogenesis",
            test_cases=structure_cases,
            input_format="viral_structural_data",
            output_format="structural_functional_analysis",
            grading_criteria=[
                {
                    "metric": "viral_structure_accuracy",
                    "weight": 0.4,
                    "parameters": {"focus": "structural_proteins"}
                },
                {
                    "metric": "pathogenesis_mechanism_understanding",
                    "weight": 0.4,
                    "parameters": {"focus": "structure_function_relationship"}
                },
                {
                    "metric": "therapeutic_target_identification",
                    "weight": 0.2,
                    "parameters": {"focus": "drug_vaccine_targets"}
                }
            ]
        )
        
        return [structure_task]
    
    def get_pathogenesis_tasks(self) -> List[EvalTask]:
        """Generate viral pathogenesis evaluation tasks"""
        
        pathogenesis_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Explain the pathogenesis of severe COVID-19. Include viral factors, "
                           "host immune responses, and the molecular basis of organ damage.",
                    context={
                        "disease": "severe_COVID-19",
                        "patient_profile": {
                            "age": 65,
                            "comorbidities": ["diabetes", "hypertension"],
                            "symptoms": ["dyspnea", "fever", "hypoxemia"]
                        },
                        "viral_load": "high",
                        "immune_markers": {
                            "IL-6": "elevated",
                            "TNF-alpha": "elevated",
                            "lymphocyte_count": "decreased"
                        }
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "viral_factors": {
                            "spike_mutations": "Enhanced ACE2 binding and immune evasion",
                            "viral_load": "High viral replication in respiratory tract",
                            "tissue_distribution": "Multi-organ involvement via ACE2 expression"
                        },
                        "immune_pathogenesis": {
                            "cytokine_storm": "Dysregulated IL-6, TNF-α, IL-1β production",
                            "lymphopenia": "T-cell exhaustion and apoptosis",
                            "complement_activation": "Excessive complement-mediated tissue damage",
                            "coagulation_disorders": "Hypercoagulable state and thrombosis"
                        },
                        "organ_damage_mechanisms": {
                            "lung": "Diffuse alveolar damage and ARDS",
                            "vascular": "Endothelial dysfunction and microthrombi",
                            "cardiac": "Myocarditis and cardiac injury",
                            "renal": "Acute kidney injury via multiple mechanisms"
                        },
                        "risk_factors": {
                            "age": "Immunosenescence and reduced adaptive immunity",
                            "comorbidities": "Pre-existing inflammation and ACE2 expression",
                            "genetics": "HLA variants and immune response genes"
                        }
                    }
                ),
                metadata={
                    "category": "viral_pathogenesis",
                    "difficulty": "expert",
                    "domain": "covid19_pathology"
                }
            )
        ]
        
        pathogenesis_task = EvalTask(
            id="virology_pathogenesis_analysis",
            name="Viral Pathogenesis and Disease Mechanisms",
            description="Evaluate understanding of viral pathogenesis and host-pathogen interactions",
            test_cases=pathogenesis_cases,
            input_format="clinical_viral_data",
            output_format="pathogenesis_analysis",
            grading_criteria=[
                {
                    "metric": "pathogenesis_mechanism_accuracy",
                    "weight": 0.5,
                    "parameters": {"focus": "molecular_mechanisms"}
                },
                {
                    "metric": "host_immune_response_understanding",
                    "weight": 0.3,
                    "parameters": {"focus": "immune_pathology"}
                },
                {
                    "metric": "clinical_correlation",
                    "weight": 0.2,
                    "parameters": {"focus": "symptoms_pathology"}
                }
            ]
        )
        
        return [pathogenesis_task]
    
    def get_epidemiology_tasks(self) -> List[EvalTask]:
        """Generate viral epidemiology evaluation tasks"""
        
        epi_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze the epidemiological characteristics of the 2009 H1N1 influenza pandemic. "
                           "Calculate key epidemiological parameters and predict intervention effectiveness.",
                    context={
                        "outbreak_data": EpidemiologicalData(
                            pathogen="H1N1_2009",
                            basic_reproduction_number=1.4,
                            incubation_period_days=(1, 4),
                            infectious_period_days=7,
                            case_fatality_rate=0.02,
                            transmission_route=["respiratory_droplets", "aerosols"],
                            geographic_distribution=["global"]
                        ),
                        "intervention_scenarios": {
                            "vaccination": {"coverage": 0.6, "efficacy": 0.8},
                            "social_distancing": {"compliance": 0.7, "effectiveness": 0.5},
                            "school_closure": {"duration_weeks": 8, "effectiveness": 0.3}
                        }
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "epidemiological_analysis": {
                            "reproduction_number": "R0 = 1.4 indicates moderate transmissibility",
                            "doubling_time": "Approximately 6-8 days in early phase",
                            "attack_rate": "Expected 20-30% population infection",
                            "peak_timing": "3-4 months after introduction"
                        },
                        "intervention_effectiveness": {
                            "vaccination_impact": "48% reduction in transmission (0.6 × 0.8)",
                            "combined_interventions": "Potential to reduce R_eff below 1.0",
                            "critical_vaccination_threshold": "Need >29% coverage for herd immunity"
                        },
                        "public_health_implications": {
                            "healthcare_burden": "Moderate ICU demand compared to seasonal flu",
                            "vulnerable_populations": "Pregnant women, young adults affected",
                            "economic_impact": "School closures major economic factor"
                        },
                        "surveillance_priorities": {
                            "genetic_monitoring": "Track antigenic drift and drug resistance",
                            "severity_assessment": "Monitor hospitalization and mortality rates",
                            "vaccine_effectiveness": "Real-world vaccine performance"
                        }
                    }
                ),
                metadata={
                    "category": "viral_epidemiology",
                    "difficulty": "expert",
                    "domain": "pandemic_preparedness"
                }
            )
        ]
        
        epi_task = EvalTask(
            id="virology_epidemiology_analysis",
            name="Viral Epidemiology and Outbreak Analysis",
            description="Evaluate epidemiological analysis and public health intervention assessment",
            test_cases=epi_cases,
            input_format="epidemiological_data",
            output_format="outbreak_analysis",
            grading_criteria=[
                {
                    "metric": "epidemiological_calculation_accuracy",
                    "weight": 0.4,
                    "parameters": {"focus": "quantitative_analysis"}
                },
                {
                    "metric": "intervention_assessment_quality",
                    "weight": 0.3,
                    "parameters": {"focus": "public_health_measures"}
                },
                {
                    "metric": "prediction_reasoning",
                    "weight": 0.3,
                    "parameters": {"focus": "outbreak_modeling"}
                }
            ]
        )
        
        return [epi_task]
    
    def get_vaccine_development_tasks(self) -> List[EvalTask]:
        """Generate vaccine development evaluation tasks"""
        
        vaccine_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Design a vaccine strategy for a novel coronavirus with 70% similarity to SARS-CoV-2. "
                           "Consider antigen selection, platform choice, and development timeline.",
                    context={
                        "novel_virus": {
                            "name": "CoV-X",
                            "similarity_to_sars_cov2": 0.7,
                            "spike_conservation": 0.6,
                            "rbd_conservation": 0.5,
                            "pathogenicity": "moderate",
                            "transmission": "airborne"
                        },
                        "available_platforms": [
                            VaccineCandidate("spike_protein", "mRNA", "high", "good", "medium", "ultra_cold"),
                            VaccineCandidate("spike_protein", "viral_vector", "high", "moderate", "low", "refrigerated"),
                            VaccineCandidate("spike_protein", "protein_subunit", "medium", "excellent", "high", "refrigerated"),
                            VaccineCandidate("whole_virus", "inactivated", "medium", "good", "medium", "refrigerated")
                        ]
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "antigen_selection": {
                            "primary_target": "Spike protein receptor binding domain",
                            "rationale": "Critical for viral entry and neutralizing antibody target",
                            "modifications": "Stabilized prefusion conformation",
                            "alternative_targets": "Nucleocapsid protein for T-cell responses"
                        },
                        "platform_recommendation": {
                            "primary_choice": "mRNA vaccine",
                            "advantages": [
                                "Rapid development and manufacturing",
                                "High immunogenicity",
                                "Easy sequence modification"
                            ],
                            "challenges": [
                                "Cold chain requirements",
                                "Limited global manufacturing capacity"
                            ],
                            "backup_platform": "Viral vector for broader accessibility"
                        },
                        "development_strategy": {
                            "phase_1": "Safety and immunogenicity (3 months)",
                            "phase_2": "Dose optimization and efficacy signals (6 months)",
                            "phase_3": "Large-scale efficacy trial (12 months)",
                            "regulatory": "Rolling review and emergency use authorization"
                        },
                        "manufacturing_considerations": {
                            "scale_up": "Billion-dose manufacturing capacity needed",
                            "global_distribution": "Multiple manufacturing sites required",
                            "cold_chain": "Infrastructure development for mRNA vaccines"
                        }
                    }
                ),
                metadata={
                    "category": "vaccine_development",
                    "difficulty": "expert",
                    "domain": "vaccine_design"
                }
            )
        ]
        
        vaccine_task = EvalTask(
            id="virology_vaccine_development",
            name="Vaccine Development and Strategy",
            description="Evaluate vaccine development strategy and platform selection",
            test_cases=vaccine_cases,
            input_format="vaccine_development_scenario",
            output_format="vaccine_strategy",
            grading_criteria=[
                {
                    "metric": "antigen_selection_rationale",
                    "weight": 0.3,
                    "parameters": {"focus": "target_selection"}
                },
                {
                    "metric": "platform_choice_justification",
                    "weight": 0.3,
                    "parameters": {"focus": "technology_assessment"}
                },
                {
                    "metric": "development_timeline_realism",
                    "weight": 0.2,
                    "parameters": {"focus": "regulatory_pathway"}
                },
                {
                    "metric": "manufacturing_scalability",
                    "weight": 0.2,
                    "parameters": {"focus": "global_health"}
                }
            ]
        )
        
        return [vaccine_task]


# Register virology-specific graders
@register_grader("viral_structure_accuracy")
def viral_structure_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of viral structure analysis"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check structural protein understanding
        structure_score = 0.0
        if "structural_analysis" in expected_content:
            struct_analysis = expected_content["structural_analysis"]
            protein_count = len(struct_analysis)
            correct_proteins = 0
            
            for protein, details in struct_analysis.items():
                if protein.replace("_", " ") in response_lower:
                    correct_proteins += 1
                    # Check function understanding
                    function = details.get("function", "").lower()
                    if function and any(word in response_lower for word in function.split()):
                        correct_proteins += 0.5
            
            structure_score = min(correct_proteins / protein_count, 1.0) if protein_count > 0 else 0.0
        
        # Check pathogenesis mechanism understanding
        pathogenesis_score = 0.0
        if "pathogenesis_mechanisms" in expected_content:
            mechanisms = expected_content["pathogenesis_mechanisms"]
            mechanism_matches = 0
            
            for mechanism, description in mechanisms.items():
                if mechanism.replace("_", " ") in response_lower:
                    mechanism_matches += 1
            
            pathogenesis_score = mechanism_matches / len(mechanisms) if mechanisms else 0.0
        
        # Check therapeutic target identification
        target_score = 0.0
        if "therapeutic_targets" in expected_content:
            targets = expected_content["therapeutic_targets"]
            target_matches = sum(1 for target in targets if target.lower() in response_lower)
            target_score = target_matches / len(targets) if targets else 0.0
        
        return (structure_score * 0.4 + pathogenesis_score * 0.4 + target_score * 0.2)
        
    except Exception:
        return 0.0


@register_grader("pathogenesis_mechanism_accuracy")
def pathogenesis_mechanism_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of pathogenesis mechanism understanding"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        total_score = 0.0
        component_count = 0
        
        # Check viral factors
        if "viral_factors" in expected_content:
            viral_factors = expected_content["viral_factors"]
            viral_score = 0.0
            for factor, description in viral_factors.items():
                if factor.replace("_", " ") in response_lower:
                    viral_score += 1
            viral_score = viral_score / len(viral_factors) if viral_factors else 0.0
            total_score += viral_score
            component_count += 1
        
        # Check immune pathogenesis
        if "immune_pathogenesis" in expected_content:
            immune_factors = expected_content["immune_pathogenesis"]
            immune_score = 0.0
            for factor, description in immune_factors.items():
                if factor.replace("_", " ") in response_lower or any(word in response_lower for word in factor.split("_")):
                    immune_score += 1
            immune_score = immune_score / len(immune_factors) if immune_factors else 0.0
            total_score += immune_score
            component_count += 1
        
        # Check organ damage mechanisms
        if "organ_damage_mechanisms" in expected_content:
            organ_mechanisms = expected_content["organ_damage_mechanisms"]
            organ_score = 0.0
            for organ, mechanism in organ_mechanisms.items():
                if organ in response_lower:
                    organ_score += 1
            organ_score = organ_score / len(organ_mechanisms) if organ_mechanisms else 0.0
            total_score += organ_score
            component_count += 1
        
        return total_score / component_count if component_count > 0 else 0.0
        
    except Exception:
        return 0.0


@register_grader("epidemiological_calculation_accuracy")
def epidemiological_calculation_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of epidemiological calculations and analysis"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check epidemiological analysis
        analysis_score = 0.0
        if "epidemiological_analysis" in expected_content:
            epi_analysis = expected_content["epidemiological_analysis"]
            correct_concepts = 0
            
            for concept, value in epi_analysis.items():
                concept_terms = concept.replace("_", " ").split()
                if any(term in response_lower for term in concept_terms):
                    correct_concepts += 1
            
            analysis_score = correct_concepts / len(epi_analysis) if epi_analysis else 0.0
        
        # Check intervention effectiveness
        intervention_score = 0.0
        if "intervention_effectiveness" in expected_content:
            interventions = expected_content["intervention_effectiveness"]
            intervention_matches = 0
            
            for intervention, effectiveness in interventions.items():
                if "vaccination" in intervention and "vaccination" in response_lower:
                    intervention_matches += 1
                elif "threshold" in intervention and "threshold" in response_lower:
                    intervention_matches += 1
                elif intervention.replace("_", " ") in response_lower:
                    intervention_matches += 1
            
            intervention_score = intervention_matches / len(interventions) if interventions else 0.0
        
        # Check quantitative understanding (look for numbers and percentages)
        quantitative_score = 0.0
        if any(char.isdigit() for char in response):
            quantitative_score = 0.8  # Bonus for including quantitative analysis
        
        return (analysis_score * 0.4 + intervention_score * 0.4 + quantitative_score * 0.2)
        
    except Exception:
        return 0.0


def create_virology_evaluation_suite() -> EvalSuiteConfig:
    """Create comprehensive virology evaluation suite configuration"""
    suite = VirologyEvaluationSuite()
    
    tasks = []
    tasks.extend(suite.get_viral_structure_tasks())
    tasks.extend(suite.get_pathogenesis_tasks())
    tasks.extend(suite.get_epidemiology_tasks())
    tasks.extend(suite.get_vaccine_development_tasks())
    
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
        id="virology_comprehensive",
        name="Comprehensive Virology & Infectious Disease Evaluation",
        description="Complete evaluation of viral biology, pathogenesis, epidemiology, and vaccine development",
        category="virology",
        tasks=task_refs,
        metadata={
            "version": "2.0.0",
            "domains": ["viral_structure", "pathogenesis", "epidemiology", "vaccine_development"],
            "difficulty": "expert",
            "estimated_time_minutes": 150
        }
    )


# Export key components
__all__ = [
    "VirologyEvaluationSuite",
    "ViralGenome",
    "ViralProtein", 
    "EpidemiologicalData",
    "VaccineCandidate",
    "create_virology_evaluation_suite",
    "viral_structure_accuracy",
    "pathogenesis_mechanism_accuracy",
    "epidemiological_calculation_accuracy"
] 