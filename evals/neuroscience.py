"""
OpenEvaluations Neuroscience Evaluation Module
==============================================

Comprehensive neuroscience evaluation suite for AI systems.
Covers neural circuits, cognitive neuroscience, neurological disorders, and brain imaging.

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
class NeuralCircuit:
    """Neural circuit structure and function"""
    name: str
    brain_regions: List[str]
    neurotransmitters: List[str]
    function: str
    disorders: List[str]
    connectivity_pattern: str


@dataclass
class NeuroimagingData:
    """Neuroimaging data structure"""
    modality: str  # fMRI, DTI, PET, EEG, MEG
    region: str
    activation_level: float
    volume_mm3: Optional[float] = None
    connectivity_strength: Optional[float] = None
    timepoint: Optional[str] = None


@dataclass
class CognitiveTask:
    """Cognitive assessment task"""
    name: str
    cognitive_domain: str
    neural_correlates: List[str]
    measurement: str
    normal_range: Tuple[float, float]
    patient_score: float


@dataclass
class NeurologicalDisorder:
    """Neurological disorder characteristics"""
    name: str
    pathophysiology: str
    affected_regions: List[str]
    symptoms: List[str]
    biomarkers: List[str]
    treatment_approaches: List[str]


class NeuroscienceEvaluationSuite:
    """Comprehensive neuroscience evaluation suite"""
    
    def __init__(self):
        self.name = "Neuroscience & Cognitive Assessment Evaluation"
        self.description = "Comprehensive evaluation of neuroscience, cognition, and neurological disorders"
        self.version = "2.0.0"
        
    def get_neural_circuit_tasks(self) -> List[EvalTask]:
        """Generate neural circuit analysis tasks"""
        
        circuit_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze the reward circuit in addiction. Describe the key brain regions, "
                           "neurotransmitter systems, and how chronic drug use alters this circuit.",
                    context={
                        "circuit": NeuralCircuit(
                            name="reward_circuit",
                            brain_regions=["VTA", "nucleus_accumbens", "prefrontal_cortex", "amygdala"],
                            neurotransmitters=["dopamine", "GABA", "glutamate"],
                            function="reward_motivation_learning",
                            disorders=["addiction", "depression", "ADHD"],
                            connectivity_pattern="mesolimbic_mesocortical"
                        ),
                        "addiction_state": "chronic_cocaine_use",
                        "neuroadaptations": [
                            "dopamine_receptor_downregulation",
                            "reduced_prefrontal_control",
                            "enhanced_amygdala_reactivity"
                        ]
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "circuit_anatomy": {
                            "VTA": "Dopaminergic neurons, reward prediction error signaling",
                            "nucleus_accumbens": "Dopamine release, reward evaluation and motivation",
                            "prefrontal_cortex": "Executive control, decision-making, impulse control",
                            "amygdala": "Emotional associations, stress and cue reactivity"
                        },
                        "neurotransmitter_functions": {
                            "dopamine": "Reward prediction, motivation, learning",
                            "GABA": "Inhibitory control, anxiety regulation",
                            "glutamate": "Learning, plasticity, excitatory transmission"
                        },
                        "addiction_neuroadaptations": {
                            "tolerance": "Dopamine receptor downregulation reduces reward sensitivity",
                            "compulsion": "Reduced prefrontal control over drug-seeking behavior",
                            "craving": "Enhanced amygdala response to drug-associated cues",
                            "withdrawal": "Dysregulated stress and negative affect systems"
                        },
                        "therapeutic_targets": [
                            "Dopamine system modulation",
                            "Cognitive behavioral therapy for prefrontal function",
                            "Stress reduction for amygdala reactivity",
                            "GABA enhancement for anxiety"
                        ]
                    }
                ),
                metadata={
                    "category": "neural_circuits",
                    "difficulty": "expert",
                    "domain": "addiction_neuroscience"
                }
            ),
            
            EvalCase(
                input=EvalInput(
                    prompt="Explain the neural basis of memory consolidation. Describe the roles of "
                           "hippocampus, neocortex, and sleep in converting short-term to long-term memory.",
                    context={
                        "memory_circuit": NeuralCircuit(
                            name="memory_consolidation",
                            brain_regions=["hippocampus", "entorhinal_cortex", "neocortex", "thalamus"],
                            neurotransmitters=["acetylcholine", "glutamate", "GABA"],
                            function="memory_encoding_consolidation",
                            disorders=["Alzheimer_disease", "amnesia"],
                            connectivity_pattern="hippocampal_neocortical"
                        ),
                        "sleep_stages": ["NREM", "REM", "slow_wave_sleep"],
                        "consolidation_timeframe": "hours_to_years"
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "hippocampal_function": {
                            "encoding": "Rapid binding of distributed cortical representations",
                            "pattern_separation": "Distinguishing similar experiences",
                            "pattern_completion": "Retrieving complete memories from partial cues",
                            "replay": "Reactivation of memory traces during rest and sleep"
                        },
                        "neocortical_function": {
                            "long_term_storage": "Gradual integration into cortical networks",
                            "semantic_processing": "Extraction of general knowledge",
                            "schema_integration": "Linking new memories to existing knowledge"
                        },
                        "sleep_mechanisms": {
                            "slow_wave_sleep": "Hippocampal-cortical dialogue, memory replay",
                            "spindles": "Thalamic gating of cortical plasticity",
                            "REM_sleep": "Memory integration and emotional processing"
                        },
                        "consolidation_timeline": {
                            "systems_consolidation": "Weeks to years for cortical independence",
                            "synaptic_consolidation": "Hours for protein synthesis-dependent stability"
                        }
                    }
                ),
                metadata={
                    "category": "memory_systems",
                    "difficulty": "expert",
                    "domain": "cognitive_neuroscience"
                }
            )
        ]
        
        circuit_task = EvalTask(
            id="neuroscience_circuit_analysis",
            name="Neural Circuit Analysis and Function",
            description="Evaluate understanding of neural circuits and their dysfunction in disease",
            test_cases=circuit_cases,
            input_format="neural_circuit_data",
            output_format="circuit_functional_analysis",
            grading_criteria=[
                {
                    "metric": "neural_circuit_accuracy",
                    "weight": 0.4,
                    "parameters": {"focus": "anatomical_connections"}
                },
                {
                    "metric": "neurotransmitter_understanding",
                    "weight": 0.3,
                    "parameters": {"focus": "synaptic_mechanisms"}
                },
                {
                    "metric": "pathophysiology_insight",
                    "weight": 0.3,
                    "parameters": {"focus": "disease_mechanisms"}
                }
            ]
        )
        
        return [circuit_task]
    
    def get_cognitive_assessment_tasks(self) -> List[EvalTask]:
        """Generate cognitive assessment tasks"""
        
        cognitive_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Interpret these neuropsychological test results for a 72-year-old patient. "
                           "Determine the pattern of cognitive deficits and likely diagnosis.",
                    context={
                        "patient_info": {
                            "age": 72,
                            "education_years": 16,
                            "complaints": ["memory_loss", "word_finding_difficulty"]
                        },
                        "cognitive_tests": [
                            CognitiveTask("MMSE", "global_cognition", ["widespread"], "total_score", (24, 30), 22),
                            CognitiveTask("verbal_fluency", "executive_language", ["left_frontal"], "words_per_minute", (15, 25), 8),
                            CognitiveTask("trail_making_B", "executive", ["prefrontal"], "seconds", (30, 90), 180),
                            CognitiveTask("delayed_recall", "episodic_memory", ["hippocampus"], "words_recalled", (8, 15), 3),
                            CognitiveTask("digit_span", "working_memory", ["frontal_parietal"], "digit_length", (5, 9), 6)
                        ],
                        "neuroimaging": [
                            NeuroimagingData("MRI", "hippocampus", 0.7, 3200, None, "structural"),
                            NeuroimagingData("MRI", "medial_temporal", 0.6, None, None, "structural"),
                            NeuroimagingData("PET", "posterior_cingulate", 0.5, None, None, "metabolism")
                        ]
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "cognitive_profile": {
                            "memory_deficits": {
                                "episodic_memory": "Severely impaired (delayed recall 3/15)",
                                "working_memory": "Relatively preserved (digit span 6)",
                                "pattern": "Disproportionate episodic memory loss"
                            },
                            "executive_deficits": {
                                "verbal_fluency": "Impaired (8 words, expected >15)",
                                "cognitive_flexibility": "Severely impaired (TMT-B 180s)",
                                "pattern": "Executive dysfunction with language involvement"
                            },
                            "global_cognition": "Mild impairment (MMSE 22/30)"
                        },
                        "neuroanatomical_correlations": {
                            "hippocampal_atrophy": "Explains severe episodic memory deficits",
                            "left_frontal_dysfunction": "Accounts for verbal fluency impairment",
                            "posterior_cingulate_hypometabolism": "Early Alzheimer signature"
                        },
                        "diagnostic_interpretation": {
                            "primary_diagnosis": "Probable Alzheimer's Disease (mild stage)",
                            "evidence": [
                                "Disproportionate episodic memory impairment",
                                "Hippocampal atrophy on MRI",
                                "Posterior cingulate hypometabolism",
                                "Executive and language involvement"
                            ],
                            "differential_considerations": [
                                "Frontotemporal dementia (less likely given memory pattern)",
                                "Vascular cognitive impairment (would expect more variable profile)"
                            ]
                        },
                        "recommendations": {
                            "further_testing": "CSF biomarkers, amyloid PET",
                            "treatment": "Cholinesterase inhibitors, cognitive rehabilitation",
                            "monitoring": "Annual neuropsychological assessment"
                        }
                    }
                ),
                metadata={
                    "category": "cognitive_assessment",
                    "difficulty": "expert",
                    "domain": "clinical_neuropsychology"
                }
            )
        ]
        
        cognitive_task = EvalTask(
            id="neuroscience_cognitive_assessment",
            name="Cognitive Assessment and Neuropsychological Interpretation",
            description="Evaluate interpretation of cognitive test results and neuroanatomical correlations",
            test_cases=cognitive_cases,
            input_format="neuropsychological_data",
            output_format="cognitive_diagnostic_report",
            grading_criteria=[
                {
                    "metric": "cognitive_profile_accuracy",
                    "weight": 0.4,
                    "parameters": {"focus": "test_interpretation"}
                },
                {
                    "metric": "neuroanatomical_correlation",
                    "weight": 0.3,
                    "parameters": {"focus": "brain_behavior_relationships"}
                },
                {
                    "metric": "diagnostic_reasoning",
                    "weight": 0.3,
                    "parameters": {"focus": "clinical_decision_making"}
                }
            ]
        )
        
        return [cognitive_task]
    
    def get_neurological_disorder_tasks(self) -> List[EvalTask]:
        """Generate neurological disorder evaluation tasks"""
        
        disorder_cases = [
            EvalCase(
                input=EvalInput(
                    prompt="Analyze the pathophysiology of Parkinson's disease. Explain the progression "
                           "from molecular to circuit-level dysfunction and clinical manifestations.",
                    context={
                        "disorder": NeurologicalDisorder(
                            name="Parkinson_disease",
                            pathophysiology="alpha_synuclein_aggregation_dopaminergic_loss",
                            affected_regions=["substantia_nigra", "striatum", "cortex", "brainstem"],
                            symptoms=["bradykinesia", "rigidity", "tremor", "gait_disturbance"],
                            biomarkers=["alpha_synuclein", "dopamine_transporter", "neuromelanin"],
                            treatment_approaches=["L-DOPA", "DBS", "exercise", "neuroprotection"]
                        ),
                        "disease_stages": ["preclinical", "early_motor", "advanced", "dementia"],
                        "pathological_progression": "brainstem_to_cortex"
                    }
                ),
                expected_output=EvalOutputReference(
                    content={
                        "molecular_pathology": {
                            "alpha_synuclein": "Misfolded protein forms Lewy bodies",
                            "neurodegeneration": "Progressive loss of dopaminergic neurons",
                            "spreading": "Prion-like transmission between brain regions",
                            "oxidative_stress": "Mitochondrial dysfunction and cell death"
                        },
                        "circuit_dysfunction": {
                            "basal_ganglia": "Reduced dopamine leads to impaired motor control",
                            "direct_pathway": "Decreased activation, reduced movement initiation",
                            "indirect_pathway": "Increased inhibition, movement suppression",
                            "oscillations": "Abnormal beta rhythms in motor circuits"
                        },
                        "clinical_progression": {
                            "preclinical": "Subtle motor and non-motor symptoms",
                            "early_motor": "Unilateral tremor, bradykinesia",
                            "advanced": "Bilateral symptoms, motor fluctuations",
                            "late_stage": "Cognitive decline, dyskinesias"
                        },
                        "therapeutic_strategies": {
                            "symptomatic": "L-DOPA replacement, DBS for motor control",
                            "neuroprotective": "Antioxidants, anti-inflammatory agents",
                            "disease_modifying": "Alpha-synuclein targeted therapies",
                            "non_pharmacological": "Exercise, cognitive training"
                        }
                    }
                ),
                metadata={
                    "category": "neurological_disorders",
                    "difficulty": "expert",
                    "domain": "movement_disorders"
                }
            )
        ]
        
        disorder_task = EvalTask(
            id="neuroscience_disorder_analysis",
            name="Neurological Disorder Pathophysiology",
            description="Evaluate understanding of neurological disease mechanisms and treatments",
            test_cases=disorder_cases,
            input_format="neurological_disorder_data",
            output_format="pathophysiology_analysis",
            grading_criteria=[
                {
                    "metric": "molecular_pathology_understanding",
                    "weight": 0.3,
                    "parameters": {"focus": "cellular_mechanisms"}
                },
                {
                    "metric": "circuit_dysfunction_analysis",
                    "weight": 0.4,
                    "parameters": {"focus": "network_pathology"}
                },
                {
                    "metric": "therapeutic_rationale",
                    "weight": 0.3,
                    "parameters": {"focus": "treatment_mechanisms"}
                }
            ]
        )
        
        return [disorder_task]


# Register neuroscience-specific graders
@register_grader("neural_circuit_accuracy")
def neural_circuit_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of neural circuit analysis"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check circuit anatomy understanding
        anatomy_score = 0.0
        if "circuit_anatomy" in expected_content:
            anatomy = expected_content["circuit_anatomy"]
            correct_regions = 0
            for region, function in anatomy.items():
                if region.replace("_", " ") in response_lower:
                    correct_regions += 1
                    # Check function understanding
                    function_terms = function.lower().split()
                    if any(term in response_lower for term in function_terms):
                        correct_regions += 0.5
            anatomy_score = min(correct_regions / len(anatomy), 1.0) if anatomy else 0.0
        
        # Check neurotransmitter understanding
        neurotransmitter_score = 0.0
        if "neurotransmitter_functions" in expected_content:
            neurotransmitters = expected_content["neurotransmitter_functions"]
            correct_nt = 0
            for nt, function in neurotransmitters.items():
                if nt in response_lower:
                    correct_nt += 1
                    if any(word in response_lower for word in function.lower().split()):
                        correct_nt += 0.5
            neurotransmitter_score = min(correct_nt / len(neurotransmitters), 1.0) if neurotransmitters else 0.0
        
        # Check pathophysiology understanding
        pathology_score = 0.0
        pathology_keys = ["addiction_neuroadaptations", "consolidation_timeline", "molecular_pathology"]
        for key in pathology_keys:
            if key in expected_content:
                pathology_data = expected_content[key]
                if isinstance(pathology_data, dict):
                    matches = sum(1 for concept in pathology_data.keys() 
                                if concept.replace("_", " ") in response_lower)
                    pathology_score = matches / len(pathology_data) if pathology_data else 0.0
                break
        
        return (anatomy_score * 0.4 + neurotransmitter_score * 0.3 + pathology_score * 0.3)
        
    except Exception:
        return 0.0


@register_grader("cognitive_profile_accuracy")
def cognitive_profile_accuracy(response: str, expected: Dict[str, Any], **kwargs) -> float:
    """Evaluate accuracy of cognitive profile interpretation"""
    try:
        response_lower = response.lower()
        expected_content = expected.get("content", {})
        
        # Check cognitive profile understanding
        profile_score = 0.0
        if "cognitive_profile" in expected_content:
            profile = expected_content["cognitive_profile"]
            domain_matches = 0
            total_domains = 0
            
            for domain, details in profile.items():
                total_domains += 1
                if domain.replace("_", " ") in response_lower:
                    domain_matches += 1
                    # Check for severity/pattern understanding
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if isinstance(value, str) and any(word in response_lower for word in value.lower().split()):
                                domain_matches += 0.3
            
            profile_score = domain_matches / total_domains if total_domains > 0 else 0.0
        
        # Check neuroanatomical correlations
        correlation_score = 0.0
        if "neuroanatomical_correlations" in expected_content:
            correlations = expected_content["neuroanatomical_correlations"]
            correlation_matches = 0
            for region, explanation in correlations.items():
                if region.replace("_", " ") in response_lower:
                    correlation_matches += 1
            correlation_score = correlation_matches / len(correlations) if correlations else 0.0
        
        # Check diagnostic reasoning
        diagnostic_score = 0.0
        if "diagnostic_interpretation" in expected_content:
            diagnostic = expected_content["diagnostic_interpretation"]
            if "primary_diagnosis" in diagnostic:
                diagnosis = diagnostic["primary_diagnosis"].lower()
                if any(word in response_lower for word in diagnosis.split()):
                    diagnostic_score = 0.8
        
        return (profile_score * 0.4 + correlation_score * 0.3 + diagnostic_score * 0.3)
        
    except Exception:
        return 0.0


def create_neuroscience_evaluation_suite() -> EvalSuiteConfig:
    """Create comprehensive neuroscience evaluation suite configuration"""
    suite = NeuroscienceEvaluationSuite()
    
    tasks = []
    tasks.extend(suite.get_neural_circuit_tasks())
    tasks.extend(suite.get_cognitive_assessment_tasks())
    tasks.extend(suite.get_neurological_disorder_tasks())
    
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
        id="neuroscience_comprehensive",
        name="Comprehensive Neuroscience & Cognitive Assessment",
        description="Complete evaluation of neuroscience, cognitive assessment, and neurological disorders",
        category="neuroscience",
        tasks=task_refs,
        metadata={
            "version": "2.0.0",
            "domains": ["neural_circuits", "cognitive_assessment", "neurological_disorders"],
            "difficulty": "expert",
            "estimated_time_minutes": 140
        }
    )


# Export key components
__all__ = [
    "NeuroscienceEvaluationSuite",
    "NeuralCircuit",
    "NeuroimagingData",
    "CognitiveTask",
    "NeurologicalDisorder",
    "create_neuroscience_evaluation_suite",
    "neural_circuit_accuracy",
    "cognitive_profile_accuracy"
] 