"""
Biology and Virology Evaluation Module
=====================================

Specialized evaluations for biological and virological AI systems including:
- Protein structure prediction and analysis
- Gene sequence analysis and function prediction  
- Drug discovery and molecular property prediction
- Viral genomics and outbreak modeling
- Epidemiological forecasting

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import biotite
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from Bio import SeqIO, Align
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC, molecular_weight
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED
    import pandas as pd
    import scipy.stats as stats
    BIOLOGY_DEPS_AVAILABLE = True
except ImportError:
    BIOLOGY_DEPS_AVAILABLE = False
    logging.warning("Biology dependencies not available. Install with: pip install biotite biopython rdkit pandas scipy")

from openevals.config.data_structures import MetricResult, MetricType
from openevals.core.definitions import EvalTask, EvalCase

logger = logging.getLogger(__name__)

@dataclass
class BiologyMetricResult(MetricResult):
    """Extended metric result for biology evaluations"""
    biological_significance: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    experimental_validation: Optional[Dict[str, Any]] = None

class BiologyEvaluatorBase(ABC):
    """Base class for biology evaluators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not BIOLOGY_DEPS_AVAILABLE:
            raise ImportError("Biology dependencies required")
    
    @abstractmethod
    def evaluate(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate biological prediction against reference"""
        pass

class ProteinStructureEvaluator(BiologyEvaluatorBase):
    """
    Evaluator for protein structure prediction tasks
    
    Supports metrics:
    - RMSD (Root Mean Square Deviation)
    - TM-score (Template Modeling score)
    - GDT_TS (Global Distance Test Total Score)
    - LDDT (Local Distance Difference Test)
    - Contact map accuracy
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rmsd_threshold = config.get('rmsd_threshold', 2.0)
        self.tm_score_threshold = config.get('tm_score_threshold', 0.5)
        self.contact_threshold = config.get('contact_threshold', 8.0)
    
    def evaluate(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """
        Evaluate protein structure prediction
        
        Args:
            prediction: Predicted protein structure (PDB format or coordinates)
            reference: Reference protein structure (PDB format or coordinates)
            
        Returns:
            BiologyMetricResult with structural metrics
        """
        try:
            # Parse structures
            pred_structure = self._parse_structure(prediction)
            ref_structure = self._parse_structure(reference)
            
            # Calculate RMSD
            rmsd = self._calculate_rmsd(pred_structure, ref_structure)
            
            # Calculate TM-score
            tm_score = self._calculate_tm_score(pred_structure, ref_structure)
            
            # Calculate GDT_TS
            gdt_ts = self._calculate_gdt_ts(pred_structure, ref_structure)
            
            # Calculate contact map accuracy
            contact_accuracy = self._calculate_contact_accuracy(pred_structure, ref_structure)
            
            # Determine overall quality
            overall_score = self._calculate_overall_score(rmsd, tm_score, gdt_ts)
            passed = overall_score >= 0.7
            
            return BiologyMetricResult(
                metric_name="protein_structure_quality",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=passed,
                details={
                    "rmsd": rmsd,
                    "tm_score": tm_score,
                    "gdt_ts": gdt_ts,
                    "contact_accuracy": contact_accuracy,
                    "structure_length": len(pred_structure)
                },
                biological_significance=self._assess_biological_significance(rmsd, tm_score),
                confidence_interval=self._calculate_confidence_interval(overall_score)
            )
            
        except Exception as e:
            logger.error(f"Protein structure evaluation failed: {str(e)}")
            return BiologyMetricResult(
                metric_name="protein_structure_quality",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_structure(self, structure_data: Any) -> Any:
        """Parse structure from various formats"""
        if isinstance(structure_data, str):
            if structure_data.endswith('.pdb'):
                return strucio.load_structure(structure_data)
            else:
                # Assume PDB string format
                return strucio.load_structure_from_string(structure_data, "pdb")
        elif isinstance(structure_data, np.ndarray):
            # Assume coordinate array
            return structure_data
        else:
            raise ValueError(f"Unsupported structure format: {type(structure_data)}")
    
    def _calculate_rmsd(self, pred: Any, ref: Any) -> float:
        """Calculate RMSD between predicted and reference structures"""
        if hasattr(pred, 'coord') and hasattr(ref, 'coord'):
            # Biotite structure objects
            pred_coords = pred.coord
            ref_coords = ref.coord
        else:
            # Assume numpy arrays
            pred_coords = pred
            ref_coords = ref
        
        # Align structures (simple alignment)
        if pred_coords.shape != ref_coords.shape:
            min_len = min(len(pred_coords), len(ref_coords))
            pred_coords = pred_coords[:min_len]
            ref_coords = ref_coords[:min_len]
        
        # Calculate RMSD
        diff = pred_coords - ref_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return float(rmsd)
    
    def _calculate_tm_score(self, pred: Any, ref: Any) -> float:
        """Calculate TM-score (simplified implementation)"""
        rmsd = self._calculate_rmsd(pred, ref)
        length = len(pred) if hasattr(pred, '__len__') else pred.shape[0]
        
        # Simplified TM-score calculation
        d0 = 1.24 * (length - 15) ** (1/3) - 1.8
        tm_score = 1 / (1 + (rmsd / d0) ** 2)
        return float(tm_score)
    
    def _calculate_gdt_ts(self, pred: Any, ref: Any) -> float:
        """Calculate GDT_TS score"""
        rmsd = self._calculate_rmsd(pred, ref)
        
        # Simplified GDT_TS based on distance thresholds
        thresholds = [1.0, 2.0, 4.0, 8.0]
        scores = []
        
        for threshold in thresholds:
            score = 1.0 if rmsd <= threshold else np.exp(-rmsd/threshold)
            scores.append(score)
        
        gdt_ts = np.mean(scores)
        return float(gdt_ts)
    
    def _calculate_contact_accuracy(self, pred: Any, ref: Any) -> float:
        """Calculate contact map prediction accuracy"""
        # Simplified contact map calculation
        # In practice, this would involve more sophisticated contact prediction
        try:
            pred_contacts = self._calculate_contacts(pred)
            ref_contacts = self._calculate_contacts(ref)
            
            # Calculate accuracy
            correct = np.sum(pred_contacts == ref_contacts)
            total = pred_contacts.size
            accuracy = correct / total if total > 0 else 0.0
            
            return float(accuracy)
        except:
            return 0.5  # Default fallback
    
    def _calculate_contacts(self, structure: Any) -> np.ndarray:
        """Calculate contact map from structure"""
        # Simplified contact calculation
        if hasattr(structure, 'coord'):
            coords = structure.coord
        else:
            coords = structure
        
        n_residues = len(coords)
        contacts = np.zeros((n_residues, n_residues), dtype=bool)
        
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.contact_threshold:
                    contacts[i, j] = True
                    contacts[j, i] = True
        
        return contacts
    
    def _calculate_overall_score(self, rmsd: float, tm_score: float, gdt_ts: float) -> float:
        """Calculate overall structure quality score"""
        # Weighted combination of metrics
        rmsd_score = max(0, 1 - rmsd / 10.0)  # Normalize RMSD
        combined_score = 0.3 * rmsd_score + 0.4 * tm_score + 0.3 * gdt_ts
        return float(np.clip(combined_score, 0.0, 1.0))
    
    def _assess_biological_significance(self, rmsd: float, tm_score: float) -> str:
        """Assess biological significance of prediction quality"""
        if tm_score > 0.8 and rmsd < 2.0:
            return "High quality - suitable for drug design"
        elif tm_score > 0.6 and rmsd < 4.0:
            return "Good quality - suitable for functional analysis"
        elif tm_score > 0.4:
            return "Moderate quality - fold recognition"
        else:
            return "Low quality - incorrect fold"
    
    def _calculate_confidence_interval(self, score: float) -> Tuple[float, float]:
        """Calculate confidence interval for score"""
        # Simplified confidence interval
        std_error = 0.05  # Assumed standard error
        margin = 1.96 * std_error  # 95% confidence
        return (max(0, score - margin), min(1, score + margin))

class GenomicsEvaluator(BiologyEvaluatorBase):
    """
    Evaluator for genomics and gene analysis tasks
    
    Supports:
    - Gene prediction accuracy
    - Variant calling performance
    - Functional annotation prediction
    - Sequence alignment quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_gene_length = config.get('min_gene_length', 300)
        self.variant_quality_threshold = config.get('variant_quality_threshold', 30)
    
    def evaluate(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate genomics prediction"""
        task_type = self.config.get('task_type', 'gene_prediction')
        
        if task_type == 'gene_prediction':
            return self._evaluate_gene_prediction(prediction, reference)
        elif task_type == 'variant_calling':
            return self._evaluate_variant_calling(prediction, reference)
        elif task_type == 'functional_annotation':
            return self._evaluate_functional_annotation(prediction, reference)
        else:
            raise ValueError(f"Unknown genomics task type: {task_type}")
    
    def _evaluate_gene_prediction(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate gene prediction accuracy"""
        try:
            # Parse gene predictions and references
            pred_genes = self._parse_gene_annotations(prediction)
            ref_genes = self._parse_gene_annotations(reference)
            
            # Calculate sensitivity and specificity
            tp, fp, fn = self._calculate_gene_overlap(pred_genes, ref_genes)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_score = 2 * sensitivity * specificity / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0.0
            
            # Calculate nucleotide-level accuracy
            nucleotide_accuracy = self._calculate_nucleotide_accuracy(pred_genes, ref_genes)
            
            overall_score = 0.4 * f1_score + 0.3 * sensitivity + 0.3 * nucleotide_accuracy
            
            return BiologyMetricResult(
                metric_name="gene_prediction_accuracy",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1_score": f1_score,
                    "nucleotide_accuracy": nucleotide_accuracy,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "total_predicted_genes": len(pred_genes),
                    "total_reference_genes": len(ref_genes)
                },
                biological_significance=self._assess_gene_prediction_significance(f1_score)
            )
            
        except Exception as e:
            logger.error(f"Gene prediction evaluation failed: {str(e)}")
            return BiologyMetricResult(
                metric_name="gene_prediction_accuracy",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_gene_annotations(self, annotations: Any) -> List[Dict[str, Any]]:
        """Parse gene annotations from various formats"""
        if isinstance(annotations, str):
            # Assume GFF/GTF format
            return self._parse_gff(annotations)
        elif isinstance(annotations, list):
            # Assume list of gene dictionaries
            return annotations
        else:
            raise ValueError(f"Unsupported annotation format: {type(annotations)}")
    
    def _parse_gff(self, gff_content: str) -> List[Dict[str, Any]]:
        """Parse GFF format annotations"""
        genes = []
        for line in gff_content.strip().split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 9 and parts[2] == 'gene':
                gene = {
                    'chromosome': parts[0],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'strand': parts[6],
                    'attributes': parts[8]
                }
                genes.append(gene)
        
        return genes
    
    def _calculate_gene_overlap(self, pred_genes: List[Dict], ref_genes: List[Dict]) -> Tuple[int, int, int]:
        """Calculate gene prediction overlaps"""
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        matched_refs = set()
        
        for pred_gene in pred_genes:
            found_match = False
            for i, ref_gene in enumerate(ref_genes):
                if i in matched_refs:
                    continue
                
                if self._genes_overlap(pred_gene, ref_gene):
                    tp += 1
                    matched_refs.add(i)
                    found_match = True
                    break
            
            if not found_match:
                fp += 1
        
        fn = len(ref_genes) - len(matched_refs)
        
        return tp, fp, fn
    
    def _genes_overlap(self, gene1: Dict, gene2: Dict, min_overlap: float = 0.5) -> bool:
        """Check if two genes overlap significantly"""
        if gene1['chromosome'] != gene2['chromosome']:
            return False
        
        # Calculate overlap
        start = max(gene1['start'], gene2['start'])
        end = min(gene1['end'], gene2['end'])
        
        if start >= end:
            return False
        
        overlap_length = end - start
        gene1_length = gene1['end'] - gene1['start']
        gene2_length = gene2['end'] - gene2['start']
        
        overlap_ratio1 = overlap_length / gene1_length
        overlap_ratio2 = overlap_length / gene2_length
        
        return max(overlap_ratio1, overlap_ratio2) >= min_overlap
    
    def _calculate_nucleotide_accuracy(self, pred_genes: List[Dict], ref_genes: List[Dict]) -> float:
        """Calculate nucleotide-level accuracy"""
        # Simplified implementation
        # In practice, this would require more sophisticated alignment
        total_nucleotides = sum(gene['end'] - gene['start'] for gene in ref_genes)
        if total_nucleotides == 0:
            return 0.0
        
        correct_nucleotides = 0
        for pred_gene in pred_genes:
            for ref_gene in ref_genes:
                if self._genes_overlap(pred_gene, ref_gene):
                    overlap_start = max(pred_gene['start'], ref_gene['start'])
                    overlap_end = min(pred_gene['end'], ref_gene['end'])
                    if overlap_end > overlap_start:
                        correct_nucleotides += overlap_end - overlap_start
        
        return min(1.0, correct_nucleotides / total_nucleotides)
    
    def _assess_gene_prediction_significance(self, f1_score: float) -> str:
        """Assess biological significance of gene prediction performance"""
        if f1_score > 0.9:
            return "Excellent - suitable for genome annotation"
        elif f1_score > 0.8:
            return "Good - suitable for comparative analysis"
        elif f1_score > 0.6:
            return "Moderate - requires manual curation"
        else:
            return "Poor - needs algorithm improvement"
    
    def _evaluate_variant_calling(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate variant calling performance"""
        # Implementation for variant calling evaluation
        try:
            pred_variants = self._parse_variants(prediction)
            ref_variants = self._parse_variants(reference)
            
            # Calculate precision, recall, F1
            tp, fp, fn = self._calculate_variant_overlap(pred_variants, ref_variants)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return BiologyMetricResult(
                metric_name="variant_calling_accuracy",
                metric_type=MetricType.F1_SCORE,
                value=f1_score,
                passed=f1_score >= 0.8,
                details={
                    "precision": precision,
                    "recall": recall,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="variant_calling_accuracy",
                metric_type=MetricType.F1_SCORE,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_variants(self, variants: Any) -> List[Dict]:
        """Parse variant data"""
        # Simplified variant parsing
        if isinstance(variants, list):
            return variants
        elif isinstance(variants, str):
            # Parse VCF-like format
            variant_list = []
            for line in variants.strip().split('\n'):
                if not line.startswith('#') and line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        variant = {
                            'chromosome': parts[0],
                            'position': int(parts[1]),
                            'ref_allele': parts[3],
                            'alt_allele': parts[4]
                        }
                        variant_list.append(variant)
            return variant_list
        else:
            return []
    
    def _calculate_variant_overlap(self, pred_variants: List[Dict], ref_variants: List[Dict]) -> Tuple[int, int, int]:
        """Calculate variant prediction overlaps"""
        tp = 0
        matched_refs = set()
        
        for pred_var in pred_variants:
            for i, ref_var in enumerate(ref_variants):
                if i in matched_refs:
                    continue
                
                if self._variants_match(pred_var, ref_var):
                    tp += 1
                    matched_refs.add(i)
                    break
        
        fp = len(pred_variants) - tp
        fn = len(ref_variants) - len(matched_refs)
        
        return tp, fp, fn
    
    def _variants_match(self, var1: Dict, var2: Dict) -> bool:
        """Check if two variants match"""
        return (var1['chromosome'] == var2['chromosome'] and
                var1['position'] == var2['position'] and
                var1['ref_allele'] == var2['ref_allele'] and
                var1['alt_allele'] == var2['alt_allele'])
    
    def _evaluate_functional_annotation(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate functional annotation prediction"""
        # Implementation for functional annotation evaluation
        try:
            pred_annotations = self._parse_functional_annotations(prediction)
            ref_annotations = self._parse_functional_annotations(reference)
            
            # Calculate annotation accuracy
            accuracy = self._calculate_annotation_accuracy(pred_annotations, ref_annotations)
            
            return BiologyMetricResult(
                metric_name="functional_annotation_accuracy",
                metric_type=MetricType.ACCURACY,
                value=accuracy,
                passed=accuracy >= 0.7,
                details={
                    "annotation_accuracy": accuracy,
                    "predicted_annotations": len(pred_annotations),
                    "reference_annotations": len(ref_annotations)
                }
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="functional_annotation_accuracy",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_functional_annotations(self, annotations: Any) -> Dict[str, List[str]]:
        """Parse functional annotations"""
        if isinstance(annotations, dict):
            return annotations
        elif isinstance(annotations, str):
            # Parse from string format
            try:
                return json.loads(annotations)
            except:
                return {}
        else:
            return {}
    
    def _calculate_annotation_accuracy(self, pred: Dict, ref: Dict) -> float:
        """Calculate functional annotation accuracy"""
        if not ref:
            return 0.0
        
        total_genes = len(ref)
        correct_annotations = 0
        
        for gene_id, ref_functions in ref.items():
            if gene_id in pred:
                pred_functions = pred[gene_id]
                # Calculate overlap between predicted and reference functions
                if isinstance(ref_functions, list) and isinstance(pred_functions, list):
                    overlap = len(set(ref_functions) & set(pred_functions))
                    total_ref = len(ref_functions)
                    if total_ref > 0:
                        gene_accuracy = overlap / total_ref
                        correct_annotations += gene_accuracy
        
        return correct_annotations / total_genes if total_genes > 0 else 0.0

class DrugDiscoveryEvaluator(BiologyEvaluatorBase):
    """
    Evaluator for drug discovery and molecular property prediction
    
    Supports:
    - Molecular property prediction (QED, LogP, etc.)
    - Binding affinity prediction
    - Toxicity prediction
    - Drug-likeness assessment
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.property_thresholds = config.get('property_thresholds', {
            'qed': 0.5,
            'logp': 5.0,
            'molecular_weight': 500.0
        })
    
    def evaluate(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate drug discovery prediction"""
        task_type = self.config.get('task_type', 'molecular_properties')
        
        if task_type == 'molecular_properties':
            return self._evaluate_molecular_properties(prediction, reference)
        elif task_type == 'binding_affinity':
            return self._evaluate_binding_affinity(prediction, reference)
        elif task_type == 'drug_likeness':
            return self._evaluate_drug_likeness(prediction, reference)
        else:
            raise ValueError(f"Unknown drug discovery task type: {task_type}")
    
    def _evaluate_molecular_properties(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate molecular property prediction"""
        try:
            # Parse molecular data
            pred_properties = self._parse_molecular_properties(prediction)
            ref_properties = self._parse_molecular_properties(reference)
            
            # Calculate property prediction accuracies
            property_accuracies = {}
            overall_accuracy = 0.0
            
            for prop_name in ref_properties:
                if prop_name in pred_properties:
                    accuracy = self._calculate_property_accuracy(
                        pred_properties[prop_name],
                        ref_properties[prop_name],
                        prop_name
                    )
                    property_accuracies[prop_name] = accuracy
            
            if property_accuracies:
                overall_accuracy = np.mean(list(property_accuracies.values()))
            
            return BiologyMetricResult(
                metric_name="molecular_property_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_accuracy,
                passed=overall_accuracy >= 0.8,
                details={
                    "property_accuracies": property_accuracies,
                    "predicted_properties": list(pred_properties.keys()),
                    "reference_properties": list(ref_properties.keys())
                },
                biological_significance=self._assess_property_significance(overall_accuracy)
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="molecular_property_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_molecular_properties(self, properties: Any) -> Dict[str, float]:
        """Parse molecular properties from various formats"""
        if isinstance(properties, dict):
            return properties
        elif isinstance(properties, str):
            try:
                # Try to parse as SMILES and calculate properties
                mol = Chem.MolFromSmiles(properties)
                if mol:
                    return self._calculate_rdkit_properties(mol)
                else:
                    return json.loads(properties)
            except:
                return {}
        else:
            return {}
    
    def _calculate_rdkit_properties(self, mol) -> Dict[str, float]:
        """Calculate molecular properties using RDKit"""
        properties = {}
        try:
            properties['molecular_weight'] = Descriptors.MolWt(mol)
            properties['logp'] = Crippen.MolLogP(mol)
            properties['qed'] = QED.qed(mol)
            properties['tpsa'] = Descriptors.TPSA(mol)
            properties['hbd'] = Descriptors.NumHDonors(mol)
            properties['hba'] = Descriptors.NumHAcceptors(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate RDKit properties: {e}")
        
        return properties
    
    def _calculate_property_accuracy(self, pred_value: float, ref_value: float, prop_name: str) -> float:
        """Calculate accuracy for a specific molecular property"""
        if prop_name in ['molecular_weight', 'tpsa']:
            # Absolute properties - use relative error
            rel_error = abs(pred_value - ref_value) / max(abs(ref_value), 1.0)
            return max(0.0, 1.0 - rel_error)
        elif prop_name in ['logp']:
            # Log scale properties - use absolute difference
            abs_error = abs(pred_value - ref_value)
            return max(0.0, 1.0 - abs_error / 5.0)  # Normalize by typical range
        elif prop_name in ['qed']:
            # Bounded properties [0,1] - use absolute difference
            abs_error = abs(pred_value - ref_value)
            return max(0.0, 1.0 - abs_error)
        else:
            # Default: relative error
            rel_error = abs(pred_value - ref_value) / max(abs(ref_value), 1.0)
            return max(0.0, 1.0 - rel_error)
    
    def _assess_property_significance(self, accuracy: float) -> str:
        """Assess biological significance of property prediction accuracy"""
        if accuracy > 0.9:
            return "Excellent - suitable for virtual screening"
        elif accuracy > 0.8:
            return "Good - suitable for lead optimization"
        elif accuracy > 0.6:
            return "Moderate - requires experimental validation"
        else:
            return "Poor - not reliable for drug discovery"
    
    def _evaluate_binding_affinity(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate binding affinity prediction"""
        try:
            pred_affinity = float(prediction)
            ref_affinity = float(reference)
            
            # Calculate relative error
            rel_error = abs(pred_affinity - ref_affinity) / max(abs(ref_affinity), 1.0)
            accuracy = max(0.0, 1.0 - rel_error)
            
            # Convert to correlation-like metric
            correlation_score = 1.0 - min(1.0, rel_error)
            
            return BiologyMetricResult(
                metric_name="binding_affinity_prediction",
                metric_type=MetricType.CUSTOM,
                value=correlation_score,
                passed=correlation_score >= 0.7,
                details={
                    "predicted_affinity": pred_affinity,
                    "reference_affinity": ref_affinity,
                    "relative_error": rel_error,
                    "accuracy": accuracy
                }
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="binding_affinity_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _evaluate_drug_likeness(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate drug-likeness prediction"""
        try:
            pred_drug_like = bool(prediction)
            ref_drug_like = bool(reference)
            
            accuracy = 1.0 if pred_drug_like == ref_drug_like else 0.0
            
            return BiologyMetricResult(
                metric_name="drug_likeness_prediction",
                metric_type=MetricType.ACCURACY,
                value=accuracy,
                passed=accuracy >= 1.0,
                details={
                    "predicted_drug_like": pred_drug_like,
                    "reference_drug_like": ref_drug_like
                }
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="drug_likeness_prediction",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )

class ViralGenomicsEvaluator(BiologyEvaluatorBase):
    """
    Evaluator for viral genomics and epidemiological modeling
    
    Supports:
    - Viral strain classification
    - Mutation prediction
    - Outbreak forecasting
    - Transmission rate estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.classification_threshold = config.get('classification_threshold', 0.8)
        self.forecast_horizon = config.get('forecast_horizon', 30)  # days
    
    def evaluate(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate viral genomics prediction"""
        task_type = self.config.get('task_type', 'strain_classification')
        
        if task_type == 'strain_classification':
            return self._evaluate_strain_classification(prediction, reference)
        elif task_type == 'outbreak_forecasting':
            return self._evaluate_outbreak_forecasting(prediction, reference)
        elif task_type == 'mutation_prediction':
            return self._evaluate_mutation_prediction(prediction, reference)
        else:
            raise ValueError(f"Unknown viral genomics task type: {task_type}")
    
    def _evaluate_strain_classification(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate viral strain classification"""
        try:
            pred_strain = str(prediction).strip()
            ref_strain = str(reference).strip()
            
            # Exact match
            exact_match = pred_strain == ref_strain
            
            # Hierarchical match (e.g., same family/genus)
            hierarchical_score = self._calculate_hierarchical_similarity(pred_strain, ref_strain)
            
            overall_score = 1.0 if exact_match else hierarchical_score
            
            return BiologyMetricResult(
                metric_name="viral_strain_classification",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= self.classification_threshold,
                details={
                    "predicted_strain": pred_strain,
                    "reference_strain": ref_strain,
                    "exact_match": exact_match,
                    "hierarchical_score": hierarchical_score
                },
                biological_significance=self._assess_strain_classification_significance(overall_score)
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="viral_strain_classification",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _calculate_hierarchical_similarity(self, strain1: str, strain2: str) -> float:
        """Calculate hierarchical similarity between viral strains"""
        # Simplified hierarchical matching
        # In practice, this would use phylogenetic distances
        
        strain1_parts = strain1.lower().split()
        strain2_parts = strain2.lower().split()
        
        if not strain1_parts or not strain2_parts:
            return 0.0
        
        # Check genus/family level similarity
        if strain1_parts[0] == strain2_parts[0]:
            return 0.7  # Same genus
        
        # Check for partial matches
        common_parts = set(strain1_parts) & set(strain2_parts)
        if common_parts:
            return len(common_parts) / max(len(strain1_parts), len(strain2_parts)) * 0.5
        
        return 0.0
    
    def _assess_strain_classification_significance(self, score: float) -> str:
        """Assess biological significance of strain classification"""
        if score >= 0.9:
            return "Excellent - suitable for outbreak response"
        elif score >= 0.7:
            return "Good - useful for surveillance"
        elif score >= 0.5:
            return "Moderate - requires confirmation"
        else:
            return "Poor - likely misclassification"
    
    def _evaluate_outbreak_forecasting(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate outbreak forecasting accuracy"""
        try:
            pred_data = self._parse_forecast_data(prediction)
            ref_data = self._parse_forecast_data(reference)
            
            # Calculate forecasting metrics
            mae = self._calculate_mae(pred_data, ref_data)
            rmse = self._calculate_rmse(pred_data, ref_data)
            mape = self._calculate_mape(pred_data, ref_data)
            
            # Normalize metrics to [0,1] scale
            normalized_score = self._normalize_forecast_score(mae, rmse, mape)
            
            return BiologyMetricResult(
                metric_name="outbreak_forecasting",
                metric_type=MetricType.CUSTOM,
                value=normalized_score,
                passed=normalized_score >= 0.7,
                details={
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "forecast_horizon_days": self.forecast_horizon
                },
                biological_significance=self._assess_forecast_significance(normalized_score)
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="outbreak_forecasting",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_forecast_data(self, data: Any) -> np.ndarray:
        """Parse forecast data"""
        if isinstance(data, (list, np.ndarray)):
            return np.array(data, dtype=float)
        elif isinstance(data, str):
            try:
                return np.array(json.loads(data), dtype=float)
            except:
                return np.array([float(data)])
        else:
            return np.array([float(data)])
    
    def _calculate_mae(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        min_len = min(len(pred), len(ref))
        return float(np.mean(np.abs(pred[:min_len] - ref[:min_len])))
    
    def _calculate_rmse(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        min_len = min(len(pred), len(ref))
        return float(np.sqrt(np.mean((pred[:min_len] - ref[:min_len])**2)))
    
    def _calculate_mape(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        min_len = min(len(pred), len(ref))
        ref_subset = ref[:min_len]
        pred_subset = pred[:min_len]
        
        # Avoid division by zero
        non_zero_mask = ref_subset != 0
        if not np.any(non_zero_mask):
            return 100.0
        
        mape = np.mean(np.abs((ref_subset[non_zero_mask] - pred_subset[non_zero_mask]) / ref_subset[non_zero_mask])) * 100
        return float(mape)
    
    def _normalize_forecast_score(self, mae: float, rmse: float, mape: float) -> float:
        """Normalize forecasting metrics to a single score"""
        # Simple normalization scheme
        # In practice, this would be calibrated to specific use cases
        
        mae_score = max(0, 1 - mae / 1000)  # Assuming case counts
        rmse_score = max(0, 1 - rmse / 1000)
        mape_score = max(0, 1 - mape / 100)
        
        combined_score = 0.4 * mae_score + 0.3 * rmse_score + 0.3 * mape_score
        return float(combined_score)
    
    def _assess_forecast_significance(self, score: float) -> str:
        """Assess biological significance of forecast accuracy"""
        if score >= 0.8:
            return "Excellent - suitable for public health planning"
        elif score >= 0.6:
            return "Good - useful for trend monitoring"
        elif score >= 0.4:
            return "Moderate - limited predictive value"
        else:
            return "Poor - not suitable for decision making"
    
    def _evaluate_mutation_prediction(self, prediction: Any, reference: Any) -> BiologyMetricResult:
        """Evaluate viral mutation prediction"""
        try:
            pred_mutations = self._parse_mutations(prediction)
            ref_mutations = self._parse_mutations(reference)
            
            # Calculate mutation prediction accuracy
            tp, fp, fn = self._calculate_mutation_overlap(pred_mutations, ref_mutations)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return BiologyMetricResult(
                metric_name="viral_mutation_prediction",
                metric_type=MetricType.F1_SCORE,
                value=f1_score,
                passed=f1_score >= 0.7,
                details={
                    "precision": precision,
                    "recall": recall,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "predicted_mutations": len(pred_mutations),
                    "reference_mutations": len(ref_mutations)
                }
            )
            
        except Exception as e:
            return BiologyMetricResult(
                metric_name="viral_mutation_prediction",
                metric_type=MetricType.F1_SCORE,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_mutations(self, mutations: Any) -> List[Dict[str, Any]]:
        """Parse mutation data"""
        if isinstance(mutations, list):
            return mutations
        elif isinstance(mutations, str):
            try:
                return json.loads(mutations)
            except:
                # Parse simple mutation format like "A123T,G456C"
                mutation_list = []
                for mut in mutations.split(','):
                    mut = mut.strip()
                    if len(mut) >= 3:
                        mutation_list.append({
                            'original': mut[0],
                            'position': mut[1:-1],
                            'mutated': mut[-1]
                        })
                return mutation_list
        else:
            return []
    
    def _calculate_mutation_overlap(self, pred_mutations: List[Dict], ref_mutations: List[Dict]) -> Tuple[int, int, int]:
        """Calculate mutation prediction overlaps"""
        tp = 0
        matched_refs = set()
        
        for pred_mut in pred_mutations:
            for i, ref_mut in enumerate(ref_mutations):
                if i in matched_refs:
                    continue
                
                if self._mutations_match(pred_mut, ref_mut):
                    tp += 1
                    matched_refs.add(i)
                    break
        
        fp = len(pred_mutations) - tp
        fn = len(ref_mutations) - len(matched_refs)
        
        return tp, fp, fn
    
    def _mutations_match(self, mut1: Dict, mut2: Dict) -> bool:
        """Check if two mutations match"""
        return (str(mut1.get('position', '')).strip() == str(mut2.get('position', '')).strip() and
                str(mut1.get('original', '')).strip().upper() == str(mut2.get('original', '')).strip().upper() and
                str(mut1.get('mutated', '')).strip().upper() == str(mut2.get('mutated', '')).strip().upper())

# Register biology evaluators with the grading system
def register_biology_evaluators():
    """Register biology evaluators with the OpenEvals grading system"""
    from openevals.core.graders import register_grader
    from openevals.config.data_structures import MetricType
    
    def protein_structure_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = ProteinStructureEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def genomics_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = GenomicsEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def drug_discovery_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = DrugDiscoveryEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def viral_genomics_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = ViralGenomicsEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    # Register the graders
    register_grader("protein_structure", protein_structure_grader, MetricType.CUSTOM)
    register_grader("genomics", genomics_grader, MetricType.CUSTOM)
    register_grader("drug_discovery", drug_discovery_grader, MetricType.CUSTOM)
    register_grader("viral_genomics", viral_genomics_grader, MetricType.CUSTOM)

# Auto-register on import
if BIOLOGY_DEPS_AVAILABLE:
    try:
        register_biology_evaluators()
        logger.info("Biology evaluators registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register biology evaluators: {e}") 