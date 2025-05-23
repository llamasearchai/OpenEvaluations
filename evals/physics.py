"""
Physics and Nuclear Fusion Evaluation Module
===========================================

Specialized evaluations for physics and nuclear fusion AI systems including:
- Plasma physics simulation and analysis
- Fusion energy system modeling
- Materials science for extreme environments
- Radiation damage prediction
- Thermal and structural analysis
- Magnetic confinement modeling

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

try:
    import scipy
    import scipy.constants as const
    import scipy.integrate as integrate
    import scipy.optimize as optimize
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    import pandas as pd
    PHYSICS_DEPS_AVAILABLE = True
except ImportError:
    PHYSICS_DEPS_AVAILABLE = False
    logging.warning("Physics dependencies not available. Install with: pip install scipy matplotlib pandas")

from openevals.config.data_structures import MetricResult, MetricType
from openevals.core.definitions import EvalTask, EvalCase

logger = logging.getLogger(__name__)

@dataclass
class PhysicsMetricResult(MetricResult):
    """Extended metric result for physics evaluations"""
    physical_significance: Optional[str] = None
    uncertainty_analysis: Optional[Dict[str, float]] = None
    dimensional_analysis: Optional[Dict[str, str]] = None
    convergence_metrics: Optional[Dict[str, float]] = None

class PhysicsEvaluatorBase(ABC):
    """Base class for physics evaluators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not PHYSICS_DEPS_AVAILABLE:
            raise ImportError("Physics dependencies required")
        
        # Physics constants
        self.constants = {
            'c': const.c,  # Speed of light
            'e': const.e,  # Elementary charge
            'k_B': const.k,  # Boltzmann constant
            'epsilon_0': const.epsilon_0,  # Vacuum permittivity
            'mu_0': const.mu_0,  # Vacuum permeability
            'm_e': const.m_e,  # Electron mass
            'm_p': const.m_p,  # Proton mass
            'h': const.h,  # Planck constant
            'hbar': const.hbar,  # Reduced Planck constant
        }
    
    @abstractmethod
    def evaluate(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate physics prediction against reference"""
        pass

class FusionPhysicsEvaluator(PhysicsEvaluatorBase):
    """
    Evaluator for fusion physics and plasma modeling tasks
    
    Supports:
    - Plasma confinement analysis
    - Fusion reaction rate prediction
    - Magnetohydrodynamic (MHD) stability analysis
    - Energy balance calculations
    - Disruption prediction
    - Transport modeling
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plasma_regime = config.get('plasma_regime', 'tokamak')
        self.confinement_threshold = config.get('confinement_threshold', 1.0)
        self.temperature_tolerance = config.get('temperature_tolerance', 0.1)  # 10%
        self.density_tolerance = config.get('density_tolerance', 0.15)  # 15%
    
    def evaluate(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate fusion physics prediction"""
        task_type = self.config.get('task_type', 'plasma_confinement')
        
        if task_type == 'plasma_confinement':
            return self._evaluate_plasma_confinement(prediction, reference)
        elif task_type == 'fusion_rate':
            return self._evaluate_fusion_rate(prediction, reference)
        elif task_type == 'mhd_stability':
            return self._evaluate_mhd_stability(prediction, reference)
        elif task_type == 'disruption_prediction':
            return self._evaluate_disruption_prediction(prediction, reference)
        elif task_type == 'transport_modeling':
            return self._evaluate_transport_modeling(prediction, reference)
        else:
            raise ValueError(f"Unknown fusion physics task type: {task_type}")
    
    def _evaluate_plasma_confinement(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma confinement predictions"""
        try:
            pred_data = self._parse_plasma_data(prediction)
            ref_data = self._parse_plasma_data(reference)
            
            # Extract key parameters
            pred_temp = pred_data.get('temperature', 0.0)  # keV
            ref_temp = ref_data.get('temperature', 0.0)
            pred_density = pred_data.get('density', 0.0)  # m^-3
            ref_density = ref_data.get('density', 0.0)
            pred_confinement_time = pred_data.get('confinement_time', 0.0)  # seconds
            ref_confinement_time = ref_data.get('confinement_time', 0.0)
            
            # Calculate relative errors
            temp_error = self._calculate_relative_error(pred_temp, ref_temp)
            density_error = self._calculate_relative_error(pred_density, ref_density)
            confinement_error = self._calculate_relative_error(pred_confinement_time, ref_confinement_time)
            
            # Calculate triple product (nT𝜏)
            pred_triple_product = pred_density * pred_temp * pred_confinement_time
            ref_triple_product = ref_density * ref_temp * ref_confinement_time
            triple_product_error = self._calculate_relative_error(pred_triple_product, ref_triple_product)
            
            # Calculate Lawson criterion assessment
            lawson_score = self._evaluate_lawson_criterion(pred_triple_product, ref_triple_product)
            
            # Overall confinement score
            temp_score = max(0, 1 - temp_error / self.temperature_tolerance)
            density_score = max(0, 1 - density_error / self.density_tolerance)
            confinement_score = max(0, 1 - confinement_error / 0.2)  # 20% tolerance for confinement time
            
            overall_score = 0.3 * temp_score + 0.3 * density_score + 0.2 * confinement_score + 0.2 * lawson_score
            overall_score = np.clip(overall_score, 0.0, 1.0)
            
            return PhysicsMetricResult(
                metric_name="plasma_confinement_quality",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "temperature_error": temp_error,
                    "density_error": density_error,
                    "confinement_time_error": confinement_error,
                    "triple_product_error": triple_product_error,
                    "lawson_score": lawson_score,
                    "predicted_triple_product": pred_triple_product,
                    "reference_triple_product": ref_triple_product,
                    "individual_scores": {
                        "temperature": temp_score,
                        "density": density_score,
                        "confinement": confinement_score
                    }
                },
                physical_significance=self._assess_confinement_significance(overall_score, pred_triple_product),
                uncertainty_analysis=self._calculate_uncertainty_analysis(pred_data, ref_data)
            )
            
        except Exception as e:
            logger.error(f"Plasma confinement evaluation failed: {str(e)}")
            return PhysicsMetricResult(
                metric_name="plasma_confinement_quality",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_plasma_data(self, data: Any) -> Dict[str, float]:
        """Parse plasma physics data from various formats"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                # Try to parse simple format: "T=10.5,n=1e20,tau=1.2"
                params = {}
                for pair in data.split(','):
                    if '=' in pair:
                        key, value = pair.split('=')
                        key = key.strip().lower()
                        # Map common parameter names
                        if key in ['t', 'temp', 'temperature']:
                            params['temperature'] = float(value)
                        elif key in ['n', 'dens', 'density']:
                            params['density'] = float(value)
                        elif key in ['tau', 'time', 'confinement_time']:
                            params['confinement_time'] = float(value)
                        elif key in ['b', 'b_field', 'magnetic_field']:
                            params['magnetic_field'] = float(value)
                        elif key in ['p', 'pressure']:
                            params['pressure'] = float(value)
                return params
        elif isinstance(data, (list, np.ndarray)):
            # Assume ordered parameters: [temperature, density, confinement_time]
            data_array = np.array(data)
            params = {}
            if len(data_array) >= 1:
                params['temperature'] = float(data_array[0])
            if len(data_array) >= 2:
                params['density'] = float(data_array[1])
            if len(data_array) >= 3:
                params['confinement_time'] = float(data_array[2])
            return params
        else:
            return {}
    
    def _calculate_relative_error(self, predicted: float, reference: float) -> float:
        """Calculate relative error between predicted and reference values"""
        if abs(reference) < 1e-10:  # Avoid division by zero
            return abs(predicted - reference)
        return abs(predicted - reference) / abs(reference)
    
    def _evaluate_lawson_criterion(self, pred_triple_product: float, ref_triple_product: float) -> float:
        """Evaluate how well the prediction satisfies the Lawson criterion"""
        # Lawson criterion for D-T fusion: nT𝜏 > 3e21 keV·s·m^-3
        lawson_threshold = 3e21
        
        ref_lawson_ratio = ref_triple_product / lawson_threshold
        pred_lawson_ratio = pred_triple_product / lawson_threshold
        
        # Score based on how close the prediction is to the reference Lawson ratio
        if ref_lawson_ratio <= 0:
            return 0.0
        
        ratio_error = abs(pred_lawson_ratio - ref_lawson_ratio) / max(ref_lawson_ratio, 1.0)
        return max(0.0, 1.0 - ratio_error)
    
    def _assess_confinement_significance(self, score: float, triple_product: float) -> str:
        """Assess physical significance of confinement prediction"""
        lawson_threshold = 3e21
        
        if score > 0.9 and triple_product > lawson_threshold:
            return "Excellent - ignition conditions predicted"
        elif score > 0.8 and triple_product > 0.1 * lawson_threshold:
            return "Good - breakeven conditions possible"
        elif score > 0.6:
            return "Moderate - research plasma regime"
        else:
            return "Poor - unphysical or sub-critical conditions"
    
    def _calculate_uncertainty_analysis(self, pred_data: Dict, ref_data: Dict) -> Dict[str, float]:
        """Calculate uncertainty metrics for physics predictions"""
        uncertainties = {}
        
        # Parameter uncertainties (simplified)
        for param in ['temperature', 'density', 'confinement_time']:
            if param in pred_data and param in ref_data:
                pred_val = pred_data[param]
                ref_val = ref_data[param]
                # Estimate uncertainty as percentage of reference value
                uncertainty = abs(pred_val - ref_val) / max(abs(ref_val), 1e-10) * 100
                uncertainties[f"{param}_uncertainty_percent"] = uncertainty
        
        return uncertainties
    
    def _evaluate_fusion_rate(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate fusion reaction rate predictions"""
        try:
            pred_rate = self._parse_fusion_rate(prediction)
            ref_rate = self._parse_fusion_rate(reference)
            
            # Calculate relative error
            rate_error = self._calculate_relative_error(pred_rate['fusion_rate'], ref_rate['fusion_rate'])
            
            # Calculate accuracy score
            accuracy = max(0.0, 1.0 - rate_error / 0.3)  # 30% tolerance
            
            # Check physics consistency
            physics_score = self._check_fusion_physics_consistency(pred_rate)
            
            overall_score = 0.7 * accuracy + 0.3 * physics_score
            
            return PhysicsMetricResult(
                metric_name="fusion_rate_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "rate_error": rate_error,
                    "accuracy_score": accuracy,
                    "physics_consistency_score": physics_score,
                    "predicted_rate": pred_rate['fusion_rate'],
                    "reference_rate": ref_rate['fusion_rate']
                },
                physical_significance=self._assess_fusion_rate_significance(overall_score, pred_rate['fusion_rate'])
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="fusion_rate_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_fusion_rate(self, data: Any) -> Dict[str, float]:
        """Parse fusion rate data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'fusion_rate': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'fusion_rate': float(data)}
        else:
            return {'fusion_rate': 0.0}
    
    def _check_fusion_physics_consistency(self, rate_data: Dict) -> float:
        """Check if fusion rate prediction is physically consistent"""
        fusion_rate = rate_data['fusion_rate']
        
        # Basic physics checks
        if fusion_rate < 0:
            return 0.0  # Negative fusion rate is unphysical
        
        # Check if rate is within reasonable bounds for given conditions
        if 'temperature' in rate_data and 'density' in rate_data:
            temp = rate_data['temperature']  # keV
            density = rate_data['density']  # m^-3
            
            # Rough estimate using simple fusion rate formula
            # For D-T fusion: <σv> ≈ 1e-22 m^3/s at 10 keV
            if temp > 0 and density > 0:
                expected_rate_order = density * density * 1e-22 * math.exp(-19.94 / temp)
                rate_ratio = fusion_rate / max(expected_rate_order, 1e-30)
                
                # Rate should be within reasonable orders of magnitude
                if 0.01 <= rate_ratio <= 100:
                    return 1.0
                elif 0.001 <= rate_ratio <= 1000:
                    return 0.5
                else:
                    return 0.1
        
        return 0.8  # Default score if we can't check consistency
    
    def _assess_fusion_rate_significance(self, score: float, rate: float) -> str:
        """Assess physical significance of fusion rate prediction"""
        if score > 0.8 and rate > 1e18:
            return "Excellent - suitable for reactor design"
        elif score > 0.6 and rate > 1e15:
            return "Good - useful for plasma physics research"
        elif score > 0.4:
            return "Moderate - basic physics captured"
        else:
            return "Poor - unphysical or highly inaccurate"
    
    def _evaluate_mhd_stability(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate MHD stability predictions"""
        try:
            pred_stability = self._parse_stability_data(prediction)
            ref_stability = self._parse_stability_data(reference)
            
            # Compare stability classifications
            stability_match = self._compare_stability_classifications(pred_stability, ref_stability)
            
            # Compare growth rates if available
            growth_rate_score = self._compare_growth_rates(pred_stability, ref_stability)
            
            # Compare mode numbers
            mode_score = self._compare_mode_numbers(pred_stability, ref_stability)
            
            overall_score = 0.5 * stability_match + 0.3 * growth_rate_score + 0.2 * mode_score
            
            return PhysicsMetricResult(
                metric_name="mhd_stability_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "stability_classification_match": stability_match,
                    "growth_rate_score": growth_rate_score,
                    "mode_number_score": mode_score,
                    "predicted_stable": pred_stability.get('stable', None),
                    "reference_stable": ref_stability.get('stable', None)
                },
                physical_significance=self._assess_mhd_significance(overall_score, pred_stability)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="mhd_stability_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_stability_data(self, data: Any) -> Dict[str, Any]:
        """Parse MHD stability data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, bool):
            return {'stable': data}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                # Parse simple format
                if data.lower() in ['stable', 'true', '1']:
                    return {'stable': True}
                elif data.lower() in ['unstable', 'false', '0']:
                    return {'stable': False}
                else:
                    return {}
        else:
            return {}
    
    def _compare_stability_classifications(self, pred: Dict, ref: Dict) -> float:
        """Compare stability classifications"""
        if 'stable' in pred and 'stable' in ref:
            return 1.0 if pred['stable'] == ref['stable'] else 0.0
        return 0.5  # Partial credit if classification is missing
    
    def _compare_growth_rates(self, pred: Dict, ref: Dict) -> float:
        """Compare instability growth rates"""
        if 'growth_rate' in pred and 'growth_rate' in ref:
            pred_rate = pred['growth_rate']
            ref_rate = ref['growth_rate']
            
            if ref_rate == 0:
                return 1.0 if pred_rate == 0 else 0.0
            
            relative_error = abs(pred_rate - ref_rate) / abs(ref_rate)
            return max(0.0, 1.0 - relative_error / 0.5)  # 50% tolerance
        
        return 0.5  # Default if growth rates not available
    
    def _compare_mode_numbers(self, pred: Dict, ref: Dict) -> float:
        """Compare mode numbers for instabilities"""
        if 'mode_n' in pred and 'mode_n' in ref:
            return 1.0 if pred['mode_n'] == ref['mode_n'] else 0.0
        return 0.5  # Default if mode numbers not available
    
    def _assess_mhd_significance(self, score: float, stability_data: Dict) -> str:
        """Assess physical significance of MHD stability prediction"""
        if score > 0.9:
            return "Excellent - reliable for disruption prediction"
        elif score > 0.7:
            return "Good - useful for stability analysis"
        elif score > 0.5:
            return "Moderate - basic stability trends captured"
        else:
            return "Poor - unreliable for plasma control"
    
    def _evaluate_disruption_prediction(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma disruption predictions"""
        try:
            pred_disruption = self._parse_disruption_data(prediction)
            ref_disruption = self._parse_disruption_data(reference)
            
            # Compare disruption occurrence
            disruption_match = pred_disruption['will_disrupt'] == ref_disruption['will_disrupt']
            
            # Compare timing if both predict disruption
            timing_score = self._compare_disruption_timing(pred_disruption, ref_disruption)
            
            # Calculate overall score
            base_score = 1.0 if disruption_match else 0.0
            overall_score = 0.7 * base_score + 0.3 * timing_score
            
            return PhysicsMetricResult(
                metric_name="disruption_prediction",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= 0.8,
                details={
                    "disruption_match": disruption_match,
                    "timing_score": timing_score,
                    "predicted_disruption": pred_disruption['will_disrupt'],
                    "reference_disruption": ref_disruption['will_disrupt'],
                    "predicted_time": pred_disruption.get('time_to_disruption'),
                    "reference_time": ref_disruption.get('time_to_disruption')
                },
                physical_significance=self._assess_disruption_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="disruption_prediction",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_disruption_data(self, data: Any) -> Dict[str, Any]:
        """Parse disruption prediction data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, bool):
            return {'will_disrupt': data}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                if data.lower() in ['disrupt', 'true', '1', 'yes']:
                    return {'will_disrupt': True}
                else:
                    return {'will_disrupt': False}
        else:
            return {'will_disrupt': False}
    
    def _compare_disruption_timing(self, pred: Dict, ref: Dict) -> float:
        """Compare disruption timing predictions"""
        if (pred['will_disrupt'] and ref['will_disrupt'] and 
            'time_to_disruption' in pred and 'time_to_disruption' in ref):
            
            pred_time = pred['time_to_disruption']
            ref_time = ref['time_to_disruption']
            
            if ref_time <= 0:
                return 1.0 if pred_time <= 0 else 0.0
            
            relative_error = abs(pred_time - ref_time) / ref_time
            return max(0.0, 1.0 - relative_error / 0.3)  # 30% tolerance
        
        return 1.0 if pred['will_disrupt'] == ref['will_disrupt'] else 0.0
    
    def _assess_disruption_significance(self, score: float) -> str:
        """Assess significance of disruption prediction"""
        if score > 0.9:
            return "Excellent - suitable for real-time disruption mitigation"
        elif score > 0.8:
            return "Good - useful for plasma protection systems"
        elif score > 0.6:
            return "Moderate - research-level prediction capability"
        else:
            return "Poor - unreliable for operational use"
    
    def _evaluate_transport_modeling(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma transport modeling predictions"""
        try:
            pred_transport = self._parse_transport_data(prediction)
            ref_transport = self._parse_transport_data(reference)
            
            # Evaluate different transport coefficients
            transport_scores = {}
            
            for coefficient in ['thermal_diffusivity', 'particle_diffusivity', 'momentum_diffusivity']:
                if coefficient in pred_transport and coefficient in ref_transport:
                    error = self._calculate_relative_error(
                        pred_transport[coefficient], 
                        ref_transport[coefficient]
                    )
                    transport_scores[coefficient] = max(0.0, 1.0 - error / 0.5)  # 50% tolerance
            
            # Overall transport score
            if transport_scores:
                overall_score = np.mean(list(transport_scores.values()))
            else:
                overall_score = 0.0
            
            return PhysicsMetricResult(
                metric_name="transport_modeling",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "transport_coefficient_scores": transport_scores,
                    "predicted_coefficients": {k: v for k, v in pred_transport.items()},
                    "reference_coefficients": {k: v for k, v in ref_transport.items()}
                },
                physical_significance=self._assess_transport_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="transport_modeling",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_transport_data(self, data: Any) -> Dict[str, float]:
        """Parse transport coefficient data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {}
        else:
            return {}
    
    def _assess_transport_significance(self, score: float) -> str:
        """Assess significance of transport modeling"""
        if score > 0.8:
            return "Excellent - suitable for predictive modeling"
        elif score > 0.6:
            return "Good - useful for transport analysis"
        elif score > 0.4:
            return "Moderate - basic transport physics captured"
        else:
            return "Poor - transport prediction unreliable"

class MaterialsScienceEvaluator(PhysicsEvaluatorBase):
    """
    Evaluator for materials science in extreme environments
    
    Supports:
    - Radiation damage assessment
    - Thermal property prediction
    - Mechanical property evaluation
    - Material degradation modeling
    - Neutron irradiation effects
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.material_type = config.get('material_type', 'steel')
        self.irradiation_dose = config.get('irradiation_dose', 0.0)  # DPA
        self.temperature_range = config.get('temperature_range', [300, 1000])  # K
    
    def evaluate(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate materials science prediction"""
        task_type = self.config.get('task_type', 'radiation_damage')
        
        if task_type == 'radiation_damage':
            return self._evaluate_radiation_damage(prediction, reference)
        elif task_type == 'thermal_properties':
            return self._evaluate_thermal_properties(prediction, reference)
        elif task_type == 'mechanical_properties':
            return self._evaluate_mechanical_properties(prediction, reference)
        elif task_type == 'material_degradation':
            return self._evaluate_material_degradation(prediction, reference)
        else:
            raise ValueError(f"Unknown materials science task type: {task_type}")
    
    def _evaluate_radiation_damage(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate radiation damage predictions"""
        try:
            pred_damage = self._parse_damage_data(prediction)
            ref_damage = self._parse_damage_data(reference)
            
            # Compare displacement damage
            dpa_error = self._calculate_relative_error(pred_damage['dpa'], ref_damage['dpa'])
            dpa_score = max(0.0, 1.0 - dpa_error / 0.3)  # 30% tolerance
            
            # Compare defect concentrations if available
            defect_score = self._compare_defect_concentrations(pred_damage, ref_damage)
            
            # Compare material property changes
            property_score = self._compare_property_changes(pred_damage, ref_damage)
            
            overall_score = 0.4 * dpa_score + 0.3 * defect_score + 0.3 * property_score
            
            return PhysicsMetricResult(
                metric_name="radiation_damage_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "dpa_error": dpa_error,
                    "dpa_score": dpa_score,
                    "defect_concentration_score": defect_score,
                    "property_change_score": property_score,
                    "predicted_dpa": pred_damage['dpa'],
                    "reference_dpa": ref_damage['dpa']
                },
                physical_significance=self._assess_damage_significance(overall_score, pred_damage['dpa'])
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="radiation_damage_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_damage_data(self, data: Any) -> Dict[str, float]:
        """Parse radiation damage data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'dpa': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                try:
                    return {'dpa': float(data)}
                except:
                    return {'dpa': 0.0}
        else:
            return {'dpa': 0.0}
    
    def _compare_defect_concentrations(self, pred: Dict, ref: Dict) -> float:
        """Compare defect concentration predictions"""
        defect_types = ['vacancies', 'interstitials', 'clusters']
        scores = []
        
        for defect in defect_types:
            if defect in pred and defect in ref:
                error = self._calculate_relative_error(pred[defect], ref[defect])
                score = max(0.0, 1.0 - error / 0.5)  # 50% tolerance
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _compare_property_changes(self, pred: Dict, ref: Dict) -> float:
        """Compare material property changes due to radiation"""
        properties = ['yield_strength', 'ultimate_strength', 'ductility', 'thermal_conductivity']
        scores = []
        
        for prop in properties:
            if prop in pred and prop in ref:
                error = self._calculate_relative_error(pred[prop], ref[prop])
                score = max(0.0, 1.0 - error / 0.4)  # 40% tolerance
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_damage_significance(self, score: float, dpa: float) -> str:
        """Assess significance of radiation damage prediction"""
        if score > 0.8 and dpa < 100:
            return "Excellent - suitable for reactor component design"
        elif score > 0.6 and dpa < 200:
            return "Good - useful for material selection"
        elif score > 0.4:
            return "Moderate - basic damage trends captured"
        else:
            return "Poor - unreliable for engineering applications"
    
    def _evaluate_thermal_properties(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate thermal property predictions"""
        try:
            pred_thermal = self._parse_thermal_data(prediction)
            ref_thermal = self._parse_thermal_data(reference)
            
            # Compare thermal conductivity
            k_error = self._calculate_relative_error(
                pred_thermal['thermal_conductivity'], 
                ref_thermal['thermal_conductivity']
            )
            k_score = max(0.0, 1.0 - k_error / 0.2)  # 20% tolerance
            
            # Compare specific heat if available
            cp_score = 1.0
            if 'specific_heat' in pred_thermal and 'specific_heat' in ref_thermal:
                cp_error = self._calculate_relative_error(
                    pred_thermal['specific_heat'], 
                    ref_thermal['specific_heat']
                )
                cp_score = max(0.0, 1.0 - cp_error / 0.15)  # 15% tolerance
            
            # Compare thermal expansion if available
            expansion_score = 1.0
            if 'thermal_expansion' in pred_thermal and 'thermal_expansion' in ref_thermal:
                expansion_error = self._calculate_relative_error(
                    pred_thermal['thermal_expansion'], 
                    ref_thermal['thermal_expansion']
                )
                expansion_score = max(0.0, 1.0 - expansion_error / 0.25)  # 25% tolerance
            
            overall_score = 0.5 * k_score + 0.3 * cp_score + 0.2 * expansion_score
            
            return PhysicsMetricResult(
                metric_name="thermal_properties_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "thermal_conductivity_score": k_score,
                    "specific_heat_score": cp_score,
                    "thermal_expansion_score": expansion_score,
                    "thermal_conductivity_error": k_error
                },
                physical_significance=self._assess_thermal_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="thermal_properties_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_thermal_data(self, data: Any) -> Dict[str, float]:
        """Parse thermal property data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'thermal_conductivity': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'thermal_conductivity': float(data)}
        else:
            return {'thermal_conductivity': 0.0}
    
    def _assess_thermal_significance(self, score: float) -> str:
        """Assess significance of thermal property prediction"""
        if score > 0.8:
            return "Excellent - suitable for thermal design calculations"
        elif score > 0.6:
            return "Good - useful for material characterization"
        elif score > 0.4:
            return "Moderate - basic thermal behavior captured"
        else:
            return "Poor - unreliable for thermal analysis"
    
    def _evaluate_mechanical_properties(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate mechanical property predictions"""
        try:
            pred_mech = self._parse_mechanical_data(prediction)
            ref_mech = self._parse_mechanical_data(reference)
            
            # Compare yield strength
            yield_error = self._calculate_relative_error(
                pred_mech['yield_strength'], 
                ref_mech['yield_strength']
            )
            yield_score = max(0.0, 1.0 - yield_error / 0.15)  # 15% tolerance
            
            # Compare ultimate tensile strength
            uts_score = 1.0
            if 'ultimate_strength' in pred_mech and 'ultimate_strength' in ref_mech:
                uts_error = self._calculate_relative_error(
                    pred_mech['ultimate_strength'], 
                    ref_mech['ultimate_strength']
                )
                uts_score = max(0.0, 1.0 - uts_error / 0.15)
            
            # Compare elastic modulus
            modulus_score = 1.0
            if 'elastic_modulus' in pred_mech and 'elastic_modulus' in ref_mech:
                modulus_error = self._calculate_relative_error(
                    pred_mech['elastic_modulus'], 
                    ref_mech['elastic_modulus']
                )
                modulus_score = max(0.0, 1.0 - modulus_error / 0.1)  # 10% tolerance
            
            overall_score = 0.4 * yield_score + 0.3 * uts_score + 0.3 * modulus_score
            
            return PhysicsMetricResult(
                metric_name="mechanical_properties_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "yield_strength_score": yield_score,
                    "ultimate_strength_score": uts_score,
                    "elastic_modulus_score": modulus_score,
                    "yield_strength_error": yield_error
                },
                physical_significance=self._assess_mechanical_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="mechanical_properties_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_mechanical_data(self, data: Any) -> Dict[str, float]:
        """Parse mechanical property data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'yield_strength': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'yield_strength': float(data)}
        else:
            return {'yield_strength': 0.0}
    
    def _assess_mechanical_significance(self, score: float) -> str:
        """Assess significance of mechanical property prediction"""
        if score > 0.8:
            return "Excellent - suitable for structural design"
        elif score > 0.6:
            return "Good - useful for material qualification"
        elif score > 0.4:
            return "Moderate - basic mechanical behavior captured"
        else:
            return "Poor - unreliable for engineering design"
    
    def _evaluate_material_degradation(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate material degradation modeling"""
        try:
            pred_degrad = self._parse_degradation_data(prediction)
            ref_degrad = self._parse_degradation_data(reference)
            
            # Compare degradation rate
            rate_error = self._calculate_relative_error(
                pred_degrad['degradation_rate'], 
                ref_degrad['degradation_rate']
            )
            rate_score = max(0.0, 1.0 - rate_error / 0.4)  # 40% tolerance
            
            # Compare lifetime prediction
            lifetime_score = 1.0
            if 'predicted_lifetime' in pred_degrad and 'predicted_lifetime' in ref_degrad:
                lifetime_error = self._calculate_relative_error(
                    pred_degrad['predicted_lifetime'], 
                    ref_degrad['predicted_lifetime']
                )
                lifetime_score = max(0.0, 1.0 - lifetime_error / 0.5)  # 50% tolerance
            
            overall_score = 0.6 * rate_score + 0.4 * lifetime_score
            
            return PhysicsMetricResult(
                metric_name="material_degradation_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "degradation_rate_score": rate_score,
                    "lifetime_prediction_score": lifetime_score,
                    "degradation_rate_error": rate_error
                },
                physical_significance=self._assess_degradation_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="material_degradation_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_degradation_data(self, data: Any) -> Dict[str, float]:
        """Parse material degradation data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'degradation_rate': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'degradation_rate': float(data)}
        else:
            return {'degradation_rate': 0.0}
    
    def _assess_degradation_significance(self, score: float) -> str:
        """Assess significance of degradation prediction"""
        if score > 0.8:
            return "Excellent - suitable for lifetime assessment"
        elif score > 0.6:
            return "Good - useful for maintenance planning"
        elif score > 0.4:
            return "Moderate - basic degradation trends captured"
        else:
            return "Poor - unreliable for lifecycle prediction"

class PlasmaPhysicsEvaluator(PhysicsEvaluatorBase):
    """
    Evaluator for general plasma physics simulations
    
    Supports:
    - Particle-in-cell (PIC) simulation analysis
    - Fluid plasma modeling
    - Kinetic theory validation
    - Wave propagation in plasmas
    - Electric and magnetic field predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plasma_type = config.get('plasma_type', 'magnetized')
        self.simulation_type = config.get('simulation_type', 'fluid')
        self.field_tolerance = config.get('field_tolerance', 0.1)  # 10%
    
    def evaluate(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma physics prediction"""
        task_type = self.config.get('task_type', 'field_evolution')
        
        if task_type == 'field_evolution':
            return self._evaluate_field_evolution(prediction, reference)
        elif task_type == 'particle_distribution':
            return self._evaluate_particle_distribution(prediction, reference)
        elif task_type == 'wave_propagation':
            return self._evaluate_wave_propagation(prediction, reference)
        elif task_type == 'instability_growth':
            return self._evaluate_instability_growth(prediction, reference)
        else:
            raise ValueError(f"Unknown plasma physics task type: {task_type}")
    
    def _evaluate_field_evolution(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate electromagnetic field evolution predictions"""
        try:
            pred_fields = self._parse_field_data(prediction)
            ref_fields = self._parse_field_data(reference)
            
            # Compare electric field components
            e_field_score = self._compare_vector_fields(
                pred_fields.get('electric_field', [0, 0, 0]),
                ref_fields.get('electric_field', [0, 0, 0])
            )
            
            # Compare magnetic field components
            b_field_score = self._compare_vector_fields(
                pred_fields.get('magnetic_field', [0, 0, 0]),
                ref_fields.get('magnetic_field', [0, 0, 0])
            )
            
            # Check field conservation laws
            conservation_score = self._check_field_conservation(pred_fields, ref_fields)
            
            overall_score = 0.4 * e_field_score + 0.4 * b_field_score + 0.2 * conservation_score
            
            return PhysicsMetricResult(
                metric_name="field_evolution_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "electric_field_score": e_field_score,
                    "magnetic_field_score": b_field_score,
                    "conservation_score": conservation_score
                },
                physical_significance=self._assess_field_significance(overall_score),
                dimensional_analysis=self._check_field_dimensions(pred_fields)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="field_evolution_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_field_data(self, data: Any) -> Dict[str, Any]:
        """Parse electromagnetic field data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {}
        else:
            return {}
    
    def _compare_vector_fields(self, pred_field: List[float], ref_field: List[float]) -> float:
        """Compare vector field components"""
        if len(pred_field) != len(ref_field):
            return 0.0
        
        scores = []
        for i in range(len(pred_field)):
            if abs(ref_field[i]) > 1e-10:
                error = abs(pred_field[i] - ref_field[i]) / abs(ref_field[i])
            else:
                error = abs(pred_field[i] - ref_field[i])
            
            score = max(0.0, 1.0 - error / self.field_tolerance)
            scores.append(score)
        
        return np.mean(scores)
    
    def _check_field_conservation(self, pred_fields: Dict, ref_fields: Dict) -> float:
        """Check conservation laws for electromagnetic fields"""
        # This is a simplified check - in practice would involve more complex calculations
        
        # Check if field magnitudes are reasonable
        conservation_score = 1.0
        
        # Check energy conservation (simplified)
        if 'energy_density' in pred_fields and 'energy_density' in ref_fields:
            energy_error = self._calculate_relative_error(
                pred_fields['energy_density'], 
                ref_fields['energy_density']
            )
            conservation_score *= max(0.0, 1.0 - energy_error / 0.2)
        
        return conservation_score
    
    def _assess_field_significance(self, score: float) -> str:
        """Assess significance of field evolution prediction"""
        if score > 0.8:
            return "Excellent - suitable for detailed plasma modeling"
        elif score > 0.6:
            return "Good - useful for plasma simulation validation"
        elif score > 0.4:
            return "Moderate - basic field evolution captured"
        else:
            return "Poor - field predictions unreliable"
    
    def _check_field_dimensions(self, fields: Dict) -> Dict[str, str]:
        """Check dimensional analysis of field quantities"""
        dimensions = {}
        
        if 'electric_field' in fields:
            dimensions['electric_field'] = "V/m"
        if 'magnetic_field' in fields:
            dimensions['magnetic_field'] = "T"
        if 'energy_density' in fields:
            dimensions['energy_density'] = "J/m³"
        
        return dimensions
    
    def _evaluate_particle_distribution(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate particle distribution function predictions"""
        try:
            pred_dist = self._parse_distribution_data(prediction)
            ref_dist = self._parse_distribution_data(reference)
            
            # Compare distribution moments
            moment_score = self._compare_distribution_moments(pred_dist, ref_dist)
            
            # Compare distribution shape if full distribution available
            shape_score = self._compare_distribution_shape(pred_dist, ref_dist)
            
            overall_score = 0.6 * moment_score + 0.4 * shape_score
            
            return PhysicsMetricResult(
                metric_name="particle_distribution_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "moment_score": moment_score,
                    "shape_score": shape_score
                },
                physical_significance=self._assess_distribution_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="particle_distribution_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_distribution_data(self, data: Any) -> Dict[str, Any]:
        """Parse particle distribution data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {}
        else:
            return {}
    
    def _compare_distribution_moments(self, pred_dist: Dict, ref_dist: Dict) -> float:
        """Compare distribution moments (density, temperature, etc.)"""
        moments = ['density', 'temperature', 'drift_velocity']
        scores = []
        
        for moment in moments:
            if moment in pred_dist and moment in ref_dist:
                error = self._calculate_relative_error(pred_dist[moment], ref_dist[moment])
                score = max(0.0, 1.0 - error / 0.3)  # 30% tolerance
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _compare_distribution_shape(self, pred_dist: Dict, ref_dist: Dict) -> float:
        """Compare shape of distribution functions"""
        # Simplified comparison - would involve more sophisticated analysis in practice
        if 'distribution_function' in pred_dist and 'distribution_function' in ref_dist:
            pred_func = np.array(pred_dist['distribution_function'])
            ref_func = np.array(ref_dist['distribution_function'])
            
            if pred_func.shape == ref_func.shape:
                # Use normalized correlation coefficient
                correlation = np.corrcoef(pred_func.flatten(), ref_func.flatten())[0, 1]
                return max(0.0, correlation)
        
        return 0.5  # Default if full distribution not available
    
    def _assess_distribution_significance(self, score: float) -> str:
        """Assess significance of distribution prediction"""
        if score > 0.8:
            return "Excellent - accurate kinetic modeling"
        elif score > 0.6:
            return "Good - useful for transport analysis"
        elif score > 0.4:
            return "Moderate - basic distribution features captured"
        else:
            return "Poor - kinetic description unreliable"
    
    def _evaluate_wave_propagation(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma wave propagation predictions"""
        try:
            pred_wave = self._parse_wave_data(prediction)
            ref_wave = self._parse_wave_data(reference)
            
            # Compare wave frequency
            freq_error = self._calculate_relative_error(pred_wave['frequency'], ref_wave['frequency'])
            freq_score = max(0.0, 1.0 - freq_error / 0.1)  # 10% tolerance
            
            # Compare wave vector
            k_score = self._compare_wave_vector(pred_wave, ref_wave)
            
            # Compare dispersion relation
            dispersion_score = self._check_dispersion_relation(pred_wave, ref_wave)
            
            overall_score = 0.4 * freq_score + 0.3 * k_score + 0.3 * dispersion_score
            
            return PhysicsMetricResult(
                metric_name="wave_propagation_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "frequency_score": freq_score,
                    "wave_vector_score": k_score,
                    "dispersion_score": dispersion_score,
                    "frequency_error": freq_error
                },
                physical_significance=self._assess_wave_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="wave_propagation_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_wave_data(self, data: Any) -> Dict[str, Any]:
        """Parse wave propagation data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {}
        else:
            return {}
    
    def _compare_wave_vector(self, pred_wave: Dict, ref_wave: Dict) -> float:
        """Compare wave vector components"""
        if 'wave_vector' in pred_wave and 'wave_vector' in ref_wave:
            return self._compare_vector_fields(pred_wave['wave_vector'], ref_wave['wave_vector'])
        return 0.5
    
    def _check_dispersion_relation(self, pred_wave: Dict, ref_wave: Dict) -> float:
        """Check if wave satisfies correct dispersion relation"""
        # Simplified check - would involve more complex plasma physics in practice
        if all(key in pred_wave for key in ['frequency', 'wave_vector']):
            # Basic consistency check
            return 1.0  # Placeholder for dispersion relation check
        return 0.5
    
    def _assess_wave_significance(self, score: float) -> str:
        """Assess significance of wave propagation prediction"""
        if score > 0.8:
            return "Excellent - accurate wave physics modeling"
        elif score > 0.6:
            return "Good - useful for wave heating analysis"
        elif score > 0.4:
            return "Moderate - basic wave properties captured"
        else:
            return "Poor - wave predictions unreliable"
    
    def _evaluate_instability_growth(self, prediction: Any, reference: Any) -> PhysicsMetricResult:
        """Evaluate plasma instability growth predictions"""
        try:
            pred_instability = self._parse_instability_data(prediction)
            ref_instability = self._parse_instability_data(reference)
            
            # Compare growth rate
            growth_rate_error = self._calculate_relative_error(
                pred_instability['growth_rate'], 
                ref_instability['growth_rate']
            )
            growth_score = max(0.0, 1.0 - growth_rate_error / 0.3)  # 30% tolerance
            
            # Compare mode structure
            mode_score = self._compare_mode_structure(pred_instability, ref_instability)
            
            overall_score = 0.7 * growth_score + 0.3 * mode_score
            
            return PhysicsMetricResult(
                metric_name="instability_growth_prediction",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "growth_rate_score": growth_score,
                    "mode_structure_score": mode_score,
                    "growth_rate_error": growth_rate_error
                },
                physical_significance=self._assess_instability_significance(overall_score)
            )
            
        except Exception as e:
            return PhysicsMetricResult(
                metric_name="instability_growth_prediction",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_instability_data(self, data: Any) -> Dict[str, Any]:
        """Parse instability data"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'growth_rate': float(data)}
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'growth_rate': float(data)}
        else:
            return {'growth_rate': 0.0}
    
    def _compare_mode_structure(self, pred_inst: Dict, ref_inst: Dict) -> float:
        """Compare instability mode structure"""
        if 'mode_structure' in pred_inst and 'mode_structure' in ref_inst:
            pred_structure = np.array(pred_inst['mode_structure'])
            ref_structure = np.array(ref_inst['mode_structure'])
            
            if pred_structure.shape == ref_structure.shape:
                correlation = np.corrcoef(pred_structure.flatten(), ref_structure.flatten())[0, 1]
                return max(0.0, correlation)
        
        return 0.5
    
    def _assess_instability_significance(self, score: float) -> str:
        """Assess significance of instability prediction"""
        if score > 0.8:
            return "Excellent - reliable for stability analysis"
        elif score > 0.6:
            return "Good - useful for plasma control"
        elif score > 0.4:
            return "Moderate - basic instability trends captured"
        else:
            return "Poor - instability prediction unreliable"

# Register physics evaluators with the grading system
def register_physics_evaluators():
    """Register physics evaluators with the OpenEvals grading system"""
    from openevals.core.graders import register_grader
    from openevals.config.data_structures import MetricType
    
    def fusion_physics_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = FusionPhysicsEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def materials_science_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = MaterialsScienceEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def plasma_physics_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = PlasmaPhysicsEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    # Register the graders
    register_grader("fusion_physics", fusion_physics_grader, MetricType.CUSTOM)
    register_grader("materials_science", materials_science_grader, MetricType.CUSTOM)
    register_grader("plasma_physics", plasma_physics_grader, MetricType.CUSTOM)

# Auto-register on import
if PHYSICS_DEPS_AVAILABLE:
    try:
        register_physics_evaluators()
        logger.info("Physics evaluators registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register physics evaluators: {e}") 