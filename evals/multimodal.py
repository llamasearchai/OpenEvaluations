"""
Multimodal AI Evaluation Module
=============================

Specialized evaluations for multimodal AI systems including:
- Image understanding and generation
- Audio processing and synthesis
- Video analysis and generation
- Cross-modal alignment and retrieval
- Vision-language models
- Audio-visual synchronization

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import base64
import io

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import cv2
    import librosa
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    MULTIMODAL_DEPS_AVAILABLE = True
except ImportError:
    MULTIMODAL_DEPS_AVAILABLE = False
    logging.warning("Multimodal dependencies not available. Install with: pip install torch torchvision pillow opencv-python librosa matplotlib scikit-learn pandas")

from openevals.config.data_structures import MetricResult, MetricType
from openevals.core.definitions import EvalTask, EvalCase

logger = logging.getLogger(__name__)

@dataclass
class MultimodalMetricResult(MetricResult):
    """Extended metric result for multimodal evaluations"""
    modality_scores: Optional[Dict[str, float]] = None
    cross_modal_alignment: Optional[float] = None
    visual_quality: Optional[float] = None
    audio_quality: Optional[float] = None
    semantic_consistency: Optional[float] = None

class MultimodalEvaluatorBase(ABC):
    """Base class for multimodal evaluators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not MULTIMODAL_DEPS_AVAILABLE:
            raise ImportError("Multimodal dependencies required")
        
        self.image_size = config.get('image_size', (224, 224))
        self.audio_sr = config.get('audio_sample_rate', 22050)
        self.video_fps = config.get('video_fps', 30)
    
    @abstractmethod
    def evaluate(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate multimodal prediction against reference"""
        pass

class ImageEvaluator(MultimodalEvaluatorBase):
    """
    Evaluator for image understanding and generation tasks
    
    Supports:
    - Image classification accuracy
    - Object detection metrics (mAP, IoU)
    - Image quality assessment (PSNR, SSIM, LPIPS)
    - Semantic segmentation evaluation
    - Image captioning metrics
    - Visual question answering
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get('task_type', 'classification')
        self.quality_metrics = config.get('quality_metrics', ['psnr', 'ssim'])
        self.iou_threshold = config.get('iou_threshold', 0.5)
    
    def evaluate(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate image prediction"""
        if self.task_type == 'classification':
            return self._evaluate_classification(prediction, reference)
        elif self.task_type == 'detection':
            return self._evaluate_detection(prediction, reference)
        elif self.task_type == 'generation':
            return self._evaluate_generation(prediction, reference)
        elif self.task_type == 'captioning':
            return self._evaluate_captioning(prediction, reference)
        elif self.task_type == 'vqa':
            return self._evaluate_vqa(prediction, reference)
        else:
            raise ValueError(f"Unknown image task type: {self.task_type}")
    
    def _evaluate_classification(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate image classification predictions"""
        try:
            pred_data = self._parse_classification_prediction(prediction)
            ref_data = self._parse_classification_reference(reference)
            
            # Calculate accuracy
            accuracy = self._calculate_classification_accuracy(pred_data, ref_data)
            
            # Calculate top-k accuracy if available
            top5_accuracy = self._calculate_topk_accuracy(pred_data, ref_data, k=5)
            
            # Calculate confidence metrics
            confidence_score = self._evaluate_confidence_calibration(pred_data, ref_data)
            
            overall_score = 0.6 * accuracy + 0.2 * top5_accuracy + 0.2 * confidence_score
            
            return MultimodalMetricResult(
                metric_name="image_classification",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "accuracy": accuracy,
                    "top5_accuracy": top5_accuracy,
                    "confidence_score": confidence_score,
                    "predicted_class": pred_data.get('class', None),
                    "true_class": ref_data.get('class', None)
                },
                modality_scores={"vision": overall_score}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="image_classification",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_classification_prediction(self, data: Any) -> Dict[str, Any]:
        """Parse image classification prediction"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'class': data}
        elif isinstance(data, (int, float)):
            return {'class': int(data)}
        else:
            return {}
    
    def _parse_classification_reference(self, data: Any) -> Dict[str, Any]:
        """Parse image classification reference"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'class': data}
        elif isinstance(data, (int, float)):
            return {'class': int(data)}
        else:
            return {}
    
    def _calculate_classification_accuracy(self, pred_data: Dict, ref_data: Dict) -> float:
        """Calculate classification accuracy"""
        pred_class = pred_data.get('class', None)
        true_class = ref_data.get('class', None)
        
        if pred_class is None or true_class is None:
            return 0.0
        
        return 1.0 if pred_class == true_class else 0.0
    
    def _calculate_topk_accuracy(self, pred_data: Dict, ref_data: Dict, k: int = 5) -> float:
        """Calculate top-k accuracy"""
        if 'top_predictions' in pred_data:
            top_preds = pred_data['top_predictions'][:k]
            true_class = ref_data.get('class', None)
            
            if true_class is not None:
                return 1.0 if true_class in top_preds else 0.0
        
        # Fallback to regular accuracy
        return self._calculate_classification_accuracy(pred_data, ref_data)
    
    def _evaluate_confidence_calibration(self, pred_data: Dict, ref_data: Dict) -> float:
        """Evaluate confidence calibration"""
        if 'confidence' in pred_data:
            confidence = pred_data['confidence']
            accuracy = self._calculate_classification_accuracy(pred_data, ref_data)
            
            # Simple calibration check - confidence should match accuracy
            calibration_error = abs(confidence - accuracy)
            return max(0.0, 1.0 - calibration_error)
        
        return 0.5  # Default neutral score
    
    def _evaluate_detection(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate object detection predictions"""
        try:
            pred_boxes = self._parse_detection_prediction(prediction)
            ref_boxes = self._parse_detection_reference(reference)
            
            # Calculate mAP (mean Average Precision)
            map_score = self._calculate_map(pred_boxes, ref_boxes)
            
            # Calculate average IoU
            avg_iou = self._calculate_average_iou(pred_boxes, ref_boxes)
            
            # Calculate precision and recall
            precision, recall = self._calculate_precision_recall(pred_boxes, ref_boxes)
            
            overall_score = 0.5 * map_score + 0.3 * avg_iou + 0.1 * precision + 0.1 * recall
            
            return MultimodalMetricResult(
                metric_name="object_detection",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "map_score": map_score,
                    "average_iou": avg_iou,
                    "precision": precision,
                    "recall": recall,
                    "num_predictions": len(pred_boxes),
                    "num_ground_truth": len(ref_boxes)
                },
                modality_scores={"vision": overall_score}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="object_detection",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_detection_prediction(self, data: Any) -> List[Dict]:
        """Parse object detection predictions"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'boxes' in data:
            return data['boxes']
        elif isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and 'boxes' in parsed:
                    return parsed['boxes']
            except:
                pass
        
        return []
    
    def _parse_detection_reference(self, data: Any) -> List[Dict]:
        """Parse object detection ground truth"""
        return self._parse_detection_prediction(data)
    
    def _calculate_map(self, pred_boxes: List[Dict], ref_boxes: List[Dict]) -> float:
        """Calculate mean Average Precision (simplified)"""
        if not pred_boxes or not ref_boxes:
            return 0.0
        
        # Simplified mAP calculation
        # In practice, this would involve proper AP calculation per class
        total_iou = 0.0
        matched_boxes = 0
        
        for pred_box in pred_boxes:
            best_iou = 0.0
            for ref_box in ref_boxes:
                iou = self._calculate_iou(pred_box, ref_box)
                if iou > best_iou:
                    best_iou = iou
            
            if best_iou >= self.iou_threshold:
                matched_boxes += 1
                total_iou += best_iou
        
        return total_iou / len(pred_boxes) if pred_boxes else 0.0
    
    def _calculate_average_iou(self, pred_boxes: List[Dict], ref_boxes: List[Dict]) -> float:
        """Calculate average IoU between predictions and ground truth"""
        if not pred_boxes or not ref_boxes:
            return 0.0
        
        total_iou = 0.0
        count = 0
        
        for pred_box in pred_boxes:
            for ref_box in ref_boxes:
                iou = self._calculate_iou(pred_box, ref_box)
                total_iou += iou
                count += 1
        
        return total_iou / count if count > 0 else 0.0
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union between two boxes"""
        # Extract coordinates (assuming format: x1, y1, x2, y2)
        x1_1, y1_1, x2_1, y2_1 = box1.get('bbox', [0, 0, 0, 0])
        x1_2, y1_2, x2_2, y2_2 = box2.get('bbox', [0, 0, 0, 0])
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_precision_recall(self, pred_boxes: List[Dict], ref_boxes: List[Dict]) -> Tuple[float, float]:
        """Calculate precision and recall"""
        if not pred_boxes:
            return 0.0, 0.0
        if not ref_boxes:
            return 0.0, 0.0
        
        true_positives = 0
        
        for pred_box in pred_boxes:
            for ref_box in ref_boxes:
                iou = self._calculate_iou(pred_box, ref_box)
                if iou >= self.iou_threshold:
                    true_positives += 1
                    break
        
        precision = true_positives / len(pred_boxes)
        recall = true_positives / len(ref_boxes)
        
        return precision, recall
    
    def _evaluate_generation(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate image generation quality"""
        try:
            pred_image = self._parse_image(prediction)
            ref_image = self._parse_image(reference)
            
            if pred_image is None or ref_image is None:
                return MultimodalMetricResult(
                    metric_name="image_generation",
                    metric_type=MetricType.CUSTOM,
                    value=0.0,
                    passed=False,
                    details={"error": "Could not parse images"}
                )
            
            # Calculate image quality metrics
            quality_scores = {}
            
            if 'psnr' in self.quality_metrics:
                psnr = self._calculate_psnr(pred_image, ref_image)
                quality_scores['psnr'] = psnr
            
            if 'ssim' in self.quality_metrics:
                ssim = self._calculate_ssim(pred_image, ref_image)
                quality_scores['ssim'] = ssim
            
            if 'mse' in self.quality_metrics:
                mse = self._calculate_mse(pred_image, ref_image)
                quality_scores['mse'] = 1.0 / (1.0 + mse)  # Convert to similarity score
            
            # Overall quality score
            overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
            
            return MultimodalMetricResult(
                metric_name="image_generation",
                metric_type=MetricType.CUSTOM,
                value=overall_quality,
                passed=overall_quality >= 0.6,
                details=quality_scores,
                visual_quality=overall_quality,
                modality_scores={"vision": overall_quality}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="image_generation",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_image(self, data: Any) -> Optional[np.ndarray]:
        """Parse image from various formats"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            try:
                # Try to decode base64
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
            except:
                try:
                    # Try to load from file path
                    image = Image.open(data)
                    return np.array(image)
                except:
                    return None
        elif hasattr(data, 'numpy'):
            # PyTorch tensor
            return data.numpy()
        else:
            return None
    
    def _calculate_psnr(self, pred_image: np.ndarray, ref_image: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((pred_image - ref_image) ** 2)
        if mse == 0:
            return 1.0
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Normalize to 0-1 range (typical PSNR range is 20-40 dB)
        return min(1.0, max(0.0, (psnr - 20) / 20))
    
    def _calculate_ssim(self, pred_image: np.ndarray, ref_image: np.ndarray) -> float:
        """Calculate Structural Similarity Index (simplified)"""
        # Simplified SSIM calculation
        mu1 = np.mean(pred_image)
        mu2 = np.mean(ref_image)
        
        sigma1_sq = np.var(pred_image)
        sigma2_sq = np.var(ref_image)
        sigma12 = np.mean((pred_image - mu1) * (ref_image - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim = numerator / denominator
        return max(0.0, min(1.0, ssim))
    
    def _calculate_mse(self, pred_image: np.ndarray, ref_image: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((pred_image - ref_image) ** 2)
    
    def _evaluate_captioning(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate image captioning"""
        try:
            pred_caption = self._parse_caption(prediction)
            ref_captions = self._parse_reference_captions(reference)
            
            # Calculate text similarity metrics
            bleu_score = self._calculate_bleu(pred_caption, ref_captions)
            meteor_score = self._calculate_meteor(pred_caption, ref_captions)
            cider_score = self._calculate_cider(pred_caption, ref_captions)
            
            overall_score = 0.4 * bleu_score + 0.3 * meteor_score + 0.3 * cider_score
            
            return MultimodalMetricResult(
                metric_name="image_captioning",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "bleu_score": bleu_score,
                    "meteor_score": meteor_score,
                    "cider_score": cider_score,
                    "predicted_caption": pred_caption,
                    "reference_captions": ref_captions
                },
                modality_scores={"vision": 0.5, "language": 0.5},
                cross_modal_alignment=overall_score
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="image_captioning",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_caption(self, data: Any) -> str:
        """Parse caption from prediction"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return data.get('caption', data.get('text', ''))
        else:
            return str(data)
    
    def _parse_reference_captions(self, data: Any) -> List[str]:
        """Parse reference captions"""
        if isinstance(data, list):
            return [str(cap) for cap in data]
        elif isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return [str(cap) for cap in parsed]
                else:
                    return [data]
            except:
                return [data]
        elif isinstance(data, dict):
            if 'captions' in data:
                return [str(cap) for cap in data['captions']]
            else:
                return [str(data.get('caption', data.get('text', '')))]
        else:
            return [str(data)]
    
    def _calculate_bleu(self, pred_caption: str, ref_captions: List[str]) -> float:
        """Calculate BLEU score (simplified)"""
        pred_words = pred_caption.lower().split()
        
        if not pred_words:
            return 0.0
        
        best_score = 0.0
        for ref_caption in ref_captions:
            ref_words = ref_caption.lower().split()
            
            # Simple unigram BLEU
            common_words = set(pred_words) & set(ref_words)
            score = len(common_words) / len(pred_words) if pred_words else 0.0
            best_score = max(best_score, score)
        
        return best_score
    
    def _calculate_meteor(self, pred_caption: str, ref_captions: List[str]) -> float:
        """Calculate METEOR score (simplified)"""
        # Simplified METEOR calculation
        return self._calculate_bleu(pred_caption, ref_captions)  # Placeholder
    
    def _calculate_cider(self, pred_caption: str, ref_captions: List[str]) -> float:
        """Calculate CIDEr score (simplified)"""
        # Simplified CIDEr calculation
        return self._calculate_bleu(pred_caption, ref_captions)  # Placeholder
    
    def _evaluate_vqa(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate visual question answering"""
        try:
            pred_answer = self._parse_vqa_answer(prediction)
            ref_answer = self._parse_vqa_answer(reference)
            
            # Calculate answer accuracy
            exact_match = self._calculate_exact_match(pred_answer, ref_answer)
            
            # Calculate semantic similarity
            semantic_sim = self._calculate_semantic_similarity(pred_answer, ref_answer)
            
            overall_score = 0.7 * exact_match + 0.3 * semantic_sim
            
            return MultimodalMetricResult(
                metric_name="visual_question_answering",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "exact_match": exact_match,
                    "semantic_similarity": semantic_sim,
                    "predicted_answer": pred_answer,
                    "reference_answer": ref_answer
                },
                modality_scores={"vision": 0.6, "language": 0.4},
                cross_modal_alignment=overall_score
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="visual_question_answering",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_vqa_answer(self, data: Any) -> str:
        """Parse VQA answer"""
        if isinstance(data, str):
            return data.strip().lower()
        elif isinstance(data, dict):
            return data.get('answer', data.get('text', '')).strip().lower()
        else:
            return str(data).strip().lower()
    
    def _calculate_exact_match(self, pred_answer: str, ref_answer: str) -> float:
        """Calculate exact match score"""
        return 1.0 if pred_answer == ref_answer else 0.0
    
    def _calculate_semantic_similarity(self, pred_answer: str, ref_answer: str) -> float:
        """Calculate semantic similarity between answers"""
        # Simple word overlap similarity
        pred_words = set(pred_answer.split())
        ref_words = set(ref_answer.split())
        
        if not pred_words and not ref_words:
            return 1.0
        if not pred_words or not ref_words:
            return 0.0
        
        intersection = len(pred_words & ref_words)
        union = len(pred_words | ref_words)
        
        return intersection / union

class AudioEvaluator(MultimodalEvaluatorBase):
    """
    Evaluator for audio processing and synthesis tasks
    
    Supports:
    - Audio classification
    - Speech recognition accuracy
    - Audio quality assessment
    - Music information retrieval
    - Audio synthesis evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get('task_type', 'classification')
        self.quality_metrics = config.get('quality_metrics', ['snr', 'pesq'])
    
    def evaluate(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate audio prediction"""
        if self.task_type == 'classification':
            return self._evaluate_classification(prediction, reference)
        elif self.task_type == 'synthesis':
            return self._evaluate_synthesis(prediction, reference)
        elif self.task_type == 'speech_recognition':
            return self._evaluate_speech_recognition(prediction, reference)
        else:
            raise ValueError(f"Unknown audio task type: {self.task_type}")
    
    def _evaluate_classification(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate audio classification"""
        try:
            pred_data = self._parse_classification_prediction(prediction)
            ref_data = self._parse_classification_reference(reference)
            
            accuracy = self._calculate_classification_accuracy(pred_data, ref_data)
            
            return MultimodalMetricResult(
                metric_name="audio_classification",
                metric_type=MetricType.ACCURACY,
                value=accuracy,
                passed=accuracy >= 0.7,
                details={
                    "accuracy": accuracy,
                    "predicted_class": pred_data.get('class', None),
                    "true_class": ref_data.get('class', None)
                },
                modality_scores={"audio": accuracy}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="audio_classification",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_classification_prediction(self, data: Any) -> Dict[str, Any]:
        """Parse audio classification prediction"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'class': data}
        else:
            return {'class': str(data)}
    
    def _parse_classification_reference(self, data: Any) -> Dict[str, Any]:
        """Parse audio classification reference"""
        return self._parse_classification_prediction(data)
    
    def _calculate_classification_accuracy(self, pred_data: Dict, ref_data: Dict) -> float:
        """Calculate audio classification accuracy"""
        pred_class = pred_data.get('class', None)
        true_class = ref_data.get('class', None)
        
        if pred_class is None or true_class is None:
            return 0.0
        
        return 1.0 if pred_class == true_class else 0.0
    
    def _evaluate_synthesis(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate audio synthesis quality"""
        try:
            pred_audio = self._parse_audio(prediction)
            ref_audio = self._parse_audio(reference)
            
            if pred_audio is None or ref_audio is None:
                return MultimodalMetricResult(
                    metric_name="audio_synthesis",
                    metric_type=MetricType.CUSTOM,
                    value=0.0,
                    passed=False,
                    details={"error": "Could not parse audio"}
                )
            
            # Calculate audio quality metrics
            quality_scores = {}
            
            if 'snr' in self.quality_metrics:
                snr = self._calculate_snr(pred_audio, ref_audio)
                quality_scores['snr'] = snr
            
            if 'mse' in self.quality_metrics:
                mse = self._calculate_audio_mse(pred_audio, ref_audio)
                quality_scores['mse'] = 1.0 / (1.0 + mse)
            
            overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
            
            return MultimodalMetricResult(
                metric_name="audio_synthesis",
                metric_type=MetricType.CUSTOM,
                value=overall_quality,
                passed=overall_quality >= 0.6,
                details=quality_scores,
                audio_quality=overall_quality,
                modality_scores={"audio": overall_quality}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="audio_synthesis",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_audio(self, data: Any) -> Optional[np.ndarray]:
        """Parse audio from various formats"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            try:
                # Try to load from file path
                audio, _ = librosa.load(data, sr=self.audio_sr)
                return audio
            except:
                return None
        elif hasattr(data, 'numpy'):
            # PyTorch tensor
            return data.numpy()
        else:
            return None
    
    def _calculate_snr(self, pred_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(ref_audio ** 2)
        noise_power = np.mean((pred_audio - ref_audio) ** 2)
        
        if noise_power == 0:
            return 1.0
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Normalize to 0-1 range (typical SNR range is 0-60 dB)
        return min(1.0, max(0.0, snr_db / 60))
    
    def _calculate_audio_mse(self, pred_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate Mean Squared Error for audio"""
        return np.mean((pred_audio - ref_audio) ** 2)
    
    def _evaluate_speech_recognition(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate speech recognition accuracy"""
        try:
            pred_text = self._parse_text(prediction)
            ref_text = self._parse_text(reference)
            
            # Calculate Word Error Rate (WER)
            wer = self._calculate_wer(pred_text, ref_text)
            accuracy = 1.0 - wer
            
            # Calculate character-level accuracy
            cer = self._calculate_cer(pred_text, ref_text)
            char_accuracy = 1.0 - cer
            
            overall_score = 0.7 * accuracy + 0.3 * char_accuracy
            
            return MultimodalMetricResult(
                metric_name="speech_recognition",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "word_error_rate": wer,
                    "character_error_rate": cer,
                    "word_accuracy": accuracy,
                    "character_accuracy": char_accuracy,
                    "predicted_text": pred_text,
                    "reference_text": ref_text
                },
                modality_scores={"audio": 0.6, "language": 0.4}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="speech_recognition",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_text(self, data: Any) -> str:
        """Parse text from various formats"""
        if isinstance(data, str):
            return data.strip().lower()
        elif isinstance(data, dict):
            return data.get('text', data.get('transcript', '')).strip().lower()
        else:
            return str(data).strip().lower()
    
    def _calculate_wer(self, pred_text: str, ref_text: str) -> float:
        """Calculate Word Error Rate"""
        pred_words = pred_text.split()
        ref_words = ref_text.split()
        
        if not ref_words:
            return 0.0 if not pred_words else 1.0
        
        # Simple edit distance calculation
        d = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]
        
        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(ref_words) + 1):
                cost = 0 if pred_words[i-1] == ref_words[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        return d[len(pred_words)][len(ref_words)] / len(ref_words)
    
    def _calculate_cer(self, pred_text: str, ref_text: str) -> float:
        """Calculate Character Error Rate"""
        if not ref_text:
            return 0.0 if not pred_text else 1.0
        
        # Simple character-level edit distance
        d = [[0] * (len(ref_text) + 1) for _ in range(len(pred_text) + 1)]
        
        for i in range(len(pred_text) + 1):
            d[i][0] = i
        for j in range(len(ref_text) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred_text) + 1):
            for j in range(1, len(ref_text) + 1):
                cost = 0 if pred_text[i-1] == ref_text[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        return d[len(pred_text)][len(ref_text)] / len(ref_text)

class VideoEvaluator(MultimodalEvaluatorBase):
    """
    Evaluator for video analysis and generation tasks
    
    Supports:
    - Video classification
    - Action recognition
    - Video quality assessment
    - Temporal consistency evaluation
    - Video generation metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get('task_type', 'classification')
        self.temporal_window = config.get('temporal_window', 16)
    
    def evaluate(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate video prediction"""
        if self.task_type == 'classification':
            return self._evaluate_classification(prediction, reference)
        elif self.task_type == 'generation':
            return self._evaluate_generation(prediction, reference)
        elif self.task_type == 'action_recognition':
            return self._evaluate_action_recognition(prediction, reference)
        else:
            raise ValueError(f"Unknown video task type: {self.task_type}")
    
    def _evaluate_classification(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate video classification"""
        try:
            pred_data = self._parse_classification_prediction(prediction)
            ref_data = self._parse_classification_reference(reference)
            
            accuracy = self._calculate_classification_accuracy(pred_data, ref_data)
            
            return MultimodalMetricResult(
                metric_name="video_classification",
                metric_type=MetricType.ACCURACY,
                value=accuracy,
                passed=accuracy >= 0.7,
                details={
                    "accuracy": accuracy,
                    "predicted_class": pred_data.get('class', None),
                    "true_class": ref_data.get('class', None)
                },
                modality_scores={"video": accuracy}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="video_classification",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_classification_prediction(self, data: Any) -> Dict[str, Any]:
        """Parse video classification prediction"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'class': data}
        else:
            return {'class': str(data)}
    
    def _parse_classification_reference(self, data: Any) -> Dict[str, Any]:
        """Parse video classification reference"""
        return self._parse_classification_prediction(data)
    
    def _calculate_classification_accuracy(self, pred_data: Dict, ref_data: Dict) -> float:
        """Calculate video classification accuracy"""
        pred_class = pred_data.get('class', None)
        true_class = ref_data.get('class', None)
        
        if pred_class is None or true_class is None:
            return 0.0
        
        return 1.0 if pred_class == true_class else 0.0
    
    def _evaluate_generation(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate video generation quality"""
        try:
            pred_video = self._parse_video(prediction)
            ref_video = self._parse_video(reference)
            
            if pred_video is None or ref_video is None:
                return MultimodalMetricResult(
                    metric_name="video_generation",
                    metric_type=MetricType.CUSTOM,
                    value=0.0,
                    passed=False,
                    details={"error": "Could not parse videos"}
                )
            
            # Calculate video quality metrics
            spatial_quality = self._calculate_spatial_quality(pred_video, ref_video)
            temporal_consistency = self._calculate_temporal_consistency(pred_video, ref_video)
            
            overall_quality = 0.6 * spatial_quality + 0.4 * temporal_consistency
            
            return MultimodalMetricResult(
                metric_name="video_generation",
                metric_type=MetricType.CUSTOM,
                value=overall_quality,
                passed=overall_quality >= 0.6,
                details={
                    "spatial_quality": spatial_quality,
                    "temporal_consistency": temporal_consistency
                },
                visual_quality=overall_quality,
                modality_scores={"video": overall_quality}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="video_generation",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_video(self, data: Any) -> Optional[np.ndarray]:
        """Parse video from various formats"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            try:
                # Try to load from file path
                cap = cv2.VideoCapture(data)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                return np.array(frames) if frames else None
            except:
                return None
        elif hasattr(data, 'numpy'):
            # PyTorch tensor
            return data.numpy()
        else:
            return None
    
    def _calculate_spatial_quality(self, pred_video: np.ndarray, ref_video: np.ndarray) -> float:
        """Calculate spatial quality of video frames"""
        if pred_video.shape != ref_video.shape:
            return 0.0
        
        frame_scores = []
        num_frames = min(pred_video.shape[0], ref_video.shape[0])
        
        for i in range(num_frames):
            pred_frame = pred_video[i]
            ref_frame = ref_video[i]
            
            # Calculate PSNR for this frame
            mse = np.mean((pred_frame - ref_frame) ** 2)
            if mse == 0:
                frame_scores.append(1.0)
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                normalized_psnr = min(1.0, max(0.0, (psnr - 20) / 20))
                frame_scores.append(normalized_psnr)
        
        return np.mean(frame_scores) if frame_scores else 0.0
    
    def _calculate_temporal_consistency(self, pred_video: np.ndarray, ref_video: np.ndarray) -> float:
        """Calculate temporal consistency between consecutive frames"""
        if pred_video.shape[0] < 2 or ref_video.shape[0] < 2:
            return 0.0
        
        pred_diffs = []
        ref_diffs = []
        
        for i in range(1, min(pred_video.shape[0], ref_video.shape[0])):
            pred_diff = np.mean(np.abs(pred_video[i] - pred_video[i-1]))
            ref_diff = np.mean(np.abs(ref_video[i] - ref_video[i-1]))
            
            pred_diffs.append(pred_diff)
            ref_diffs.append(ref_diff)
        
        if not pred_diffs or not ref_diffs:
            return 0.0
        
        # Calculate correlation between temporal changes
        correlation = np.corrcoef(pred_diffs, ref_diffs)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _evaluate_action_recognition(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate action recognition in videos"""
        try:
            pred_action = self._parse_action(prediction)
            ref_action = self._parse_action(reference)
            
            # Calculate action classification accuracy
            accuracy = 1.0 if pred_action == ref_action else 0.0
            
            # Calculate temporal localization if available
            temporal_score = self._calculate_temporal_localization(prediction, reference)
            
            overall_score = 0.7 * accuracy + 0.3 * temporal_score
            
            return MultimodalMetricResult(
                metric_name="action_recognition",
                metric_type=MetricType.ACCURACY,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "action_accuracy": accuracy,
                    "temporal_localization_score": temporal_score,
                    "predicted_action": pred_action,
                    "reference_action": ref_action
                },
                modality_scores={"video": overall_score}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="action_recognition",
                metric_type=MetricType.ACCURACY,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_action(self, data: Any) -> str:
        """Parse action from prediction/reference"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return data.get('action', data.get('class', ''))
        else:
            return str(data)
    
    def _calculate_temporal_localization(self, prediction: Any, reference: Any) -> float:
        """Calculate temporal localization accuracy"""
        if isinstance(prediction, dict) and isinstance(reference, dict):
            pred_start = prediction.get('start_time', 0)
            pred_end = prediction.get('end_time', 0)
            ref_start = reference.get('start_time', 0)
            ref_end = reference.get('end_time', 0)
            
            if pred_start == 0 and pred_end == 0 and ref_start == 0 and ref_end == 0:
                return 1.0  # No temporal info available
            
            # Calculate IoU for temporal segments
            intersection_start = max(pred_start, ref_start)
            intersection_end = min(pred_end, ref_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union = (pred_end - pred_start) + (ref_end - ref_start) - intersection
            
            return intersection / union if union > 0 else 0.0
        
        return 1.0  # Default if no temporal info

class CrossModalEvaluator(MultimodalEvaluatorBase):
    """
    Evaluator for cross-modal tasks and alignment
    
    Supports:
    - Image-text retrieval
    - Audio-visual synchronization
    - Video-text alignment
    - Cross-modal embeddings evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get('task_type', 'retrieval')
        self.similarity_metric = config.get('similarity_metric', 'cosine')
    
    def evaluate(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate cross-modal prediction"""
        if self.task_type == 'retrieval':
            return self._evaluate_retrieval(prediction, reference)
        elif self.task_type == 'synchronization':
            return self._evaluate_synchronization(prediction, reference)
        elif self.task_type == 'alignment':
            return self._evaluate_alignment(prediction, reference)
        else:
            raise ValueError(f"Unknown cross-modal task type: {self.task_type}")
    
    def _evaluate_retrieval(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate cross-modal retrieval performance"""
        try:
            pred_data = self._parse_retrieval_prediction(prediction)
            ref_data = self._parse_retrieval_reference(reference)
            
            # Calculate retrieval metrics
            recall_at_1 = self._calculate_recall_at_k(pred_data, ref_data, k=1)
            recall_at_5 = self._calculate_recall_at_k(pred_data, ref_data, k=5)
            recall_at_10 = self._calculate_recall_at_k(pred_data, ref_data, k=10)
            
            # Calculate mean reciprocal rank
            mrr = self._calculate_mrr(pred_data, ref_data)
            
            overall_score = 0.4 * recall_at_1 + 0.3 * recall_at_5 + 0.2 * recall_at_10 + 0.1 * mrr
            
            return MultimodalMetricResult(
                metric_name="cross_modal_retrieval",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.6,
                details={
                    "recall_at_1": recall_at_1,
                    "recall_at_5": recall_at_5,
                    "recall_at_10": recall_at_10,
                    "mean_reciprocal_rank": mrr
                },
                cross_modal_alignment=overall_score,
                modality_scores={"cross_modal": overall_score}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="cross_modal_retrieval",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_retrieval_prediction(self, data: Any) -> Dict[str, Any]:
        """Parse retrieval prediction"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            return {'rankings': data}
        else:
            return {}
    
    def _parse_retrieval_reference(self, data: Any) -> Dict[str, Any]:
        """Parse retrieval reference"""
        return self._parse_retrieval_prediction(data)
    
    def _calculate_recall_at_k(self, pred_data: Dict, ref_data: Dict, k: int) -> float:
        """Calculate Recall@K"""
        pred_rankings = pred_data.get('rankings', [])
        relevant_items = ref_data.get('relevant_items', [])
        
        if not relevant_items:
            return 1.0  # No relevant items to retrieve
        
        top_k_predictions = pred_rankings[:k]
        found_relevant = len(set(top_k_predictions) & set(relevant_items))
        
        return found_relevant / len(relevant_items)
    
    def _calculate_mrr(self, pred_data: Dict, ref_data: Dict) -> float:
        """Calculate Mean Reciprocal Rank"""
        pred_rankings = pred_data.get('rankings', [])
        relevant_items = ref_data.get('relevant_items', [])
        
        if not relevant_items:
            return 1.0
        
        for i, pred_item in enumerate(pred_rankings):
            if pred_item in relevant_items:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _evaluate_synchronization(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate audio-visual synchronization"""
        try:
            pred_sync = self._parse_sync_prediction(prediction)
            ref_sync = self._parse_sync_reference(reference)
            
            # Calculate temporal alignment score
            temporal_alignment = self._calculate_temporal_alignment(pred_sync, ref_sync)
            
            # Calculate cross-correlation if available
            cross_correlation = self._calculate_cross_correlation(pred_sync, ref_sync)
            
            overall_score = 0.6 * temporal_alignment + 0.4 * cross_correlation
            
            return MultimodalMetricResult(
                metric_name="audio_visual_synchronization",
                metric_type=MetricType.CUSTOM,
                value=overall_score,
                passed=overall_score >= 0.7,
                details={
                    "temporal_alignment": temporal_alignment,
                    "cross_correlation": cross_correlation
                },
                cross_modal_alignment=overall_score,
                modality_scores={"audio": 0.5, "video": 0.5}
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="audio_visual_synchronization",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_sync_prediction(self, data: Any) -> Dict[str, Any]:
        """Parse synchronization prediction"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, (int, float)):
            return {'offset': float(data)}
        else:
            return {}
    
    def _parse_sync_reference(self, data: Any) -> Dict[str, Any]:
        """Parse synchronization reference"""
        return self._parse_sync_prediction(data)
    
    def _calculate_temporal_alignment(self, pred_sync: Dict, ref_sync: Dict) -> float:
        """Calculate temporal alignment score"""
        pred_offset = pred_sync.get('offset', 0.0)
        ref_offset = ref_sync.get('offset', 0.0)
        
        offset_diff = abs(pred_offset - ref_offset)
        tolerance = 0.1  # 100ms tolerance
        
        return max(0.0, 1.0 - offset_diff / tolerance)
    
    def _calculate_cross_correlation(self, pred_sync: Dict, ref_sync: Dict) -> float:
        """Calculate cross-correlation between modalities"""
        if 'correlation' in pred_sync and 'correlation' in ref_sync:
            pred_corr = pred_sync['correlation']
            ref_corr = ref_sync['correlation']
            
            return 1.0 - abs(pred_corr - ref_corr)
        
        return 0.5  # Default neutral score
    
    def _evaluate_alignment(self, prediction: Any, reference: Any) -> MultimodalMetricResult:
        """Evaluate general cross-modal alignment"""
        try:
            pred_embeddings = self._parse_embeddings(prediction)
            ref_embeddings = self._parse_embeddings(reference)
            
            # Calculate embedding similarity
            similarity = self._calculate_embedding_similarity(pred_embeddings, ref_embeddings)
            
            return MultimodalMetricResult(
                metric_name="cross_modal_alignment",
                metric_type=MetricType.CUSTOM,
                value=similarity,
                passed=similarity >= 0.7,
                details={
                    "embedding_similarity": similarity,
                    "similarity_metric": self.similarity_metric
                },
                cross_modal_alignment=similarity,
                semantic_consistency=similarity
            )
            
        except Exception as e:
            return MultimodalMetricResult(
                metric_name="cross_modal_alignment",
                metric_type=MetricType.CUSTOM,
                value=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    def _parse_embeddings(self, data: Any) -> Optional[np.ndarray]:
        """Parse embeddings from various formats"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict) and 'embedding' in data:
            return np.array(data['embedding'])
        elif hasattr(data, 'numpy'):
            return data.numpy()
        else:
            return None
    
    def _calculate_embedding_similarity(self, pred_emb: Optional[np.ndarray], ref_emb: Optional[np.ndarray]) -> float:
        """Calculate similarity between embeddings"""
        if pred_emb is None or ref_emb is None:
            return 0.0
        
        if pred_emb.shape != ref_emb.shape:
            return 0.0
        
        if self.similarity_metric == 'cosine':
            # Reshape for sklearn cosine_similarity
            pred_emb = pred_emb.reshape(1, -1)
            ref_emb = ref_emb.reshape(1, -1)
            similarity = cosine_similarity(pred_emb, ref_emb)[0, 0]
            return max(0.0, similarity)
        elif self.similarity_metric == 'euclidean':
            distance = np.linalg.norm(pred_emb - ref_emb)
            # Convert distance to similarity (0-1 range)
            return max(0.0, 1.0 / (1.0 + distance))
        else:
            # Default to cosine similarity
            pred_norm = np.linalg.norm(pred_emb)
            ref_norm = np.linalg.norm(ref_emb)
            
            if pred_norm == 0 or ref_norm == 0:
                return 0.0
            
            cosine_sim = np.dot(pred_emb, ref_emb) / (pred_norm * ref_norm)
            return max(0.0, cosine_sim)

# Register multimodal evaluators with the grading system
def register_multimodal_evaluators():
    """Register multimodal evaluators with the OpenEvals grading system"""
    from openevals.core.graders import register_grader
    from openevals.config.data_structures import MetricType
    
    def image_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = ImageEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def audio_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = AudioEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def video_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = VideoEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    def cross_modal_grader(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        evaluator = CrossModalEvaluator(kwargs)
        result = evaluator.evaluate(system_output, reference)
        return {
            'value': result.value,
            'passed': result.passed,
            'details': result.details
        }
    
    # Register the graders
    register_grader("image_evaluation", image_grader, MetricType.CUSTOM)
    register_grader("audio_evaluation", audio_grader, MetricType.CUSTOM)
    register_grader("video_evaluation", video_grader, MetricType.CUSTOM)
    register_grader("cross_modal_evaluation", cross_modal_grader, MetricType.CUSTOM)

# Auto-register on import
if MULTIMODAL_DEPS_AVAILABLE:
    try:
        register_multimodal_evaluators()
        logger.info("Multimodal evaluators registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register multimodal evaluators: {e}")
else:
    logger.info("Multimodal evaluators available in limited mode (dependencies not installed)") 