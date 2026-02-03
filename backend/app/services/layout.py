"""
D2: FUNSD Layout Understanding Service
Uses LayoutLMv3 to parse document structure
"""

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

class LayoutParser:
    def __init__(self):
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load LayoutLMv3 fine-tuned on FUNSD"""
        try:
            model_name = "microsoft/layoutlmv3-base-finetuned-funsd"
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
            logger.info("LayoutLMv3 (FUNSD) loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LayoutLMv3: {e}")
            raise
    
    def parse(self, image: Image.Image, text: str = None):
        """
        Parse document layout using D2 (FUNSD)
        
        Returns:
            - entities: headers, questions, answers, other
            - bounding boxes
            - confidence scores
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Layout model not loaded")
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # If no text provided, use OCR (simplified - use pytesseract in production)
        if text is None:
            text = "Document text not provided"  # Placeholder
        
        # Process
        encoding = self.processor(
            image,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Decode predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())
        
        # Map to labels
        id2label = self.model.config.id2label
        entities = []
        
        for token, pred_id in zip(tokens, predictions):
            if token.startswith("##"):  # Skip subword tokens for simplicity
                continue
            
            label = id2label[pred_id]
            if label != "O":  # Not "outside"
                entities.append({
                    "token": token.replace("▁", ""),  # Remove RoBERTa prefix
                    "label": label,
                    "label_type": label.split("-")[-1] if "-" in label else label
                })
        
        # Group by structure
        grouped = self._group_entities(entities)
        
        return {
            "raw_entities": entities[:50],  # Limit for response size
            "grouped": grouped,
            "entity_counts": {
                "headers": len(grouped["headers"]),
                "questions": len(grouped["questions"]),
                "answers": len(grouped["answers"]),
                "other": len(grouped["other"])
            },
            "model": "microsoft/layoutlmv3-base-finetuned-funsd",
            "dataset": "FUNSD"
        }
    
    def _group_entities(self, entities):
        """Group tokens by semantic type"""
        groups = {
            "headers": [],
            "questions": [],
            "answers": [],
            "other": []
        }
        
        current_group = []
        current_label = None
        
        for ent in entities:
            label = ent["label"]
            token = ent["token"]
            
            # Start of new entity
            if label.startswith("B-"):
                if current_group:
                    self._add_to_group(groups, current_label, current_group)
                current_group = [token]
                current_label = label[2:]  # Remove B- prefix
            
            # Continuation of entity
            elif label.startswith("I-") and current_label == label[2:]:
                current_group.append(token)
            
            # Single token entity or other
            else:
                if current_group:
                    self._add_to_group(groups, current_label, current_group)
                    current_group = []
                    current_label = None
                self._add_to_group(groups, label, [token])
        
        # Don't forget last group
        if current_group:
            self._add_to_group(groups, current_label, current_group)
        
        # Join tokens into strings
        for key in groups:
            groups[key] = [" ".join(g) for g in groups[key]]
        
        return groups
    
    def _add_to_group(self, groups, label_type, tokens):
        """Add token group to appropriate category"""
        if not tokens or not label_type:
            return
        
        label_type = label_type.lower()
        
        if "header" in label_type:
            groups["headers"].append(tokens)
        elif "question" in label_type:
            groups["questions"].append(tokens)
        elif "answer" in label_type:
            groups["answers"].append(tokens)
        else:
            groups["other"].append(tokens)

# Singleton instance
layout_parser = LayoutParser()