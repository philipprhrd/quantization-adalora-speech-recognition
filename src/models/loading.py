from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
from .quantization import build_quant_config
import logging

def load_model_and_processor(model_id: str, quantization="none"):
    bnb_config, extra = build_quant_config(quantization)

    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        class SimpleProcessor:
            def __init__(self, tok, fe): self.tokenizer, self.feature_extractor = tok, fe
        processor = SimpleProcessor(tokenizer, feature_extractor)
    
    logging.info(f"Device: {extra.get("device_map")}")

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            quantization_config=bnb_config if bnb_config else None,
            device_map=extra.get("device_map") if bnb_config else None
        )

        return model, processor
    except Exception as e:
        logging.error(f"Error loading model '{model_id}': {e}")