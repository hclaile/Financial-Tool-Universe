"""
Sentiment analysis module - based on BERT/FinBERT
Supports financial text sentiment analysis, outputting positive/negative/neutral sentiment labels and confidence scores
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from types import SimpleNamespace
from collections import Counter

import pandas as pd
import numpy as np


def _print(*args, **kwargs):
    """Print to stderr to avoid interfering with MCP JSON-RPC communication"""
    print(*args, file=sys.stderr, **kwargs)


DEFAULT_LOCAL_MODEL = os.environ.get("FTU_FINBERT_MODEL", "")
# When local path is empty or does not exist, fallback to loading from HuggingFace model
DEFAULT_HF_MODEL = "yiyanghkust/finbert-tone"


class SentimentAnalyzer:
    """Sentiment analyzer class supporting local BERT/FinBERT models"""

    def __init__(self, model_name: str = "finbert-tone", device: str = "auto", model_path: Optional[str] = None):
        """
        Initialize the sentiment analyzer

        Args:
            model_name: Model name (default finbert-tone)
            device: Running device(cpu, cuda, auto)
            model_path: Local model directory, default configured FTU_FINBERT_MODEL path
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        raw_path = (model_path or "").strip() or DEFAULT_LOCAL_MODEL
        self.model_path = os.path.expanduser(raw_path) if raw_path else ""
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Get running device"""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """Load model: load locally if local path is valid (offline), otherwise fallback to loading from HuggingFace(yiyanghkust/finbert-tone)"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            _print(f"Error: transformers is not installed. Please install: pip install transformers")
            raise ImportError("transformers Not installed. Please run: pip install transformers") from e

        try:
            import torch
        except ImportError as e:
            _print(f"Error: torch is not installed. Please install: pip install torch")
            raise ImportError("torch Not installed. Please run: pip install torch") from e

        use_local = bool(self.model_path and Path(self.model_path).exists())
        load_from = self.model_path if use_local else DEFAULT_HF_MODEL

        if use_local:
            try:
                _print(f"Loading local model: {load_from}")
                self.tokenizer = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(load_from, local_files_only=True)
            except Exception as e:
                _print(f"Local model loading failed, attempting fallback from HuggingFace: {e}")
                use_local = False
                load_from = DEFAULT_HF_MODEL

        if not use_local:
            _print(f"Local path is empty or unavailable, loading from HuggingFace: {load_from}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_from)
                self.model = AutoModelForSequenceClassification.from_pretrained(load_from)
            except Exception as e:
                _print(f"HuggingFace Model loading failed: {e}")
                raise RuntimeError(f"Model loading failed (neither local nor HuggingFace available): {e}") from e

        self.model.to(self.device)
        self.model.eval()
        _print(f"Model loaded successfully, running device: {self.device}")

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on a single text

        Args:
            text: Input text

        Returns:
            A dictionary containing emotion labels and confidence scores
        """
        import torch

        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        # Label mapping(finbert-tone: 0 neutral, 1 positive, 2 negative)
        if "finbert" in self.model_name.lower():
            label_map = {0: "neutral", 1: "positive", 2: "negative"}
        else:
            # Common order compatible with other three-class models
            label_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Create reverse mapping: from label name to index
        label_to_idx = {v: k for k, v in label_map.items()}

        # Get probabilities for all classes
        all_probs = {
            "positive": float(probs[0][label_to_idx.get("positive", 0)].item()) if "positive" in label_to_idx else 0.0,
            "negative": float(probs[0][label_to_idx.get("negative", 1)].item()) if "negative" in label_to_idx else 0.0,
            "neutral": float(probs[0][label_to_idx.get("neutral", 2)].item()) if "neutral" in label_to_idx else 0.0
        }

        return {
            "text": text,
            "sentiment": label_map.get(prediction, "unknown"),
            "confidence": float(confidence),
            "probabilities": all_probs
        }

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Batch predict text sentiment

        Args:
            texts: List of texts
            batch_size: Batch size

        Returns:
            List of prediction results
        """
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            _print(f"Process batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

            for text in batch_texts:
                try:
                    result = self.predict_text(text)
                    results.append(result)
                except Exception as e:
                    _print(f"Error processing text: {str(e)}")
                    results.append({
                        "text": text,
                        "sentiment": "error",
                        "confidence": 0.0,
                        "error": str(e)
                    })

        return results


def _load_input_data(input_path: str, text_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load input data

    Args:
        input_path: Input file path(CSV/JSON/TXT)
        text_column: Text column name (if not specified, auto-detect))

    Returns:
        DataFrame
    """
    input_path = os.path.expanduser(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Load based on file type
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    elif input_path.endswith((".txt", ".text")):
        # TXT Each line of the file is a text
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({"text": texts})
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    # Auto-detect text column
    if text_column is None:
        possible_columns = ["text", "content", "sentence", "news", "title", "description"]
        for col in possible_columns:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            # Use first column
            text_column = df.columns[0]
            _print(f"Warning: text column not specified, using first column: {text_column}")

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' does not exist. Available columns: {list(df.columns)}")

    return df, text_column


def _save_results(results: List[Dict[str, Any]], output_path: str, format: str = "auto"):
    """
    Save analysis results

    Args:
        results: List of prediction results
        output_path: output path
        format: Output format(csv, json, auto)
    """
    output_path = os.path.expanduser(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # convert to DataFrame
    df = pd.DataFrame(results)

    # auto-detect format
    if format == "auto":
        if output_path.endswith(".csv"):
            format = "csv"
        elif output_path.endswith(".json"):
            format = "json"
        else:
            format = "json"  # default use JSON

    # save
    if format == "csv":
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif format == "json":
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
    else:
        raise ValueError(f"unsupported output format: {format}")

    _print(f"result saved to: {output_path}")


def _analyze_dataframe(
    df: pd.DataFrame,
    text_col: str,
    analyzer: SentimentAnalyzer,
    batch_size: int,
    output_path: str,
    output_format: str,
) -> Dict[str, Any]:
    """
    perform sentiment analysis on the given DataFrame and save the results, return statistics
    """
    texts = df[text_col].astype(str).tolist()
    _print(f"loaded {len(texts)} text entries")

    # execute prediction
    _print("start sentiment analysis...")
    results = analyzer.predict_batch(texts, batch_size=batch_size)

    # add additional columns from the original data
    for i, result in enumerate(results):
        for col in df.columns:
            if col != text_col:
                result[col] = df.iloc[i][col]

    # save results
    _save_results(results, output_path, output_format)

    # statistics
    sentiment_counts = pd.DataFrame(results)["sentiment"].value_counts().to_dict()
    avg_confidence = pd.DataFrame(results)["confidence"].mean()

    _print(f"number of processed texts: {len(results)}")
    _print(f"sentiment distribution: {sentiment_counts}")
    _print(f"average confidence: {avg_confidence:.4f}")
    _print(f"result file: {output_path}")

    return {
        "total_texts": len(results),
        "sentiment_distribution": sentiment_counts,
        "average_confidence": float(avg_confidence),
        "output_path": output_path,
    }


def run(args: SimpleNamespace) -> None:
    """
    execute sentiment analysis

    Args:
        args: namespace containing the following properties:
            - input_path: input file path (required)
            - output_path: output file path (optional, default in the same directory as input file)
            - text_column: text column name (optional, auto-detect)
            - model_name: model name (optional, default finbert)
            - batch_size: batch size (optional, default 16)
            - device: Run device (optional, default auto)
            - output_format: Output format (optional, default auto)
    """
    try:
        # Get parameters
        input_path = getattr(args, "input_path", None) or getattr(args, "input", None)
        input_token = getattr(args, "input_token", None)

        if not input_path and input_token is None:
            raise ValueError("Either input_path or input_token must be provided")

        output_path = getattr(args, "output_path", None) or getattr(args, "output", None)
        text_column = getattr(args, "text_column", None)
        model_name = getattr(args, "model_name", "finbert-tone")
        model_path = getattr(args, "model_path", None)
        batch_size = getattr(args, "batch_size", 16)
        device = getattr(args, "device", "auto")
        output_format = getattr(args, "output_format", "auto")

        _print(f"=" * 60)
        _print(f"Sentiment analysis task started")
        _print(f"Input file: {input_path}" if input_path else "Input is a single text segment (input_token)")
        _print(f"Model: {model_name}")
        _print(f"=" * 60)

        # Initialize analyzer (shared model instance for directory or single file)
        analyzer = SentimentAnalyzer(model_name=model_name, device=device, model_path=model_path)

        # Case 1: input_token for single text
        if not input_path:
            if not output_path:
                output_path = "input_token_sentiment.json"
            _print("Analyze text using input_token")
            df = pd.DataFrame({"text": [str(input_token)]})
            text_col = "text"
            stats = _analyze_dataframe(
                df=df,
                text_col=text_col,
                analyzer=analyzer,
                batch_size=batch_size,
                output_path=output_path,
                output_format=output_format,
            )
            sentiment_counts = stats["sentiment_distribution"]
            avg_confidence = stats["average_confidence"]
            total_texts = stats["total_texts"]

        else:
            # input_path Exists: could be a file or a directory
            input_path_expanded = os.path.expanduser(input_path)

            # Case 2: directory mode, process all json/txt files under the directory
            if os.path.isdir(input_path_expanded):
                input_dir = Path(input_path_expanded)
                _print(f"Directory input detected, will iterate through json/txt files within: {input_dir}")

                # Determine output directory
                if output_path:
                    out_dir = Path(os.path.expanduser(output_path))
                else:
                    out_dir = input_dir / "_sentiment"
                out_dir.mkdir(parents=True, exist_ok=True)

                total_texts = 0
                sentiment_counter: Counter = Counter()
                avg_conf_list: List[float] = []
                processed_files: List[str] = []

                # Iterate through json/txt files
                for file in sorted(input_dir.iterdir()):
                    if not file.is_file():
                        continue
                    if file.suffix.lower() not in [".json", ".txt", ".text"]:
                        continue

                    single_output = out_dir / f"{file.stem}_sentiment.json"
                    _print(f"--- Process file: {file.name} ---")
                    df, text_col = _load_input_data(str(file), text_column)
                    stats = _analyze_dataframe(
                        df=df,
                        text_col=text_col,
                        analyzer=analyzer,
                        batch_size=batch_size,
                        output_path=str(single_output),
                        output_format=output_format,
                    )
                    total_texts += stats["total_texts"]
                    sentiment_counter.update(stats["sentiment_distribution"])
                    avg_conf_list.append(stats["average_confidence"])
                    processed_files.append(str(file))

                if total_texts == 0:
                    raise ValueError(f"No json/txt files found to process in directory {input_dir}")

                sentiment_counts = dict(sentiment_counter)
                avg_confidence = float(sum(avg_conf_list) / len(avg_conf_list)) if avg_conf_list else 0.0
                _print(f"Directory mode completed, processed files count: {len(processed_files)}, total text count: {total_texts}")
                _print(f"Output directory: {out_dir}")
                output_path = str(out_dir)  # Returned as directory

            # Case 3: single file mode (original logic)
            else:
                # Generate default output path
                if not output_path:
                    input_path_obj = Path(input_path)
                    output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_sentiment.json")

                _print("Load input data...")
                df, text_col = _load_input_data(input_path, text_column)
                stats = _analyze_dataframe(
                    df=df,
                    text_col=text_col,
                    analyzer=analyzer,
                    batch_size=batch_size,
                    output_path=output_path,
                    output_format=output_format,
                )
                sentiment_counts = stats["sentiment_distribution"]
                avg_confidence = stats["average_confidence"]
                total_texts = stats["total_texts"]

        _print(f"=" * 60)
        _print(f"Sentiment analysis completed!")
        _print(f"number of processed texts: {total_texts}")
        _print(f"sentiment distribution: {sentiment_counts}")
        _print(f"average confidence: {avg_confidence:.4f}")
        _print(f"result file: {output_path}")
        _print(f"=" * 60)

        # Return statistics (for API response)
        return {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "total_texts": total_texts,
            "sentiment_distribution": sentiment_counts,
            "average_confidence": float(avg_confidence),
            "model_name": model_name
        }

    except Exception as e:
        _print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    # Test code
    test_args = SimpleNamespace(
        input_path="./test_texts.txt",
        output_path="./test_results.json",
        model_name="finbert",
        batch_size=4,
        device="auto"
    )
    run(test_args)
