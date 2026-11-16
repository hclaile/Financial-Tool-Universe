"""
Technical metrics calculation module - based on TA-Lib library
Supports over 150 technical indicators, covering trends, momentum, volatility, and pattern recognition
"""
import os
import sys
import json
import inspect
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Try to import TA-Lib
try:
    import talib
except ImportError:
    talib = None
    print("Warning: TA-Lib is not installed, please run: pip install TA-Lib", file=sys.stderr)

# Import indicator whitelist
try:
    from indicator_whitelist import SUPPORTED_INDICATORS
except ImportError:
    # If import fails, use simplified list
    SUPPORTED_INDICATORS = [
        "SMA", "EMA", "MACD", "RSI", "BBANDS", "STOCH",
        "ATR", "ADX", "CCI", "MOM", "ROC", "WILLR"
    ]


def _print(*args, **kwargs):
    """Print to stderr to avoid interfering with MCP JSON-RPC communication"""
    print(*args, file=sys.stderr, **kwargs)


def _is_datetime_like(obj: Any) -> bool:
    """
    Unified detection function: determines if an object is a datetime type

    Supports detection:
    - pandas Timestamp
    - Python datetime/date
    - pandas Period
    - datetime64 type
    - Third-party library datetime objects (arrow, pendulum, etc., detected via strftime method)
    - numpy datetime64

    Args:
        obj: The object to be detected

    Returns:
        True True if it is a datetime type, False otherwise
    """
    if obj is None:
        return False

    # Standard type check
    if isinstance(obj, (pd.Timestamp, datetime, pd.Period)):
        return True

    # Check if it is a datetime64 type
    if pd.api.types.is_datetime64_any_dtype(type(obj)):
        return True

    # Check if it is numpy datetime64
    if isinstance(obj, np.datetime64):
        return True

    # Check if it has a strftime method (third-party libraries such as arrow, pendulum, etc.)
    if hasattr(obj, 'strftime') and callable(getattr(obj, 'strftime', None)):
        # Additional verification: attempt to call strftime to confirm it is actually a datetime object
        try:
            obj.strftime('%Y-%m-%d')
            return True
        except (AttributeError, TypeError, ValueError):
            pass

    # Check if it is a date type
    try:
        from datetime import date
        if isinstance(obj, date):
            return True
    except ImportError:
        pass

    return False


def _convert_datetime_to_string(obj: Any) -> str:
    """
    Unified conversion function: convert datetime objects to strings

    Supports all types detectable by _is_datetime_like

    Args:
        obj: datetime object

    Returns:
        Formatted string
    """
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, np.datetime64):
        try:
            return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(obj)
    elif pd.api.types.is_datetime64_any_dtype(type(obj)):
        try:
            return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(obj)
    elif hasattr(obj, 'strftime'):
        # Third-party library datetime objects
        try:
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                return obj.strftime('%Y-%m-%d')
            except Exception:
                return str(obj)
    else:
        # Final fallback: convert to string
        return str(obj)


class TimestampJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to automatically handle Timestamp and datetime objects
    As a last resort, ensure all datetime objects can be properly serialized

    Enhanced version: use a unified type detection and conversion function to handle more edge cases
    """
    def default(self, obj):
        # Use a unified datetime detection and conversion function
        if _is_datetime_like(obj):
            return _convert_datetime_to_string(obj)

        # Handle numpy numeric types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            # Recursively process elements within arrays (which may contain datetimes))
            try:
                return obj.tolist()
            except (TypeError, ValueError):
                # If tolist fails, try converting element by element
                return [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in obj]
        # Handle numpy datetime64 arrays
        elif isinstance(obj, (np.datetime64,)):
            return _convert_datetime_to_string(obj)
        # Handle other types
        return super().default(obj)


def _convert_timestamps_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all Timestamp objects in the DataFrame to strings to support JSON serialization

    Improved version: more thoroughly handle all possible datetime types, including:
    - DatetimeIndex
    - datetime64 column types
    - object Timestamp objects in column types
    - Period objects

    Args:
        df: Input DataFrame

    Returns:
        The converted DataFrame (all Timestamp/Period columns have been converted to strings)
    """
    df = df.copy()

    # Handle Timestamp/Period in the index (including MultiIndex)
    if isinstance(df.index, pd.MultiIndex):
        # MultiIndex:check each level
        try:
            new_levels_data = []
            for i in range(df.index.nlevels):
                level_values = df.index.get_level_values(i)
                # check if the first value is of datetime type
                if len(level_values) > 0 and _is_datetime_like(level_values[0]):
                    new_levels_data.append([_convert_datetime_to_string(v) if _is_datetime_like(v) else v for v in level_values])
                else:
                    new_levels_data.append(list(level_values))
            # rebuild MultiIndex
            df.index = pd.MultiIndex.from_arrays(new_levels_data)
        except Exception as e:
            # If MultiIndex conversion fails, log a warning and continue
            _print(f"  ⚠️  Warning: MultiIndex datetime conversion failed: {str(e)[:100]}")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    elif isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.astype(str)
    elif hasattr(df.index, 'dtype'):
        if pd.api.types.is_datetime64_any_dtype(df.index.dtype):
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        elif pd.api.types.is_period_dtype(df.index.dtype):
            df.index = df.index.astype(str)
        # Check if index values themselves are datetime objects (object-type indices))
        elif len(df.index) > 0:
            sample_idx = df.index[0]
            if _is_datetime_like(sample_idx):
                df.index = [_convert_datetime_to_string(v) if _is_datetime_like(v) else v for v in df.index]

    # Handle all columns for Timestamp/Period
    for col in df.columns:
        col_dtype = df[col].dtype

        # Case 1: explicit datetime64 type
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            try:
                # Convert to string, keeping only the date part (to avoid inconsistent formatting due to time components))
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
            except Exception:
                # If conversion fails, try direct string conversion
                df[col] = df[col].astype(str)

        # Case 2: Period type
        elif pd.api.types.is_period_dtype(col_dtype):
            df[col] = df[col].astype(str)

        # Case 3: object type (may contain Timestamp objects)
        elif col_dtype == 'object':
            try:
                # Use a unified type detection function for more thorough checking
                has_datetime = False

                # Check the first few non-empty values (up to 20 values, to improve detection coverage)
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_size = min(20, len(non_null_values))
                    for val in non_null_values.head(sample_size):
                        if _is_datetime_like(val):
                            has_datetime = True
                            break
                        # Check nested structures (lists, tuples, dictionaries)
                        elif isinstance(val, (list, tuple)):
                            for item in val[:5]:  # Check only the first 5 elements
                                if _is_datetime_like(item):
                                    has_datetime = True
                                    break
                            if has_datetime:
                                break
                        elif isinstance(val, dict):
                            for item in list(val.values())[:5]:  # Check only the first 5 values
                                if _is_datetime_like(item):
                                    has_datetime = True
                                    break
                            if has_datetime:
                                break

                # If a datetime object is detected, convert it
                if has_datetime:
                    def _convert_object_type(x):
                        if pd.isna(x):
                            return x
                        # Use a unified detection and conversion function
                        if _is_datetime_like(x):
                            return _convert_datetime_to_string(x)
                        # Handle nested structures
                        elif isinstance(x, (list, tuple)):
                            converted = [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in x]
                            return tuple(converted) if isinstance(x, tuple) else converted
                        elif isinstance(x, dict):
                            return {k: (_convert_datetime_to_string(v) if _is_datetime_like(v) else v) for k, v in x.items()}
                        else:
                            return x

                    df[col] = df[col].apply(_convert_object_type)
            except Exception as e:
                # If conversion fails, try a more general approach
                try:
                    # More aggressive conversion: attempt conversion for all values using a unified function)
                    def _convert_aggressive(x):
                        if pd.isna(x):
                            return x
                        # Use a unified detection and conversion function
                        if _is_datetime_like(x):
                            return _convert_datetime_to_string(x)
                        # Handle nested structures
                        elif isinstance(x, (list, tuple)):
                            converted = [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in x]
                            return tuple(converted) if isinstance(x, tuple) else converted
                        elif isinstance(x, dict):
                            return {k: (_convert_datetime_to_string(v) if _is_datetime_like(v) else v) for k, v in x.items()}
                        # Try parsing strings as datetime (use with caution to avoid incorrect conversions)
                        elif isinstance(x, str) and len(x) > 0:
                            # Try parsing only strings that look like dates
                            if any(char.isdigit() for char in x[:10]):  # The first 10 characters contain digits
                                try:
                                    dt = pd.to_datetime(x)
                                    if not pd.isna(dt):
                                        return dt.strftime('%Y-%m-%d')
                                except:
                                    pass
                        return x

                    df[col] = df[col].apply(_convert_aggressive)
                except Exception as e2:
                    # If it still fails, log a warning and continue
                    _print(f"  ⚠️  Warning: Timestamp conversion for column '{col}' failed: {str(e2)[:100]}")
                    pass  # If it still fails, keep the original value

    return df


def _convert_dict_timestamps_to_strings(obj: Any, seen: Optional[set] = None) -> Any:
    """
    Recursively convert Timestamp objects in dictionaries and lists to strings

    Enhanced version: use a unified type detection and conversion function to more thoroughly handle all possible datetime types

    Args:
        obj: Could be dict, list, Timestamp, or other types
        seen: A set used for detecting cyclic references (internal use)

    Returns:
        The converted object
    """
    # Prevent cyclic references
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return "[Circular Reference]"
    seen.add(obj_id)

    try:
        # Use a unified datetime detection and conversion function
        if _is_datetime_like(obj):
            result = _convert_datetime_to_string(obj)
            seen.remove(obj_id)
            return result
        # Handle dictionaries
        elif isinstance(obj, dict):
            result = {k: _convert_dict_timestamps_to_strings(v, seen) for k, v in obj.items()}
            seen.remove(obj_id)
            return result
        # Process list
        elif isinstance(obj, (list, tuple)):
            result = [_convert_dict_timestamps_to_strings(item, seen) for item in obj]
            seen.remove(obj_id)
            # Preserve original type (list or tuple)
            if isinstance(obj, tuple):
                return tuple(result)
            return result
        # Handle numpy numeric types
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            seen.remove(obj_id)
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            seen.remove(obj_id)
            return float(obj)
        elif isinstance(obj, np.bool_):
            seen.remove(obj_id)
            return bool(obj)
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            result = obj.tolist()
            result = _convert_dict_timestamps_to_strings(result, seen)
            seen.remove(obj_id)
            return result
        # Handle other types (check if has strftime method)
        elif hasattr(obj, 'strftime'):
            try:
                result = obj.strftime('%Y-%m-%d %H:%M:%S')
                seen.remove(obj_id)
                return result
            except Exception:
                seen.remove(obj_id)
                return str(obj)
        # Other types, return directly
        else:
            seen.remove(obj_id)
            return obj
    except Exception as e:
        # If conversion fails, return string representation
        seen.discard(obj_id)  # Ensure removal from seen
        try:
            return str(obj)
        except Exception:
            return f"[Conversion Error: {type(obj).__name__}]"


def _load_json_file_smart(input_path: str) -> pd.DataFrame:
    """
    Smartly load JSON files, support multiple formats:
    1. Nested structure (such as fmp_historical_data output):{"data": [...]}
    2. Array format:[{...}, {...}]
    3. Single object:{...}

    Args:
        input_path: JSON File path

    Returns:
        DataFrame: Loaded data
    """
    try:
        # First read as regular JSON, check structure
        _print(f"  Read JSON file: {input_path}")
        file_size = os.path.getsize(input_path)
        _print(f"  File size: {file_size / 1024:.2f} KB")

        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        _print(f"  ✓ JSON Parse successfully, data type: {type(raw_data).__name__}")

        # Case 1: nested structure (fmp_historical_data output format)
        if isinstance(raw_data, dict):
            _print(f"  Detected dictionary structure, key: {list(raw_data.keys())[:10]}")
            # Check for 'data' field
            if 'data' in raw_data:
                data_list = raw_data['data']
                if isinstance(data_list, list):
                    if len(data_list) > 0:
                        # Extract 'data' field and convert to DataFrame
                        df = pd.DataFrame(data_list)
                        # Immediately convert date columns to strings to prevent pandas from automatically recognizing as datetime type
                        df = _convert_timestamps_to_strings(df)
                        _print(f"✓ Detected nested JSON structure, 'data' field extracted: {len(df)} rows × {len(df.columns)} columns")
                        return df
                    else:
                        raise ValueError(f"JSON The 'data' field in the structure is an empty list, cannot convert to DataFrame")
                else:
                    raise ValueError(f"JSON The 'data' field in the structure is not a list type, actual type: {type(data_list).__name__}")
            # If it's a single object, try direct conversion to DataFrame
            elif len(raw_data) > 0:
                df = pd.DataFrame([raw_data])
                # Immediately convert date columns to strings
                df = _convert_timestamps_to_strings(df)
                _print(f"✓ Detected single object JSON, converted to DataFrame: {len(df)} rows × {len(df.columns)} columns")
                return df
            else:
                raise ValueError(f"JSON Dictionary is empty, cannot convert to DataFrame")

        # Case 2: Record array format
        elif isinstance(raw_data, list):
            if len(raw_data) > 0:
                # Check element types in the list
                first_elem = raw_data[0]
                if not isinstance(first_elem, dict):
                    raise ValueError(f"JSON Array elements are not dictionary type. Actual type: {type(first_elem).__name__}. Expected format: [{{...}}, {{...}}]")

                df = pd.DataFrame(raw_data)
                # Immediately convert date columns to strings to prevent pandas from automatically recognizing as datetime type
                df = _convert_timestamps_to_strings(df)
                _print(f"✓ Detected record array JSON format: {len(df)} rows × {len(df.columns)} columns")
                return df
            else:
                raise ValueError(f"JSON Array is empty, cannot convert to DataFrame")

        # If none match, raise an error
        raise ValueError(
            f"Unrecognized JSON structure: expected dict or list, actual is {type(raw_data).__name__}\n"
            f"Supported formats:\n"
            f"  1. Nested structure: {{\"data\": [{{...}}, {{...}}]}}\n"
            f"  2. Record array: [{{...}}, {{...}}]\n"
            f"  3. Single object: {{...}}"
        )

    except json.JSONDecodeError as e:
        error_msg = f"JSON Parsing failed: {str(e)}"
        # Provide more detailed error information
        if hasattr(e, 'pos'):
            error_msg += f"\n  Error position: Line {e.lineno}, Column {e.colno}"
        error_msg += f"\n  Tip: Please check if the JSON file format is correct (parentheses matching, commas, quotes, etc))"
        raise ValueError(error_msg)
    except UnicodeDecodeError as e:
        raise ValueError(
            f"File encoding error: {str(e)}\n"
            f"  Tip: JSON files should be UTF-8 encoded. Please check the file encoding format"
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        raise ValueError(
            f"Failed to read JSON file ({error_type}): {error_msg}\n"
            f"  File path: {input_path}\n"
            f"  Tip: Please check if the file exists, is readable, and has the correct format"
        )


def _load_data(input_path: str) -> pd.DataFrame:
    """
    Load data file (supports CSV, JSON, Parquet formats)
    Automatically detect file format, correctly load even when extension does not match content
    Support nested JSON structures (such as fmp_historical_data output format)

    Args:
        input_path: Input file path

    Returns:
        DataFrame: Loaded data
    """
    input_path = os.path.expanduser(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File does not exist: {input_path}")

    file_ext = os.path.splitext(input_path)[1].lower()

    # Define all supported formats and their reading methods
    # For JSON, use smart loading function to handle nested structures
    format_readers = {
        'csv': lambda: pd.read_csv(input_path),
        'json': lambda: _load_json_file_smart(input_path),
        'parquet': lambda: pd.read_parquet(input_path),
    }

    # Determine the preferred format based on file extension
    ext_to_format = {
        '.csv': 'csv',
        '.json': 'json',
        '.parquet': 'parquet',
    }

    primary_format = ext_to_format.get(file_ext, None)
    attempted_formats = []
    format_errors = {}  # Save detailed error information for each format
    skipped_formats = []  # Log skipped formats and reasons

    # First try to read by the format corresponding to the extension
    if primary_format:
        try:
            _print(f"Try to read the file in {primary_format.upper()} format...")
            df = format_readers[primary_format]()
            _print(f"✓ Successfully loaded data: {len(df)} rows × {len(df.columns)} columns")
            return df
        except Exception as e:
            attempted_formats.append(primary_format)
            error_msg = str(e)
            format_errors[primary_format] = error_msg
            _print(f"⚠️  Warning: Reading the file in {primary_format.upper()} format failed: {error_msg[:200]}")
            _print(f"   Try to automatically detect the file format...")

    # If reading by extension fails, try other supported formats
    # Important: For .json files, never attempt Parquet (to avoid false positives))
    for format_name, reader_func in format_readers.items():
        if format_name in attempted_formats:
            continue  # Already tried, skip

        # Critical fix: If the file extension is .json, never attempt Parquet
        if file_ext == '.json' and format_name == 'parquet':
            skipped_formats.append(('parquet', f'File extension is .json, skip Parquet format to avoid false positives'))
            _print(f"  Skip Parquet format (file extension is .json, to avoid false positives)")
            continue

        # Similarly, if the file extension is .parquet, do not attempt JSON (to avoid confusion))
        if file_ext == '.parquet' and format_name == 'json':
            skipped_formats.append(('json', f'File extension is .parquet, skip JSON format to avoid confusion'))
            _print(f"  Skip JSON format (file extension is .parquet)")
            continue

        try:
            _print(f"Try to read the file in {format_name.upper()} format...")
            df = reader_func()
            _print(f"✓ Successfully loaded data in {format_name.upper()} format: {len(df)} rows × {len(df.columns)} columns")
            if primary_format and format_name != primary_format:
                _print(f"⚠️  Note: File extension is {file_ext}, but actual content is {format_name.upper()} format")
            return df
        except Exception as e:
            attempted_formats.append(format_name)
            error_msg = str(e)
            format_errors[format_name] = error_msg
            _print(f"  ✗ {format_name.upper()} Format reading failed: {error_msg[:200]}")
            continue

    # All formats failed, build detailed error information
    error_parts = []
    error_parts.append(f"Data loading failed: unable to read the file")
    error_parts.append(f"File path: {input_path}")
    error_parts.append(f"File extension: {file_ext}")

    # List of actual formats tried
    if attempted_formats:
        error_parts.append(f"Formats already tried: {', '.join([f.upper() for f in attempted_formats])}")
        error_parts.append(f"Detailed error information per format:")
        for fmt in attempted_formats:
            if fmt in format_errors:
                err_msg = format_errors[fmt]
                # Truncate long error messages (keep first 300 characters)
                if len(err_msg) > 300:
                    err_msg = err_msg[:300] + "..."
                error_parts.append(f"  - {fmt.upper()}: {err_msg}")

    # List skipped formats and reasons
    if skipped_formats:
        error_parts.append(f"Skipped formats:")
        for fmt, reason in skipped_formats:
            error_parts.append(f"  - {fmt.upper()}: {reason}")

    # Provide diagnostic suggestions
    error_parts.append(f"\nDiagnostic suggestions:")
    if file_ext == '.json':
        error_parts.append(f"  1. Check if JSON file format is correct (should be valid JSON array or object)")
        error_parts.append(f"  2. Check if file is corrupted or has encoding issues (should be UTF-8 encoded)")
        error_parts.append(f"  3. Try manually opening the file to verify JSON format is correct")
        error_parts.append(f"  4. If the file is actually in another format, change the file extension")
    elif file_ext == '.csv':
        error_parts.append(f"  1. Check if CSV file format is correct (delimiter, encoding, etc)")
        error_parts.append(f"  2. Try opening the file with Excel or other tools to verify")
    elif file_ext == '.parquet':
        error_parts.append(f"  1. Check if Parquet file is complete and not corrupted")
        error_parts.append(f"  2. Try using pandas.read_parquet() to directly read and verify")
    else:
        error_parts.append(f"  1. Check if file format is supported (currently supports: CSV, JSON, Parquet)")
        error_parts.append(f"  2. If the file is in another format, convert it to a supported format")

    raise RuntimeError("\n".join(error_parts))


def _validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, str]:
    """
    Validate and map column names (supports different naming conventions)

    Args:
        df: Input DataFrame
        required_cols: Required columns (such as 'open', 'high', 'low', 'close', 'volume')

    Returns:
        Column name mapping dictionary {'open': 'actual column name', ...}
    """
    col_mapping = {}

    # Common column name variants
    name_variants = {
        # Description:
        # - These variants cover common case variations and Chinese/English mix, as well as aliases used by some data sources
        # - For FMP historical output like "Open_AAPL / Close_AAPL / Volume_AAPL" column names,
        #   will be automatically identified below using "prefix matching + exclude Benchmark" logic, no need to hardcode specific values symbol.
        'open': ['open', 'Open', '\u5f00\u76d8\u4ef7', '\u5f00\u76d8', 'OPEN', 'opening_price', 'open_price', 'opening'],
        'high': ['high', 'High', '\u6700\u9ad8\u4ef7', '\u6700\u9ad8', 'HIGH', 'highest_price', 'high_price'],
        'low': ['low', 'Low', '\u6700\u4f4e\u4ef7', '\u6700\u4f4e', 'LOW', 'lowest_price', 'low_price'],
        'close': ['close', 'Close', '\u6536\u76d8\u4ef7', '\u6536\u76d8', 'CLOSE', 'closing_price', 'close_price', 'last', 'last_price'],
        'volume': ['volume', 'Volume', '\u6210\u4ea4\u91cf', 'VOLUME', 'vol', 'Vol', 'VOL', 'turnover', '\u6210\u4ea4\u989d'],
    }

    # Low-cost column name standardization: remove leading and trailing whitespaces for easier matching (keep original column names for return))
    df_cols_lower = {str(col).strip().lower(): str(col) for col in df.columns}

    for req_col in required_cols:
        found = False
        for variant in name_variants.get(req_col, [req_col]):
            v = str(variant).strip()
            v_lower = v.lower()

            # 1) exact match (complete consistency)
            if v in df.columns:
                col_mapping[req_col] = v
                found = True
                break
            # 2) case-insensitive match
            elif v_lower in df_cols_lower:
                col_mapping[req_col] = df_cols_lower[v_lower]
                found = True
                break

            # 3) key enhancement: prefix match (solves column names like Open_AAPL / Close_AAPL / Volume_AAPL)
            #    - match pattern:"{variant}_{xxx}" / "{variant} {xxx}" / "{variant}-{xxx}"
            #    - if *_Benchmark exists, prefer non-benchmark columns
            try:
                candidates = []
                for col in df.columns:
                    col_s = str(col).strip()
                    col_l = col_s.lower()
                    if col_l.startswith(v_lower + "_") or col_l.startswith(v_lower + " ") or col_l.startswith(v_lower + "-"):
                        candidates.append(col_s)
                if candidates:
                    # prioritize exclusion benchmark
                    non_benchmark = [c for c in candidates if "benchmark" not in c.lower()]
                    chosen_pool = non_benchmark if non_benchmark else candidates
                    # choose shorter/simpler column names (usually the target symbol column)
                    chosen = sorted(chosen_pool, key=lambda x: (len(x), x.lower()))[0]
                    col_mapping[req_col] = chosen
                    found = True
                    break
            except Exception:
                # maintain compatibility: failure in matching does not affect subsequent logic
                pass

        if not found:
            # for some indicators, certain columns may be optional
            if req_col in ['volume']:
                continue
            raise ValueError(
                f"missing required columns: {req_col}\n"
                f"available columns: {list(df.columns)}\n"
                f"tip: supported column name variants: {name_variants.get(req_col, [req_col])}"
            )

    return col_mapping


def _get_indicator_params(indicator_name: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    retrieve input parameters required by the indicator

    Args:
        indicator_name: indicator name (e.g. 'SMA', 'MACD')

    Returns:
        tuple: (required_inputs, default_params)
            - required_inputs: list of required input column names (e.g. ['close'], ['high', 'low', 'close'])
            - default_params: default parameter dictionary
    """
    if talib is None:
        return ([], {})

    # retrieve TA-Lib function
    indicator_func = getattr(talib, indicator_name, None)
    if indicator_func is None:
        return ([], {})

    # retrieve function signature
    try:
        sig = inspect.signature(indicator_func)
        params = sig.parameters

        required_inputs = []
        default_params = {}

        # common price input parameters
        price_params = ['open', 'high', 'low', 'close', 'volume']

        for param_name, param in params.items():
            # if price data is provided as input
            if param_name.lower() in price_params:
                required_inputs.append(param_name.lower())
            # if default values exist, record them
            elif param.default != inspect.Parameter.empty:
                default_params[param_name] = param.default

        return (required_inputs, default_params)
    except Exception:
        # If signature cannot be obtained, return empty
        return ([], {})


def calculate_indicator(df: pd.DataFrame, indicator_name: str,
                       col_mapping: Dict[str, str],
                       params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    General indicator calculation function supporting all TA-Lib indicators

    Args:
        df: Input data
        indicator_name: indicator name (e.g. 'SMA', 'MACD', 'RSI')
        col_mapping: Column name mapping dictionary
        params: Custom parameters (optional)

    Returns:
        Added indicator columns DataFrame
    """
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    if indicator_name not in SUPPORTED_INDICATORS:
        _print(f"⚠️  Warning: {indicator_name} is not in the supported indicators list, skipping")
        return df

    # retrieve TA-Lib function
    indicator_func = getattr(talib, indicator_name, None)
    if indicator_func is None:
        _print(f"⚠️  Warning: {indicator_name} not found in TA-Lib, skipping")
        return df

    try:
        # Preparing input data
        inputs = []
        kwargs = {}

        # Detect required inputs (via function signature)
        sig = inspect.signature(indicator_func)
        param_names = list(sig.parameters.keys())

        # Build input data
        for param_name in param_names:
            param_lower = param_name.lower()
            param_obj = sig.parameters[param_name]

            # If price data (position parameters)
            if param_lower in ['open', 'high', 'low', 'close', 'volume', 'real', 'real0', 'real1']:
                # For 'real' type inputs, default use close
                if param_lower in ['real', 'real0']:
                    price_col = 'close'
                elif param_lower == 'real1':
                    price_col = 'close'  # or open, adjusted as needed
                else:
                    price_col = param_lower

                if price_col in col_mapping:
                    inputs.append(df[col_mapping[price_col]].values.astype(np.float64))
                else:
                    # If required inputs are missing, skip this indicator
                    _print(f"⚠️  Warning: {indicator_name} requires {price_col} column, but it does not exist in data, skipping")
                    return df
            # If optional parameters
            elif param_obj.default != inspect.Parameter.empty:
                # If user provides parameters, use user's values
                if params and param_name in params:
                    kwargs[param_name] = params[param_name]
                # Otherwise use default values (no explicit passing required)

        # Call indicator function
        if kwargs:
            result = indicator_func(*inputs, **kwargs)
        else:
            result = indicator_func(*inputs)

        # Process return results
        if isinstance(result, tuple):
            # Multiple outputs (e.g., MACD returns macd, signal, hist)
            output_names = _get_output_names(indicator_name, len(result))
            for i, output in enumerate(result):
                col_name = output_names[i] if i < len(output_names) else f"{indicator_name}_{i}"
                df[col_name] = output
        else:
            # Single output
            df[indicator_name] = result

        return df

    except Exception as e:
        _print(f"⚠️  Warning: Error occurred while calculating {indicator_name}: {str(e)}, skipping")
        return df


def _get_output_names(indicator_name: str, num_outputs: int) -> List[str]:
    """
    Get the output column names of the indicator

    Args:
        indicator_name: Indicator name
        num_outputs: Output count

    Returns:
        List of output column names
    """
    # Name mapping for special indicators
    output_name_map = {
        'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'MACDEXT': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'MACDFIX': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'STOCH': ['STOCH_K', 'STOCH_D'],
        'STOCHF': ['STOCHF_K', 'STOCHF_D'],
        'STOCHRSI': ['STOCHRSI_K', 'STOCHRSI_D'],
        'BBANDS': ['BBANDS_Upper', 'BBANDS_Middle', 'BBANDS_Lower'],
        'AROON': ['AROON_Down', 'AROON_Up'],
        'HT_PHASOR': ['HT_PHASOR_InPhase', 'HT_PHASOR_Quadrature'],
        'HT_SINE': ['HT_SINE', 'HT_LEADSINE'],
        'MAMA': ['MAMA', 'FAMA'],
        'MINMAX': ['MIN', 'MAX'],
        'MINMAXINDEX': ['MIN_IDX', 'MAX_IDX'],
    }

    if indicator_name in output_name_map:
        return output_name_map[indicator_name]

    # Default naming
    if num_outputs == 1:
        return [indicator_name]
    else:
        return [f"{indicator_name}_{i}" for i in range(num_outputs)]


def calculate_ma(df: pd.DataFrame, col_mapping: Dict[str, str],
                 periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """Calculate moving average (MA)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)

    for period in periods:
        df[f'MA_{period}'] = talib.SMA(close, timeperiod=period)

    _print(f"✓ Calculate MA: {periods}")
    return df


def calculate_ema(df: pd.DataFrame, col_mapping: Dict[str, str],
                  periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """Calculate exponential moving average (EMA)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)

    for period in periods:
        df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)

    _print(f"✓ Calculate EMA: {periods}")
    return df


def calculate_macd(df: pd.DataFrame, col_mapping: Dict[str, str],
                   fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """Calculate MACD indicator"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)

    macd, signal, hist = talib.MACD(
        close,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )

    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    _print(f"✓ Calculate MACD ({fast_period}, {slow_period}, {signal_period})")
    return df


def calculate_rsi(df: pd.DataFrame, col_mapping: Dict[str, str],
                  periods: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI) (RSI)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)

    for period in periods:
        df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

    _print(f"✓ Calculate RSI: {periods}")
    return df


def calculate_bollinger_bands(df: pd.DataFrame, col_mapping: Dict[str, str],
                               period: int = 20, nbdevup: float = 2.0,
                               nbdevdn: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands (BOLL)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)

    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn
    )

    df['BOLL_Upper'] = upper
    df['BOLL_Middle'] = middle
    df['BOLL_Lower'] = lower
    df['BOLL_Width'] = (upper - lower) / middle  # Bollinger Bands width

    _print(f"✓ Calculate Bollinger Bands (period={period}, std={nbdevup})")
    return df


def calculate_kdj(df: pd.DataFrame, col_mapping: Dict[str, str],
                  fastk_period: int = 9, slowk_period: int = 3,
                  slowd_period: int = 3) -> pd.DataFrame:
    """Calculate KDJ indicator"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)

    # Calculate K and D D
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0
    )

    # Calculate J = 3K - 2D
    j = 3 * slowk - 2 * slowd

    df['KDJ_K'] = slowk
    df['KDJ_D'] = slowd
    df['KDJ_J'] = j

    _print(f"✓ Calculate KDJ ({fastk_period}, {slowk_period}, {slowd_period})")
    return df


def calculate_atr(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range (ATR) (ATR)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)

    df['ATR'] = talib.ATR(high, low, close, timeperiod=period)

    _print(f"✓ Calculate ATR (period={period})")
    return df


def calculate_obv(df: pd.DataFrame, col_mapping: Dict[str, str]) -> pd.DataFrame:
    """Calculate Energy Index (OBV)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    close = df[col_mapping['close']].values.astype(np.float64)
    volume = df[col_mapping['volume']].values.astype(np.float64)

    df['OBV'] = talib.OBV(close, volume)

    _print(f"✓ Calculate OBV")
    return df


def calculate_adx(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) (ADX)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)

    df['ADX'] = talib.ADX(high, low, close, timeperiod=period)

    _print(f"✓ Calculate ADX (period={period})")
    return df


def calculate_cci(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    """Calculate Momentum indicator (CCI)"""
    if talib is None:
        raise ImportError("TA-Lib Not installed")

    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)

    df['CCI'] = talib.CCI(high, low, close, timeperiod=period)

    _print(f"✓ Calculate CCI (period={period})")
    return df


def calculate_all_indicators(df: pd.DataFrame, col_mapping: Dict[str, str],
                             config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Calculate all common technical indicators

    Args:
        df: Input data
        col_mapping: Column name mapping
        config: Custom configuration (optional)

    Returns:
        Added technical indicators DataFrame
    """
    if config is None:
        config = {}

    # MA
    if config.get('ma', True):
        df = calculate_ma(df, col_mapping, periods=config.get('ma_periods', [5, 10, 20, 60]))

    # EMA
    if config.get('ema', True):
        df = calculate_ema(df, col_mapping, periods=config.get('ema_periods', [5, 10, 20, 60]))

    # MACD
    if config.get('macd', True):
        df = calculate_macd(df, col_mapping)

    # RSI
    if config.get('rsi', True):
        df = calculate_rsi(df, col_mapping, periods=config.get('rsi_periods', [6, 12, 24]))

    # Bollinger Bands
    if config.get('boll', True):
        df = calculate_bollinger_bands(df, col_mapping)

    # KDJ
    if config.get('kdj', True):
        df = calculate_kdj(df, col_mapping)

    # ATR
    if config.get('atr', True):
        df = calculate_atr(df, col_mapping)

    # OBV
    if config.get('obv', True) and 'volume' in col_mapping:
        df = calculate_obv(df, col_mapping)

    # ADX
    if config.get('adx', True):
        df = calculate_adx(df, col_mapping)

    # CCI
    if config.get('cci', True):
        df = calculate_cci(df, col_mapping)

    return df


def run(args) -> None:
    """
    MCP Tool entry function

    Args:
        args: An object containing the following parameters
            - input_path: Path to the input data file
            - output_path: Output file path (optional)
            - indicators: List of indicators to calculate (optional, default all 150 indicators)
            - config: Custom configuration (optional)
            - validate: Whether to return validation data (optional, defaultFalse)
    """
    if talib is None:
        raise ImportError(
            "TA-Lib Not installed, please install first:\n"
            "  Ubuntu/Debian: sudo apt-get install ta-lib && pip install TA-Lib\n"
            "  Mac: brew install ta-lib && pip install TA-Lib\n"
            "  Windows: Download whl file to install: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
        )

    # Get parameters
    input_path = getattr(args, 'input_path', None)
    if not input_path:
        raise ValueError("Missing required parameters: input_path")

    output_path = getattr(args, 'output_path', None)
    indicators = getattr(args, 'indicators', None)
    config_dict = getattr(args, 'config', {})
    validate = getattr(args, 'validate', False)

    # If the user does not specify indicators, use all supported indicators
    if indicators is None or (isinstance(indicators, list) and 'all' in indicators):
        indicators = SUPPORTED_INDICATORS.copy()
        _print(f"Calculating all supported indicators (total {len(indicators)} indicators)")
    elif isinstance(indicators, str):
        indicators = [indicators]

    _print("="*60)
    _print(f"Technical indicator calculator - based on TA-Lib")
    _print("="*60)
    _print(f"Total number of supported indicators: {len(SUPPORTED_INDICATORS)}")
    _print(f"Number of indicators calculated this time: {len(indicators)}")

    # Loading data
    _print(f"\nStep 1: Load data file")
    _print(f"  File path: {input_path}")
    df_original = _load_data(input_path)
    df = df_original.copy()
    _print(f"  Number of data columns: {list(df.columns)}")

    # Immediately convert all date columns to strings to avoid issues from automatic conversion later
    _print(f"  Preprocessing: convert date columns to string format...")
    df = _convert_timestamps_to_strings(df)
    df_original = df.copy()  # Update original data copy

    # Validate column names (attempt to map all potentially needed columns)
    _print(f"\nStep 2: Validate data columns")
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    col_mapping = {}
    for col in required_cols:
        try:
            mapping = _validate_columns(df, [col])
            col_mapping.update(mapping)
        except ValueError:
            # Some columns may not exist, skipping
            pass

    _print(f"  Column name mapping: {col_mapping}")
    _print(f"  Available price columns: {list(col_mapping.keys())}")

    # Calculate indicators
    _print(f"\nStep 3: Calculate technical indicators")
    _print(f"  Indicator list: {indicators if len(indicators) <= 10 else f'{indicators[:10]}... (total {len(indicators)} items)'}")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, indicator_name in enumerate(indicators, 1):
        try:
            original_cols = len(df.columns)
            df = calculate_indicator(df, indicator_name, col_mapping, config_dict)
            new_cols = len(df.columns)

            if new_cols > original_cols:
                success_count += 1
                if (i % 20 == 0) or (i == len(indicators)):
                    _print(f"  Progress: {i}/{len(indicators)} | Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}")
            else:
                skipped_count += 1
        except Exception as e:
            failed_count += 1
            _print(f"  ❌ {indicator_name} Calculation failed: {str(e)}")

    _print(f"\nIndicator calculation statistics:")
    _print(f"  ✅ Success: {success_count}")
    _print(f"  ⏭️  Skipped: {skipped_count}")
    _print(f"  ❌ Failed: {failed_count}")

    # Before saving, ensure all Timestamps have been converted (new Timestamps might be reintroduced after calculating indicators))
    _print(f"\nStep 3.5: Final data cleaning (ensure no Timestamp objects))")
    df = _convert_timestamps_to_strings(df)

    # Additional validation: use a unified type detection function to thoroughly check for any remaining datetime objects
    timestamp_count = 0
    problematic_cols = []

    # Check index
    if len(df.index) > 0:
        sample_idx = df.index[0]
        if _is_datetime_like(sample_idx):
            timestamp_count += 1
            problematic_cols.append("Index")

    # Check all columns
    for col in df.columns:
        col_dtype = df[col].dtype

        # Check explicitly datetime types
        if pd.api.types.is_datetime64_any_dtype(col_dtype) or pd.api.types.is_period_dtype(col_dtype):
            timestamp_count += 1
            problematic_cols.append(col)
        elif col_dtype == 'object':
            # Use a unified type detection function to thoroughly check object columns
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Check more samples (up to 20))
                sample_size = min(20, len(non_null_values))
                for val in non_null_values.head(sample_size):
                    if _is_datetime_like(val):
                        timestamp_count += 1
                        problematic_cols.append(col)
                        break
                    # Check nested structures
                    elif isinstance(val, (list, tuple)):
                        for item in val[:5]:
                            if _is_datetime_like(item):
                                timestamp_count += 1
                                problematic_cols.append(col)
                                break
                        if col in problematic_cols:
                            break
                    elif isinstance(val, dict):
                        for item in list(val.values())[:5]:
                            if _is_datetime_like(item):
                                timestamp_count += 1
                                problematic_cols.append(col)
                                break
                        if col in problematic_cols:
                            break

    if timestamp_count > 0:
        _print(f"  ⚠️  Warning: {timestamp_count} columns/indices potentially still contain datetime types detected: {problematic_cols[:10]}")
        _print(f"  Perform secondary cleaning...")
        df = _convert_timestamps_to_strings(df)  # Clean again

        # Secondary validation: if issues remain, perform tertiary cleaning
        timestamp_count_2 = 0
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                timestamp_count_2 += 1
            elif df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                sample = df[col].dropna().head(10)
                for val in sample:
                    if _is_datetime_like(val):
                        timestamp_count_2 += 1
                        break

        if timestamp_count_2 > 0:
            _print(f"  ⚠️  Warning: {timestamp_count_2} issues remain after secondary cleaning, performing tertiary cleaning...")
            df = _convert_timestamps_to_strings(df)  # Tertiary cleaning

    _print(f"  ✓ Data cleaning completed, ensuring all datetime objects have been converted to strings")

    # save results
    _print(f"\nStep 4: Save results")
    if output_path:
        output_path = os.path.expanduser(output_path)
    else:
        # Generate output path automatically if not specified
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path) or '.'
        output_path = os.path.join(output_dir, f"{input_basename}_indicators.csv")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_ext = os.path.splitext(output_path)[1].lower()

    # For JSON format, convert Timestamp objects to strings first
    if file_ext == '.json':
        # Ensure all Timestamps are converted again (double check))
        df_json = _convert_timestamps_to_strings(df.copy())

        # Pre-check validation: test serialization of a small portion of data (first 3 rows))
        _print(f"  Pre-check: test serialization of the first 3 rows of data...")
        try:
            test_df = df_json.head(3).copy()
            # Attempt to serialize test data
            test_records = test_df.to_dict('records')
            test_records = _convert_dict_timestamps_to_strings(test_records)
            # Test serialization using a custom encoder
            test_json_str = json.dumps(test_records, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
            # Verify it can be parsed
            json.loads(test_json_str)
            _print(f"  ✓ Pre-check passed: the first 3 rows of data can be successfully serialized")
        except Exception as precheck_err:
            _print(f"  ⚠️  Pre-check warning: {str(precheck_err)[:150]}")
            # Pre-check failed, perform additional cleanup
            df_json = _convert_timestamps_to_strings(df_json)

        # Use a safer method to save JSON with multi-layer protection
        try:
            # Method 1: try using to_json directly (if conversion succeeds, this should work))
            json_str = df_json.to_json(orient='records', indent=2, force_ascii=False, date_format='iso')
            # Verify if the generated JSON string is valid (test parsing))
            try:
                json.loads(json_str)
                # If valid, write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                _print(f"  ✓ Successfully saved using the to_json method")
            except json.JSONDecodeError as parse_err:
                # JSON String is invalid, try method2
                raise ValueError(f"Generated JSON string is invalid: {parse_err}")
        except (TypeError, ValueError, AttributeError) as e:
            # Method 2: if to_json fails, use a custom encoder
            _print(f"  ⚠️  to_json Failed, use custom encoder: {str(e)[:150]}")
            try:
                # Convert to dictionary, then use custom encoder
                records = df_json.to_dict('records')
                # Recursively clean all Timestamps (double check))
                records = _convert_dict_timestamps_to_strings(records)
                # Save using custom encoder
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
                _print(f"  ✓ Successfully saved using custom encoder")
            except Exception as e2:
                # Method 3: last resort - convert row by row
                _print(f"  ⚠️  Custom encoder also fails, use row-by-row conversion: {str(e2)[:150]}")
                try:
                    # Process line by line, ensuring each line is fully cleaned
                    clean_records = []
                    for idx, row in df_json.iterrows():
                        row_dict = row.to_dict()
                        # Recursively clean
                        clean_row = _convert_dict_timestamps_to_strings(row_dict)
                        clean_records.append(clean_row)
                    # Save using custom encoder
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_records, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
                    _print(f"  ✓ Successfully saved using line-by-line transformation")
                except Exception as e3:
                    # All methods failed
                    error_msg = (
                        f"Failed to save JSON file; all serialization methods failed\n"
                        f"  File path: {output_path}\n"
                        f"  Number of data rows: {len(df_json)}\n"
                        f"  Number of data columns: {len(df_json.columns)}\n"
                        f"  Column names: {list(df_json.columns)}\n"
                        f"  Error1 (to_json): {str(e)[:200]}\n"
                        f"  Error 2 (custom encoder): {str(e2)[:200]}\n"
                        f"  Error 3 (line-by-line transformation): {str(e3)[:200]}\n"
                        f"  Tip: Please check if your data contains objects that cannot be serialized"
                    )
                    _print(f"  ✗ {error_msg}")
                    raise RuntimeError(error_msg)
    elif file_ext == '.csv' or not file_ext:
        df.to_csv(output_path, index=False)
    elif file_ext == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    file_size = os.path.getsize(output_path) / 1024  # KB
    _print(f"  ✓ Results have been saved: {output_path}")
    _print(f"  File size: {file_size:.1f} KB")

    # Generate validation data
    if validate:
        _print(f"\nStep 5: Generate validation data")
        # Use the already-transformed DataFrame (to avoid re-creating with to_dict) Timestamp)
        # First transform, then take the first 3 rows, ensuring all dates are strings
        df_for_preview = _convert_timestamps_to_strings(df.head(3).copy())

        # Use a safer approach: first convert to JSON string, then parse into a dictionary
        # This avoids re-creating Timestamp objects with to_dict
        preview_dict = None
        preview_error = None

        # Method 1: Try using to_json (the safest)
        try:
            json_str = df_for_preview.to_json(orient='records', date_format='iso')
            preview_dict = json.loads(json_str)  # Validate and parse
            _print(f"  ✓ Generate preview data using the to_json method")
        except Exception as e1:
            preview_error = str(e1)
            _print(f"  ⚠️  to_json Method failed: {str(e1)[:150]}")
            # Method 2: Use to_dict + recursive cleaning
            try:
                preview_dict = df_for_preview.to_dict('records')
                # Recursively clean all Timestamp
                preview_dict = _convert_dict_timestamps_to_strings(preview_dict)
                _print(f"  ✓ Generate preview data using to_dict + recursive cleaning")
            except Exception as e2:
                preview_error = f"{str(e1)}; {str(e2)}"
                _print(f"  ⚠️  to_dict Method also failed: {str(e2)[:150]}")
                # Method 3: Process line by line
                try:
                    preview_dict = []
                    for idx, row in df_for_preview.iterrows():
                        row_dict = row.to_dict()
                        clean_row = _convert_dict_timestamps_to_strings(row_dict)
                        preview_dict.append(clean_row)
                    _print(f"  ✓ Generate preview data by processing line by line")
                except Exception as e3:
                    preview_error = f"{str(e1)}; {str(e2)}; {str(e3)}"
                    _print(f"  ✗ All preview data generation methods failed")
                    # If all methods fail, use an empty list
                    preview_dict = []

        validation_data = {
            'input_file': input_path,
            'output_file': output_path,
            'total_rows': len(df),
            'original_columns': len(df_original.columns),
            'total_columns': len(df.columns),
            'new_indicators': len(df.columns) - len(df_original.columns),
            'requested_indicators': len(indicators),
            'success_count': success_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'new_column_names': [col for col in df.columns if col not in df_original.columns],
            'data_preview': preview_dict
        }

        # Save validation data (ensure all Timestamps are converted and use custom encoder))
        validation_data = _convert_dict_timestamps_to_strings(validation_data)
        validation_file = output_path.replace(file_ext, '_validation.json')

        try:
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
            _print(f"  ✓ Validation data saved: {validation_file}")
        except Exception as e:
            # If saving fails, try without custom encoder (theoretically should not happen, as already cleaned))
            _print(f"  ⚠️  Failed to save validation data with custom encoder; trying standard method: {str(e)[:150]}")
            try:
                # Clean again
                validation_data = _convert_dict_timestamps_to_strings(validation_data)
                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(validation_data, f, indent=2, ensure_ascii=False)
                _print(f"  ✓ Validation data saved (using standard method)): {validation_file}")
            except Exception as e2:
                _print(f"  ✗ Validation data save failed: {str(e2)[:200]}")
                # Do not raise an exception because the main output file may have been successfully saved
                _print(f"  Warning: Validation data save failed, but main output file may have been successfully saved")

        # Print validation summary
        _print(f"\nData validation:")
        _print(f"  Original column count: {validation_data['original_columns']}")
        _print(f"  New metrics: {validation_data['new_indicators']}")
        _print(f"  Total column count: {validation_data['total_columns']}")
        _print(f"  Number of data rows: {validation_data['total_rows']}")

    _print(f"\n" + "="*60)
    _print(f"✅ Technical indicators calculation completed!")
    _print(f"  Input file: {input_path}")
    _print(f"  Output file: {output_path}")
    _print(f"  Number of data rows: {len(df)}")
    _print(f"  Original column count: {len(df_original.columns)}")
    _print(f"  New column count: {len(df.columns) - len(df_original.columns)}")
    _print(f"  Total column count: {len(df.columns)}")
    _print("="*60)
