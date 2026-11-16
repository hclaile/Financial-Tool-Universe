import os
import sys
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import numpy as np

# Default output path
DEFAULT_OUTPUT_DIR = os.path.join(os.environ.get("FTU_OUTPUT_DIR", "output"), "Data")


def _print(*args, **kwargs):
	"""Print to stderr to avoid interfering with MCP JSON-RPC communication"""
	print(*args, file=sys.stderr, **kwargs)


def _convert_txt_to_csv(txt_path: str) -> str:
	"""
	Convert TXT file to CSV file

	Args:
		txt_path: TXT File path

	Returns:
		Path of the converted CSV file
	"""
	_print(f"Detected TXT file, converting to CSV format...")

	# Generate CSV file path (same directory as TXT file))
	csv_path = txt_path.replace('.txt', '_converted.csv')

	try:
		# Attempting to read TXT file (assuming tab or comma-separated))
		# First attempt to detect delimiter
		with open(txt_path, 'r', encoding='utf-8') as f:
			first_line = f.readline()

		# Determine delimiter
		if '\t' in first_line:
			separator = '\t'
			_print(f"  Delimiter detected: Tab")
		elif ',' in first_line:
			separator = ','
			_print(f"  Delimiter detected: comma")
		elif ' ' in first_line:
			separator = r'\s+'  # One or more spaces
			_print(f"  Delimiter detected: space")
		else:
			separator = ','  # Default delimiter
			_print(f"  Using default delimiter: comma")

		# Reading TXT file
		df = pd.read_csv(txt_path, sep=separator, engine='python')

		# Save as CSV
		df.to_csv(csv_path, index=False)
		_print(f"✓ TXT File converted to CSV: {csv_path}")
		_print(f"  Number of data rows: {len(df)}")
		_print(f"  Number of data columns: {list(df.columns)}")

		return csv_path

	except Exception as e:
		raise RuntimeError(f"TXT CSV conversion failed: {str(e)}")


def _load_input_file(input_path: str) -> tuple[pd.DataFrame, str]:
	"""
	Load input file, supports CSV, JSON, TXT formats

	Args:
		input_path: Input file path

	Returns:
		tuple: (DataFrame, Actual loaded file path)
	"""
	input_path = os.path.abspath(os.path.expanduser(input_path))

	if not os.path.exists(input_path):
		raise FileNotFoundError(f"File does not exist: {input_path}")

	file_ext = os.path.splitext(input_path)[1].lower()
	actual_file_path = input_path

	_print(f"Loading file: {input_path}")

	try:
		# If it's a TXT file, convert it to CSV
		if file_ext == ".txt":
			actual_file_path = _convert_txt_to_csv(input_path)
			df = pd.read_csv(actual_file_path)
		elif file_ext == ".csv":
			# CSV file
			df = pd.read_csv(input_path)
		elif file_ext == ".json":
			# JSON file, trying multiple reading methods
			try:
				df = pd.read_json(input_path, orient='records')
			except ValueError:
				try:
					df = pd.read_json(input_path, lines=True)
				except:
					df = pd.read_json(input_path)
		else:
			raise ValueError(f"Unsupported file format: {file_ext}, supported formats: .csv, .json, .txt")

		_print(f"✓ Successfully loaded data: {len(df)} rows × {len(df.columns)} columns")
		return df, actual_file_path

	except Exception as e:
		raise RuntimeError(f"Data loading failed: {str(e)}")


def _normalize_dataframe(df: pd.DataFrame, datetime_col: str, instrument_col: str, value_cols: list[str], tz: str | None) -> pd.DataFrame:
	"""
	Standardize DataFrame to Qlib MultiIndex format

	Args:
		df: Input DataFrame
		datetime_col: Date column name
		instrument_col: Instrument code column name
		value_cols: List of value column names
		tz: Timezone (optional)

	Returns:
		Standardized DataFrame
	"""
	if datetime_col not in df.columns or instrument_col not in df.columns:
		raise ValueError(f"Input missing required columns: {datetime_col} or {instrument_col}\nAvailable columns: {list(df.columns)}")

	keep_cols = [c for c in value_cols if c in df.columns]
	if not keep_cols:
		raise ValueError(f"No value columns found to keep, please check column names\nExpected columns: {value_cols}\nAvailable columns: {list(df.columns)}")

	df = df.copy()
	df[datetime_col] = pd.to_datetime(df[datetime_col])

	# Timezone handling (optional)
	if tz:
		try:
			df[datetime_col] = df[datetime_col].dt.tz_localize(tz) if df[datetime_col].dt.tz is None else df[datetime_col].dt.tz_convert(tz)
		except Exception:
			df[datetime_col] = df[datetime_col].dt.tz_convert(tz)

	out = df[[datetime_col, instrument_col] + keep_cols].copy()
	out = out.rename(columns={datetime_col: "datetime", instrument_col: "instrument"})
	out = out.set_index(["datetime", "instrument"]).sort_index()

	_print(f"✓ Data standardization completed: {len(out)} rows, {len(keep_cols)} feature columns")
	return out




def run(args) -> None:
	"""
	Convert CSV/JSON/TXT to Qlib Parquet format
	Supported output formats: Parquet (recommended)
	"""
	_print("="*60)
	_print("Qlib Data import tool - convert to Parquet format")
	_print("="*60)

	# Get parameters
	input_path = getattr(args, "input", None)
	if not input_path:
		raise ValueError("Missing required parameters: input (input file path)")

	output_path = getattr(args, "output", None)
	date_field = getattr(args, "datetime_col", "date")
	symbol_field = getattr(args, "instrument_col", "symbol")
	include_fields = getattr(args, "columns", ["open", "high", "low", "close", "volume"])
	include_fields = list(include_fields) if isinstance(include_fields, (list, tuple)) else str(include_fields).split(",")
	include_fields = [c.strip() for c in include_fields if c and str(c).strip()]
	tz = getattr(args, "tz", None)

	_print(f"\nConfiguration information:")
	_print(f"  Input file: {input_path}")
	_print(f"  Output format: Parquet")
	_print(f"  Date column: {date_field}")
	_print(f"  Stock column: {symbol_field}")
	_print(f"  Number of data columns: {include_fields}")
	if tz:
		_print(f"  Time zone: {tz}")

	# Step 1: Load input file (if TXT, convert toCSV)
	_print(f"\nStep 1: Load input file")
	df, actual_file = _load_input_file(input_path)

	# Step 2: Standardize to MultiIndex
	_print(f"\nStep 2: Standardize data format")
	out_df = _normalize_dataframe(df, date_field, symbol_field, include_fields, tz)

	# Step 3: Save as Parquet format
	_print(f"\nStep 3: Save as Parquet format")

	# Generate output file path
	if not output_path:
		# Auto-generate output filename
		input_basename = os.path.splitext(os.path.basename(input_path))[0]
		output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{input_basename}_qlib.parquet")

	output_path = os.path.abspath(os.path.expanduser(output_path))

	# Ensure output directory exists
	output_dir = os.path.dirname(output_path)
	if output_dir:
		Path(output_dir).mkdir(parents=True, exist_ok=True)

	# Save as Parquet
	try:
		out_df.to_parquet(output_path, compression='snappy')
		file_size = os.path.getsize(output_path) / 1024  # KB
		_print(f"✓ Saved as Parquet format: {output_path}")
		_print(f"  File size: {file_size:.1f} KB")
		_print(f"  Compression method: snappy")
	except Exception as e:
		raise RuntimeError(f"Failed to save Parquet file: {str(e)}")

	_print(f"\n" + "="*60)
	_print("✅ Data import completed!")
	_print(f"  Input: {input_path}")
	if actual_file != input_path:
		_print(f"  Intermediate file: {actual_file}")
	_print(f"  Output: {output_path}")
	_print("="*60)
