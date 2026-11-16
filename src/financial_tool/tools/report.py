"""
Report Generation Tool Module

Generate PDF reports following the workflow:
1. Data Layer: Pandas reads CSV/JSON data
2. Visualization Layer: mplfinance generates candlestick charts and indicator charts, saves as PNG
3. Orchestration Layer: Jinja2 combines LLM text analysis + generated image paths into HTML
4. Rendering Layer: WeasyPrint renders HTML into final PDF
"""
import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Optional dependency imports
try:
    import pandas as pd
except ImportError:
    pd = None
    raise ImportError("pandas package not installed, please run: pip install pandas")

try:
    import mplfinance as mpf
except ImportError:
    mpf = None
    raise ImportError("mplfinance package not installed, please run: pip install mplfinance")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    raise ImportError("matplotlib package not installed, please run: pip install matplotlib")

try:
    from jinja2 import Template, Environment, FileSystemLoader
except ImportError:
    Template = None
    Environment = None
    FileSystemLoader = None
    raise ImportError("jinja2 package not installed, please run: pip install jinja2")

# WeasyPrint Optional: if the underlying library (such as libgobject) is not installed or missing, skip PDF only and do not block module import
HTML = None
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as _e:
    print(f"[WARN] WeasyPrint Unavailable, will skip PDF generation (only retain HTML or skip rendering). Reason: {_e}", file=sys.stderr)


# Default path configuration
DEFAULT_REPORT_DIR = os.path.join(os.environ.get("FTU_OUTPUT_DIR", "output"), "report")
DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _print(*args, **kwargs):
    """Print to stderr to avoid interfering with MCP JSON-RPC communication"""
    print(*args, file=sys.stderr, **kwargs)


def _ensure_dir(path: str) -> str:
    """Ensure directory exists"""
    path = os.path.expanduser(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _validate_data_file(file_path: str) -> tuple[bool, str]:
    """
    Validate if a data file is readable and in correct format (CSV or JSON)

    Args:
        file_path: Data file path to validate

    Returns:
        tuple: (is_valid, error_message)
        - is_valid: True if file is valid, False otherwise
        - error_message: Error message if invalid, empty string if valid
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"

    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"

    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"

    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.csv', '.json']:
        return False, f"Unsupported file format: {file_ext}, only .csv and .json are supported"

    # Check file size (skip empty files)
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, f"File is empty: {file_path}"
    except OSError as e:
        return False, f"Cannot get file size: {file_path}, error: {str(e)}"

    # Try to read and parse the file based on format
    try:
        if file_ext == '.csv':
            # Try to read first few lines to validate CSV format
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first line to check if it's valid
                first_line = f.readline()
                if not first_line.strip():
                    return False, f"CSV file appears to be empty or invalid: {file_path}"

            # Try to parse with pandas to validate format
            try:
                test_df = pd.read_csv(file_path, nrows=1)
                if test_df.empty:
                    return False, f"CSV file has no data rows: {file_path}"
            except pd.errors.EmptyDataError:
                return False, f"CSV file is empty: {file_path}"
            except pd.errors.ParserError as e:
                return False, f"CSV file has parsing errors: {file_path}, error: {str(e)}"
            except Exception as e:
                return False, f"CSV file validation failed: {file_path}, error: {str(e)}"

        elif file_ext == '.json':
            # Try to parse JSON to validate format
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check if JSON data is valid (not None, and is dict or list)
                if data is None:
                    return False, f"JSON file contains null data: {file_path}"

                if not isinstance(data, (dict, list)):
                    return False, f"JSON file does not contain valid structure (dict or list): {file_path}, got {type(data)}"

                # If it's a dict, check if it's empty
                if isinstance(data, dict) and len(data) == 0:
                    return False, f"JSON file contains empty dictionary: {file_path}"

                # If it's a list, check if it's empty
                if isinstance(data, list) and len(data) == 0:
                    return False, f"JSON file contains empty list: {file_path}"

            except json.JSONDecodeError as e:
                return False, f"JSON file has parsing errors: {file_path}, error: {str(e)}"
            except UnicodeDecodeError as e:
                return False, f"JSON file encoding error: {file_path}, error: {str(e)}"
            except Exception as e:
                return False, f"JSON file validation failed: {file_path}, error: {str(e)}"

    except Exception as e:
        return False, f"File validation failed: {file_path}, error: {str(e)}"

    return True, ""


def _load_data_file(file_path: str) -> pd.DataFrame:
    """
    Data Layer: Read CSV or JSON data file

    Args:
        file_path: Data file path (supports .csv, .json)

    Returns:
        pandas DataFrame

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid or cannot be parsed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError(f"CSV file is empty or has no data: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV file parsing error: {file_path}, error: {str(e)}")
    elif file_ext == '.json':
        # Try to read JSON, may be a list of records or nested structure
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON file parsing error: {file_path}, error: {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"JSON file encoding error: {file_path}, error: {str(e)}")

        # If it's a dict, try to extract 'data' field
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'historical' in data:
                data = data['historical']
            elif 'data_preview' in data:
                # Support for validation JSON files with data_preview field
                data = data['data_preview']

        # Convert to DataFrame
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"JSON file contains empty list: {file_path}")
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError(f"Cannot parse JSON data format: {type(data)}")

        if df.empty:
            raise ValueError(f"JSON file results in empty DataFrame: {file_path}")

        # Handle string "NaN" values in JSON (JSON standard doesn't support NaN,
        # but some files may contain string "NaN" which pandas will treat as string)
        # Convert string "NaN" to actual NaN values
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                # Replace string "NaN" (case-insensitive) with None (pandas converts None to NaN)
                df[col] = df[col].replace(['NaN', 'nan', 'null', 'None'], None)
                # Try to convert numeric columns with NaN to numeric type
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails
    else:
        raise ValueError(f"Unsupported file format: {file_ext}, only .csv and .json are supported")

    _print(f"[INFO] Successfully read data file: {file_path}, shape: {df.shape}")
    return df


def _safe_load_json(content: str) -> Optional[Union[Dict, List]]:
    """
    Highly fault-tolerant JSON parsing, supports Python single-quote dictionary format
    """
    if not content or not isinstance(content, str):
        return None

    content = content.strip()
    if not (content.startswith('{') or content.startswith('[')):
        return None

    # 1. Standard JSON parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Attempt to process Python single-quote format (format produced by str(dict))
    try:
        import ast
        return ast.literal_eval(content)
    except:
        pass

    # 3. Last resort: simple quote replacement (only when it looks like a dictionary))
    try:
        # Replace single quotes with double quotes, but be cautious with abbreviations like "don't"
        # This is a very rough heuristic method
        fixed_content = content.replace("'", '"')
        return json.loads(fixed_content)
    except:
        pass

    return None


def _format_analysis_content(content: str) -> str:
    """
    Format analysis content into a clean HTML format

    Args:
        content: Original content string

    Returns:
        Formatted HTML string
    """
    import html as html_module
    import re

    if not content:
        return ''

    # If already in HTML format (contains HTML tags), return directly
    if '<' in content and '>' in content:
        # Check if complete HTML tags are present
        html_pattern = re.compile(r'<[^>]+>')
        if html_pattern.search(content):
            return content

    content = content.strip()

    # Try to parse into structured data (JSON or Python dict string)
    parsed_data = _safe_load_json(content)
    if parsed_data is not None:
        return _format_json_to_html(parsed_data)

    # Process plain text: convert to formattedHTML
    # Do not escape first, as we need to handle Markdown format
    # Process line by line, supporting Markdown format
    lines = content.split('\n')
    formatted_lines = []
    in_list = False
    list_type = None  # 'ul' or 'ol'
    current_paragraph = []

    def _flush_paragraph():
        """Refresh current paragraph content toHTML"""
        nonlocal current_paragraph, formatted_lines
        if current_paragraph:
            para_text = ' '.join(current_paragraph).strip()
            if para_text:
                # Process inline Markdown format (bold, italic, code, etc))
                para_html = _format_inline_markdown(para_text)
                formatted_lines.append(f'<p>{para_html}</p>')
            current_paragraph = []

    def _flush_list():
        """Refresh current list toHTML"""
        nonlocal in_list, list_type, formatted_lines
        if in_list:
            formatted_lines.append(f'</{list_type}>')
            in_list = False
            list_type = None

    for line in lines:
        line = line.rstrip()  # Preserve leading whitespace (for indentation detection)
        line_stripped = line.strip()

        if not line_stripped:
            # Empty line: end current paragraph and list
            _flush_paragraph()
            _flush_list()
            continue

        # Check if it's a Markdown heading (starts with #)
        if line_stripped.startswith('#'):
            _flush_paragraph()
            _flush_list()
            if line_stripped.startswith('####'):
                title_text = line_stripped[4:].strip()
                title_html = _format_inline_markdown(title_text)
                formatted_lines.append(f'<h4>{title_html}</h4>')
            elif line_stripped.startswith('###'):
                title_text = line_stripped[3:].strip()
                title_html = _format_inline_markdown(title_text)
                formatted_lines.append(f'<h3>{title_html}</h3>')
            elif line_stripped.startswith('##'):
                title_text = line_stripped[2:].strip()
                title_html = _format_inline_markdown(title_text)
                formatted_lines.append(f'<h2>{title_html}</h2>')
            elif line_stripped.startswith('#'):
                title_text = line_stripped[1:].strip()
                title_html = _format_inline_markdown(title_text)
                formatted_lines.append(f'<h1>{title_html}</h1>')
            continue

        # Check if it's a list item (starts with - , * , + or a number)
        list_match = re.match(r'^(\s*)([-*+]|\d+[.)])\s+(.+)$', line_stripped)
        if list_match:
            _flush_paragraph()
            indent, marker, item_text = list_match.groups()

            # Determine list type
            new_list_type = 'ol' if marker[0].isdigit() else 'ul'

            # Start a new list if list type changes or was not in a list before
            if not in_list or list_type != new_list_type:
                _flush_list()
                formatted_lines.append(f'<{new_list_type}>')
                in_list = True
                list_type = new_list_type

            # Process list item content
            item_html = _format_inline_markdown(item_text)
            formatted_lines.append(f'<li>{item_html}</li>')
            continue
        else:
            # Not a list item, end the list
            _flush_list()

        # Regular text line, add to current paragraph
        current_paragraph.append(line_stripped)

    # Process the final content
    _flush_paragraph()
    _flush_list()

    if not formatted_lines:
        # If no formatted content, process the entire content directly
        # Note: no longer perform html.escape here, as _format_inline_markdown internally handles escaping
        formatted_content = _format_inline_markdown(content)
        return f"<p>{formatted_content}</p>"

    return "".join(formatted_lines)


def _format_inline_markdown(text: str) -> str:
    """
    Format inline Markdown syntax (bold, italic, code, etc))
    Use absolutely safe placeholder mechanism to prevent placeholder leakage or mis-parsing.

    Args:
        text: Original content string

    Returns:
        The resulting HTML string
    """
    import html as html_module
    import re
    import time
    import random

    # Dictionary to store placeholders
    placeholders = {}

    # Use absolutely safe, non-matching identifiers with Markdown regex
    # Avoid using __ or ** or ` or [] etc, as these would be matched by Markdown regex causing leakage
    # Use pure uppercase alphanumeric combinations, without any characters that would be escaped by html.escape
    PREFIX = "MARKERPH"
    SUFFIX = "PHMARKER"

    def _create_placeholder(content_type, *args):
        """Create a unique and secure placeholder"""
        # Generate a unique ID based on microsecond timestamp and random number to ensure no conflicts across multiple calls
        uid = f"{int(time.time() * 1000000)}{random.randint(1000, 9999)}"
        key = f"{PREFIX}{uid}{SUFFIX}"
        placeholders[key] = (content_type, args)
        return key

    # --- Phase one: identify and extract markers ---
    # Must be executed in priority order, process code blocks first

    # 1. Process code blocks:`code`
    text = re.sub(r"`([^`]+)`", lambda m: _create_placeholder("code", m.group(1)), text)

    # 2. Process bold: **text** or __text__
    text = re.sub(r"\*\*([^*]+?)\*\*", lambda m: _create_placeholder("bold", m.group(1)), text)
    text = re.sub(r"__([^_]+?)__", lambda m: _create_placeholder("bold", m.group(1)), text)

    # 3. Process italic: *text* or _text_
    # Use negative lookahead to ensure no match with already processed bold markers
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", lambda m: _create_placeholder("italic", m.group(1)), text)
    text = re.sub(r"(?<!_)_([^_]+?)_(?!_)", lambda m: _create_placeholder("italic", m.group(1)), text)

    # 4. Process links:[text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: _create_placeholder("link", m.group(1), m.group(2)), text)

    # --- Phase two: HTML-escape remaining plain text ---
    text = html_module.escape(text)

    # --- Phase three: precise refill ---
    # Use loop replacement to ensure all placeholders are correctly processed
    sorted_keys = sorted(placeholders.keys(), reverse=True)

    for key in sorted_keys:
        if key in text:
            content_type, args = placeholders[key]
            if content_type == "code":
                replacement = f"<code>{html_module.escape(args[0])}</code>"
            elif content_type == "bold":
                replacement = f"<strong>{html_module.escape(args[0])}</strong>"
            elif content_type == "italic":
                replacement = f"<em>{html_module.escape(args[0])}</em>"
            elif content_type == "link":
                replacement = f'<a href="{html_module.escape(args[1])}">{html_module.escape(args[0])}</a>'
            else:
                replacement = html_module.escape(str(args[0]))

            # Use string replacement, which is safer and faster than regex
            text = text.replace(key, replacement)

    # --- Phase four: final cleanup ---
    # Clean up any formatting remnants (including old placeholder formats))
    text = re.sub(re.escape(PREFIX) + r"\d+" + re.escape(SUFFIX), "", text)
    text = re.sub(r"\[MD_PLACEHOLDER_\d+\]", "", text)
    text = re.sub(r"\u200C\u200DMD_PLACEHOLDER_[\d_\-]+\u200D\u200C", "", text)

    return text

    # Escape HTML special characters but preserve Markdown markers
    # We need to process Markdown markers first, then escape
    # Strategy: extract Markdown markers, escape the text, then restore the markers

    # Save placeholders for Markdown markers
    # Use a safer placeholder format: invisible characters and unique identifiers to avoid conflicts with normal text
    placeholders = {}
    placeholder_counter = 0

    # Use zero-width non-joiner (U+200C) and zero-width joiner (U+200D) as placeholder separators
    # These characters remain unchanged after HTML escaping and do not render visibly
    PLACEHOLDER_PREFIX = '\u200C\u200DMD_PLACEHOLDER_'
    PLACEHOLDER_SUFFIX = '\u200D\u200C'

    def _create_placeholder(content):
        nonlocal placeholder_counter
        # Generate a unique identifier using timestamp + random number + counter to ensure placeholder uniqueness
        # Use zero-width characters as separators to avoid conflicts with normal text, and remain identifiable after HTML escaping
        timestamp = int(time.time() * 1000000)  # Microsecond timestamp
        random_id = random.randint(10000, 99999)
        unique_id = f"{timestamp}_{random_id}_{placeholder_counter}"
        key = f"{PLACEHOLDER_PREFIX}{unique_id}{PLACEHOLDER_SUFFIX}"
        placeholder_counter += 1
        placeholders[key] = content
        return key

    # 1. Process code block: `code` (prioritized to avoid being matched by other rules)
    def _replace_code(match):
        code_text = match.group(1)
        placeholder = _create_placeholder(('code', code_text))
        return placeholder
    text = re.sub(r'`([^`]+)`', _replace_code, text)

    # 2. Process bold: **text** or __text__ (prioritized over italic)
    def _replace_bold_double(match):
        bold_text = match.group(1)
        placeholder = _create_placeholder(('bold', bold_text))
        return placeholder
    text = re.sub(r'\*\*([^*]+?)\*\*', _replace_bold_double, text)
    text = re.sub(r'__([^_]+?)__', _replace_bold_double, text)

    # 3. Process italic: *text* or _text_ (after bold)
    def _replace_italic(match):
        italic_text = match.group(1)
        placeholder = _create_placeholder(('italic', italic_text))
        return placeholder
    # Match only single asterisks or underscores not inside bold
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', _replace_italic, text)
    text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', _replace_italic, text)

    # 4. Process links:[text](url)
    def _replace_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        placeholder = _create_placeholder(('link', link_text, link_url))
        return placeholder
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _replace_link, text)

    # 5. Escape HTML special characters
    text = html_module.escape(text)

    # 6. Convert Markdown markers to HTML using regular expressions for precise global replacement
    # Build regular expression pattern for placeholders
    # Placeholder format: prefix + unique ID (timestamp_random_number_counter) + suffix
    # Unique ID contains digits, underscores, and hyphens, using non-greedy matching to ensure exact matching
    placeholder_pattern = re.compile(
        re.escape(PLACEHOLDER_PREFIX) + r'[\d_\-]+' + re.escape(PLACEHOLDER_SUFFIX)
    )

    def _replace_placeholder(match):
        """Replace matched placeholders"""
        placeholder = match.group(0)
        if placeholder in placeholders:
            content = placeholders[placeholder]
            if content[0] == 'code':
                return f'<code>{html_module.escape(content[1])}</code>'
            elif content[0] == 'bold':
                return f'<strong>{html_module.escape(content[1])}</strong>'
            elif content[0] == 'italic':
                return f'<em>{html_module.escape(content[1])}</em>'
            elif content[0] == 'link':
                return f'<a href="{html_module.escape(content[2])}">{html_module.escape(content[1])}</a>'
            else:
                return html_module.escape(str(content))
        else:
            # If placeholder not in dictionary, log warning and return empty string
            _print(f"[WARN] Discover unregistered placeholder: {placeholder[:50]}...")
            return ''

    # Perform global replacement using regular expressions
    text = placeholder_pattern.sub(_replace_placeholder, text)

    # 7. Cleanup mechanism: check and clean all un-replaced placeholders (including old format placeholders)
    # Match all possible placeholder formats (including old format [MD_PLACEHOLDER_X] and new format)
    old_placeholder_pattern = re.compile(r'\[MD_PLACEHOLDER_\d+\]')
    new_placeholder_pattern = re.compile(
        re.escape(PLACEHOLDER_PREFIX) + r'[\d_\-]+' + re.escape(PLACEHOLDER_SUFFIX)
    )

    # Check and clean old format placeholders
    old_placeholders_found = old_placeholder_pattern.findall(text)
    if old_placeholders_found:
        _print(f"[WARN] Found {len(old_placeholders_found)} un-replaced old format placeholders, cleaning up...")
        text = old_placeholder_pattern.sub('', text)

    # Check and clean new format placeholders (theoretically should not exist, but for safety)
    remaining_placeholders = new_placeholder_pattern.findall(text)
    if remaining_placeholders:
        _print(f"[WARN] Found {len(remaining_placeholders)} un-replaced new format placeholders, cleaning up...")
        text = new_placeholder_pattern.sub('', text)

    return text


def _format_json_to_html(json_data: Any, level: int = 0) -> str:
    """
    Convert JSON data into a professional and well-formatted HTML report

    Args:
        json_data: JSONData (dict, list, or other type)
        level: Nesting level

    Returns:
        Formatted HTML string
    """
    import html as html_module

    if isinstance(json_data, dict):
        if not json_data:
            return '<p><em>No data provided</em></p>'

        # Define core semantic fields and their corresponding rendering methods
        html_parts = []
        processed_keys = set()

        # 1. Handle heading types (Title/Header)
        for key in ['report_title', 'title']:
            if key in json_data:
                val = json_data[key]
                if isinstance(val, str) and val.strip():
                    tag = 'h1' if key == 'report_title' else 'h2'
                    html_parts.append(f'<{tag}>{html_module.escape(val)}</{tag}>')
                    processed_keys.add(key)

        # 2. Handle summary types (Summary/Overview)
        for key in ['summary', 'executive_summary', 'overview', 'description', 'business_overview']:
            if key in json_data:
                val = json_data[key]
                if val:
                    html_parts.append('<div class="summary-card">')
                    html_parts.append(f'<span class="conclusion-title">{key.replace("_", " ").upper()}</span>')
                    if isinstance(val, str):
                        html_parts.append(_format_analysis_content(val))
                    else:
                        html_parts.append(_format_json_to_html(val, level + 1))
                    html_parts.append('</div>')
                    processed_keys.add(key)

        # 3. Handle highlight types (Highlights/Findings)
        for key in ['highlights', 'key_findings', 'findings', 'key_highlights', 'results', 'key_metrics']:
            if key in json_data:
                val = json_data[key]
                if val:
                    html_parts.append(f'<h3>{key.replace("_", " ").title()}</h3>')
                    if isinstance(val, list):
                        for item in val:
                            html_parts.append('<div class="highlight-item">')
                            html_parts.append('<span class="highlight-marker">●</span>')
                            if isinstance(item, (dict, list)):
                                html_parts.append(f'<div>{_format_json_to_html(item, level + 1)}</div>')
                            else:
                                html_parts.append(f'<span>{html_module.escape(str(item))}</span>')
                            html_parts.append('</div>')
                    elif isinstance(val, str):
                        html_parts.append(_format_analysis_content(val))
                    else:
                        html_parts.append(_format_json_to_html(val, level + 1))
                    processed_keys.add(key)

        # 4. Handle conclusion types (Conclusion/Recommendation)
        for key in ['conclusion', 'conclusions', 'recommendation', 'recommendations', 'final_answer', 'investment_conclusion', 'investment_recommendation']:
            if key in json_data:
                val = json_data[key]
                if val:
                    html_parts.append('<div class="conclusion-box">')
                    html_parts.append(f'<span class="conclusion-title">{key.replace("_", " ").upper()}</span>')
                    if isinstance(val, str):
                        html_parts.append(_format_analysis_content(val))
                    else:
                        html_parts.append(_format_json_to_html(val, level + 1))
                    html_parts.append('</div>')
                    processed_keys.add(key)

        # 5. Handle all remaining keys
        remaining_keys = [k for k in json_data.keys() if k not in processed_keys]
        if remaining_keys:
            for key in remaining_keys:
                val = json_data[key]
                key_display = key.replace('_', ' ').title()

                if isinstance(val, (int, float, bool, type(None))):
                    html_parts.append(f'<p><span class="data-label">{key_display}:</span> <span class="data-value">{html_module.escape(str(val))}</span></p>')
                elif isinstance(val, str):
                    if len(val) < 150:
                        html_parts.append(f'<p><span class="data-label">{key_display}:</span> <span class="data-value">{html_module.escape(val)}</span></p>')
                    else:
                        html_parts.append(f'<h3>{key_display}</h3>')
                        html_parts.append(_format_analysis_content(val))
                else:
                    html_parts.append(f'<h3>{key_display}</h3>')
                    html_parts.append(_format_json_to_html(val, level + 1))

        return '<div>' + ''.join(html_parts) + '</div>'

    elif isinstance(json_data, list):
        if not json_data:
            return '<p><em>Empty list</em></p>'

        html_parts = ['<ul>']
        for item in json_data:
            if isinstance(item, (dict, list)):
                html_parts.append(f'<li>{_format_json_to_html(item, level + 1)}</li>')
            else:
                html_parts.append(f'<li>{html_module.escape(str(item))}</li>')
        html_parts.append('</ul>')
        return ''.join(html_parts)

    else:
        # Render basic types directly
        value_str = str(json_data)
        if len(value_str) > 200:
            return f'<p>{html_module.escape(value_str)}</p>'
        return html_module.escape(value_str)


def _detect_indicators(df: pd.DataFrame) -> List[str]:
    """
    Detect technical indicator columns in DataFrame

    Args:
        df: DataFrame with potential indicator columns

    Returns:
        List of indicator column names
    """
    # Common technical indicator column names (case-insensitive)
    indicator_keywords = [
        'rsi', 'macd', 'atr', 'sma', 'ema', 'bollinger', 'bb',
        'stoch', 'stochastic', 'williams', 'wr', 'cci', 'adx',
        'obv', 'mfi', 'roc', 'momentum', 'signal', 'hist', 'histogram',
        'upper', 'lower', 'middle', 'band'
    ]

    indicators = []
    for col in df.columns:
        col_lower = str(col).lower()
        # Skip OHLCV columns
        if col_lower in ['date', 'datetime', 'time', 'timestamp',
                        'open', 'high', 'low', 'close', 'volume', 'vol',
                        'o', 'h', 'l', 'c', 'v', 'dividends', 'stock splits',
                        'price']:
            continue

        # Check if column name contains indicator keywords
        for keyword in indicator_keywords:
            if keyword in col_lower:
                indicators.append(col)
                break

    return indicators


def _prepare_indicator_data(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Prepare data for indicator charts

    Args:
        df: Original DataFrame
        date_col: Date column name (if None, will be detected)

    Returns:
        DataFrame with Date as index
    """
    df_prep = df.copy()

    # Find date column
    if date_col is None:
        for col in df_prep.columns:
            col_lower = str(col).lower()
            if col_lower in ['date', 'datetime', 'time', 'timestamp']:
                date_col = col
                break

        if date_col is None:
            if isinstance(df_prep.index, pd.DatetimeIndex):
                df_prep = df_prep.reset_index()
                date_col = df_prep.columns[0]
            else:
                raise ValueError("Date column not found in data")

    # Ensure date column is datetime type and set as index
    df_prep[date_col] = pd.to_datetime(df_prep[date_col])
    df_prep = df_prep.set_index(date_col)
    df_prep.index.name = 'Date'

    # Sort by date
    df_prep = df_prep.sort_index()

    return df_prep


def _generate_indicator_charts(
    df: pd.DataFrame,
    output_dir: str,
    file_basename: str,
    chart_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate technical indicator charts using matplotlib

    Args:
        df: DataFrame containing indicator data
        output_dir: Chart output directory
        file_basename: Base name for chart files
        chart_config: Chart configuration (optional)

    Returns:
        List of generated chart file paths
    """
    if plt is None:
        raise ImportError("matplotlib is not installed")

    chart_paths = []

    # Default configuration
    default_config = {
        'figsize': (12, 6),
        'dpi': 100,
        'style': 'default'
    }

    if chart_config:
        default_config.update(chart_config)

    # Detect indicators
    indicators = _detect_indicators(df)

    if not indicators:
        _print(f"[INFO] No technical indicators detected in data")
        return []

    # Prepare indicator data with date index
    try:
        df_prep = _prepare_indicator_data(df)
    except Exception as e:
        _print(f"[WARN] Failed to prepare indicator data: {e}")
        return []

    # Group indicators for combined charts
    # MACD group: MACD, MACD_Signal, MACD_Hist
    macd_indicators = [ind for ind in indicators if 'macd' in str(ind).lower()]
    other_indicators = [ind for ind in indicators if 'macd' not in str(ind).lower()]

    # Generate MACD chart (if MACD indicators exist)
    if macd_indicators:
        try:
            has_data = False
            fig, axes = plt.subplots(2, 1, figsize=default_config['figsize'],
                                    sharex=True, height_ratios=[2, 1])

            # Top subplot: MACD and MACD_Signal
            ax1 = axes[0]
            for ind in macd_indicators:
                ind_lower = str(ind).lower()
                # Exclude histogram, plot MACD and MACD_Signal lines
                if 'hist' not in ind_lower:
                    if ind in df_prep.columns:
                        # Remove NaN values for plotting
                        data = df_prep[ind].dropna()
                        if len(data) > 0:
                            # data.index is already the date index from df_prep
                            ax1.plot(data.index, data.values, label=ind, linewidth=1.5)
                            has_data = True

            # Only proceed if we have data to plot
            if has_data:
                ax1.set_ylabel('MACD', fontsize=10)
                ax1.set_title('MACD Indicators', fontsize=12, fontweight='bold')
                if ax1.get_lines():  # Only add legend if there are lines
                    ax1.legend(loc='best', fontsize=9)
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

                # Bottom subplot: MACD_Hist
                ax2 = axes[1]
                hist_has_data = False
                for ind in macd_indicators:
                    ind_lower = str(ind).lower()
                    if 'hist' in ind_lower:
                        if ind in df_prep.columns:
                            data = df_prep[ind].dropna()
                            if len(data) > 0:
                                # data.index is already the date index from df_prep
                                colors = ['green' if x >= 0 else 'red' for x in data.values]
                                ax2.bar(data.index, data.values, label=ind, color=colors, alpha=0.6, width=0.8)
                                hist_has_data = True

                ax2.set_ylabel('Histogram', fontsize=10)
                ax2.set_xlabel('Date', fontsize=10)
                if hist_has_data and ax2.patches:  # Only add legend if there are bars
                    ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                plt.tight_layout()
                chart_filename = f"chart_{file_basename}_MACD.png"
                chart_path = os.path.join(output_dir, chart_filename)
                plt.savefig(chart_path, dpi=default_config['dpi'],
                           bbox_inches='tight', facecolor='white')
                plt.close()

                chart_paths.append(chart_path)
                _print(f"[SUCCESS] MACD chart saved: {chart_path}")
            else:
                plt.close()
                _print(f"[WARN] MACD indicators detected but all values are NaN, skipping MACD chart")
        except Exception as e:
            _print(f"[ERROR] Failed to generate MACD chart: {e}")
            plt.close('all')

    # Generate RSI chart (if RSI exists)
    if any('rsi' in str(ind).lower() for ind in indicators):
        try:
            rsi_col = None
            for ind in indicators:
                if 'rsi' in str(ind).lower():
                    rsi_col = ind
                    break

            if rsi_col and rsi_col in df_prep.columns:
                fig, ax = plt.subplots(figsize=default_config['figsize'])

                data = df_prep[rsi_col].dropna()
                if len(data) > 0:
                    # data.index is already the date index from df_prep
                    ax.plot(data.index, data.values,
                           label=rsi_col, linewidth=1.5, color='purple')

                    # Add overbought/oversold lines
                    ax.axhline(y=70, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
                    ax.axhline(y=30, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
                    ax.fill_between(data.index, 30, 70,
                                   alpha=0.1, color='gray')

                    ax.set_ylabel('RSI', fontsize=10)
                    ax.set_xlabel('Date', fontsize=10)
                    ax.set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
                    ax.set_ylim(0, 100)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    chart_filename = f"chart_{file_basename}_RSI.png"
                    chart_path = os.path.join(output_dir, chart_filename)
                    plt.savefig(chart_path, dpi=default_config['dpi'],
                               bbox_inches='tight', facecolor='white')
                    plt.close()

                    chart_paths.append(chart_path)
                    _print(f"[SUCCESS] RSI chart saved: {chart_path}")
                else:
                    plt.close()
                    _print(f"[WARN] RSI indicator detected but all values are NaN, skipping RSI chart")
        except Exception as e:
            _print(f"[ERROR] Failed to generate RSI chart: {e}")
            plt.close('all')

    # Generate ATR chart (if ATR exists)
    if any('atr' in str(ind).lower() for ind in indicators):
        try:
            atr_col = None
            for ind in indicators:
                if 'atr' in str(ind).lower():
                    atr_col = ind
                    break

            if atr_col and atr_col in df_prep.columns:
                fig, ax = plt.subplots(figsize=default_config['figsize'])

                data = df_prep[atr_col].dropna()
                if len(data) > 0:
                    # data.index is already the date index from df_prep
                    ax.plot(data.index, data.values,
                           label=atr_col, linewidth=1.5, color='orange')

                    ax.set_ylabel('ATR', fontsize=10)
                    ax.set_xlabel('Date', fontsize=10)
                    ax.set_title('Average True Range (ATR)', fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    chart_filename = f"chart_{file_basename}_ATR.png"
                    chart_path = os.path.join(output_dir, chart_filename)
                    plt.savefig(chart_path, dpi=default_config['dpi'],
                               bbox_inches='tight', facecolor='white')
                    plt.close()

                    chart_paths.append(chart_path)
                    _print(f"[SUCCESS] ATR chart saved: {chart_path}")
                else:
                    plt.close()
                    _print(f"[WARN] ATR indicator detected but all values are NaN, skipping ATR chart")
        except Exception as e:
            _print(f"[ERROR] Failed to generate ATR chart: {e}")
            plt.close('all')

    # Generate chart for other indicators (if any)
    remaining_indicators = [ind for ind in other_indicators
                           if 'rsi' not in str(ind).lower() and 'atr' not in str(ind).lower()]

    if remaining_indicators:
        try:
            fig, ax = plt.subplots(figsize=default_config['figsize'])

            for ind in remaining_indicators:
                if ind in df_prep.columns:
                    data = df_prep[ind].dropna()
                    if len(data) > 0:
                        # data.index is already the date index from df_prep
                        ax.plot(data.index, data.values,
                               label=ind, linewidth=1.5)

            if ax.get_lines():  # Check if any lines were plotted
                ax.set_ylabel('Value', fontsize=10)
                ax.set_xlabel('Date', fontsize=10)
                ax.set_title('Other Technical Indicators', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                chart_filename = f"chart_{file_basename}_Other_Indicators.png"
                chart_path = os.path.join(output_dir, chart_filename)
                plt.savefig(chart_path, dpi=default_config['dpi'],
                           bbox_inches='tight', facecolor='white')
                plt.close()

                chart_paths.append(chart_path)
                _print(f"[SUCCESS] Other indicators chart saved: {chart_path}")
            else:
                plt.close()
        except Exception as e:
            _print(f"[ERROR] Failed to generate other indicators chart: {e}")
            plt.close('all')

    return chart_paths


def _prepare_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare OHLCV data for mplfinance

    Args:
        df: Original DataFrame

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    # Find date column
    date_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in ['date', 'datetime', 'time', 'timestamp']:
            date_col = col
            break

    if date_col is None:
        # If no date column, try to use index
        if isinstance(df.index, pd.DatetimeIndex):
            df_prep = df.copy()
            df_prep = df_prep.reset_index()
            date_col = df_prep.columns[0]
        else:
            raise ValueError("Date column not found in data, and index is not DatetimeIndex")
    else:
        df_prep = df.copy()

    # Ensure date column is datetime type
    df_prep[date_col] = pd.to_datetime(df_prep[date_col])
    df_prep = df_prep.set_index(date_col)
    df_prep.index.name = 'Date'

    # Normalize column names (case-insensitive)
    col_mapping = {}
    for col in df_prep.columns:
        col_lower = str(col).lower()
        if col_lower in ['open', 'o']:
            col_mapping[col] = 'Open'
        elif col_lower in ['high', 'h']:
            col_mapping[col] = 'High'
        elif col_lower in ['low', 'l']:
            col_mapping[col] = 'Low'
        elif col_lower in ['close', 'c', 'price']:
            col_mapping[col] = 'Close'
        elif col_lower in ['volume', 'vol', 'v']:
            col_mapping[col] = 'Volume'

    df_prep = df_prep.rename(columns=col_mapping)

    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df_prep.columns]
    if missing_cols:
        raise ValueError(f"Data missing required columns: {missing_cols}")

    # If Volume doesn't exist, create a dummy column
    if 'Volume' not in df_prep.columns:
        df_prep['Volume'] = 0

    # Keep only required columns
    df_prep = df_prep[required_cols + ['Volume']]

    # Handle NaN values: Remove rows with NaN in required OHLC columns
    # This is important because mplfinance cannot handle NaN values in OHLC data
    initial_rows = len(df_prep)
    df_prep = df_prep.dropna(subset=required_cols)

    # Check if we still have valid data after removing NaN rows
    if df_prep.empty:
        raise ValueError(
            f"After removing rows with NaN values in OHLC columns, no valid data remains. "
            f"Original data had {initial_rows} rows."
        )

    # Warn if we removed many rows
    removed_rows = initial_rows - len(df_prep)
    if removed_rows > 0:
        _print(f"[WARN] Removed {removed_rows} row(s) with NaN values in OHLC columns "
              f"(out of {initial_rows} total rows)")

    # Handle NaN values in Volume column (fill with 0 if NaN)
    if 'Volume' in df_prep.columns:
        df_prep['Volume'] = df_prep['Volume'].fillna(0)

    # Sort by date
    df_prep = df_prep.sort_index()

    return df_prep


def _generate_charts(
    data_files: List[str],
    output_dir: str,
    chart_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Visualization Layer: Use mplfinance to generate candlestick charts and indicator charts, save as PNG

    Args:
        data_files: List of data file paths
        output_dir: Chart output directory
        chart_config: Chart configuration (optional)

    Returns:
        List of generated chart file paths
    """
    if mpf is None:
        raise ImportError("mplfinance is not installed")

    _ensure_dir(output_dir)
    chart_paths = []

    # Default chart configuration
    default_config = {
        'type': 'candle',  # 'candle', 'line', 'ohlc', 'renko', 'pnf'
        'style': 'yahoo',  # 'yahoo', 'binance', 'charles', 'checkers', 'classic', 'default', 'mike', 'nightclouds', 'sas', 'starsandstripes'
        'volume': True,
        'figsize': (12, 8),
        'savefig': {
            'dpi': 100,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        },
        'generate_ohlcv': True,  # Whether to generate OHLCV charts
        'generate_indicators': True  # Whether to generate indicator charts
    }

    if chart_config:
        default_config.update(chart_config)

    # Note: data_files should already be validated before calling this function
    # But we still do a quick check here for safety
    if not data_files:
        _print(f"[WARN] No data files provided, skipping chart generation")
        return []

    for idx, data_file in enumerate(data_files):
        try:
            _print(f"[INFO] Generating chart for file {data_file}...")

            # Read data (already validated, but catch any unexpected errors)
            df = _load_data_file(data_file)

            # Generate chart filename base
            file_basename = os.path.splitext(os.path.basename(data_file))[0]

            # Try to generate OHLCV chart (if enabled and data available)
            if default_config.get('generate_ohlcv', True):
                try:
                    df_ohlcv = _prepare_ohlcv_data(df)

                    chart_filename = f"chart_{file_basename}_OHLCV_{idx+1}.png"
                    chart_path = os.path.join(output_dir, chart_filename)

                    # Generate chart
                    savefig_config = default_config['savefig'].copy()
                    savefig_config['fname'] = chart_path

                    mpf.plot(
                        df_ohlcv,
                        type=default_config['type'],
                        style=default_config['style'],
                        volume=default_config['volume'],
                        figsize=default_config['figsize'],
                        savefig=savefig_config
                    )

                    chart_paths.append(chart_path)
                    _print(f"[SUCCESS] OHLCV chart saved: {chart_path}")
                except Exception as e:
                    _print(f"[WARN] Failed to generate OHLCV chart for {data_file}: {e}")
                    # Continue to try indicator charts

            # Try to generate indicator charts (if enabled and indicators available)
            if default_config.get('generate_indicators', True):
                try:
                    indicator_charts = _generate_indicator_charts(
                        df=df,
                        output_dir=output_dir,
                        file_basename=f"{file_basename}_{idx+1}",
                        chart_config=chart_config
                    )
                    chart_paths.extend(indicator_charts)
                    if indicator_charts:
                        _print(f"[SUCCESS] Generated {len(indicator_charts)} indicator chart(s) for {data_file}")
                except Exception as e:
                    _print(f"[WARN] Failed to generate indicator charts for {data_file}: {e}")
                    # Continue processing other files

        except Exception as e:
            _print(f"[ERROR] Failed to generate chart for {data_file}: {e}")
            # Continue processing other files
            continue

    return chart_paths


def _get_default_template() -> str:
    """
    Get optimized default HTML template.
    Addressed issues: Fragile layout, Global pollution, Rudimentary styling, Overflow bugs.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title | default('Financial Analysis Report') }}</title>
    <style>
        :root {
            --primary: #1e40af;       /* Core brand blue */
            --primary-light: #3b82f6;
            --accent: #7c3aed;        /* Purple accent */
            --text-main: #111827;
            --text-muted: #4b5563;
            --bg-page: #f9fafb;
            --bg-card: #ffffff;
            --border: #e5e7eb;
        }

        /* --- Page definition (Page Settings) --- */
        @page {
            size: A4;
            margin: 2cm 1.5cm;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: var(--text-muted);
            }
            @bottom-left {
                content: "Finance Tool Universe Analysis Report";
                font-size: 9pt;
                color: var(--text-muted);
            }
        }

        /* Cover page independent rules: no header or footer, no margins */
        @page :first {
            margin: 0;
            @bottom-right { content: none; }
            @bottom-left { content: none; }
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial,
                         "PingFang SC", "Microsoft YaHei", "Source Han Sans CN", sans-serif;
            line-height: 1.6;
            color: var(--text-main);
            margin: 0;
            padding: 0;
            background-color: var(--bg-page);
            -webkit-print-color-adjust: exact;
        }

        /* --- Cover (Cover Page) --- */
        .report-cover {
            height: 297mm; /* Strictly conform to A4 height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            color: white;
            text-align: center;
            padding: 0 10%;
            box-sizing: border-box;
            page-break-after: always;
            break-after: page;
        }

        .cover-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 2rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .cover-meta {
            font-size: 1.2rem;
            opacity: 0.9;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .symbol-badge {
            display: inline-block;
            margin-top: 1.5rem;
            padding: 0.5rem 1.5rem;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 50px;
            font-weight: 600;
            letter-spacing: 0.05em;
        }

        /* --- Main text layout (Main Content) --- */
        main {
            padding-bottom: 3rem;
        }

        .report-section {
            background: var(--bg-card);
            margin: 1.5rem;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            page-break-before: always;
            break-before: page;
        }

        /* The first chapter does not enforce a page break immediately after the cover */
        .report-section:first-of-type {
            page-break-before: avoid;
            break-before: avoid;
            margin-top: 0;
        }

        .section-header {
            font-size: 1.8rem;
            color: var(--primary);
            border-left: 6px solid var(--primary);
            padding-left: 1rem;
            margin-bottom: 2rem;
            page-break-after: avoid;
        }

        /* --- Text styling (Prose) --- */
        .prose {
            font-size: 1.05rem;
            line-height: 1.8;
            color: #374151;
            margin: 1rem 0;
        }

        .prose p { margin-bottom: 1.2rem; text-align: justify; }
        .prose p:last-child { margin-bottom: 0; }
        .prose h1, .prose h2, .prose h3 { color: var(--primary); page-break-after: avoid; margin-top: 1.5rem; }
        .prose h1:first-child, .prose h2:first-child, .prose h3:first-child { margin-top: 0; }

        .prose ul, .prose ol { padding-left: 1.5rem; margin-bottom: 1.5rem; }
        .prose li { margin-bottom: 0.5rem; page-break-inside: avoid; }

        /* Semantic components (Semantic Components) */
        .summary-card {
            background-color: #f8fafc;
            border-left: 5px solid var(--primary-light);
            padding: 1.5rem 2rem;
            border-radius: 0 12px 12px 0;
            margin: 1.5rem 0;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.02);
        }

        .highlight-item {
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            padding: 0.5rem;
            background: rgba(59, 130, 246, 0.05);
            border-radius: 8px;
        }

        .highlight-marker {
            color: var(--primary);
            font-weight: bold;
            margin-right: 0.75rem;
            font-size: 1.2rem;
            line-height: 1;
        }

        .conclusion-box {
            background: linear-gradient(to right, rgba(124, 58, 237, 0.05), rgba(59, 130, 246, 0.05));
            border: 1px dashed var(--accent);
            padding: 1.5rem;
            border-radius: 12px;
            margin-top: 2rem;
        }

        .conclusion-title {
            color: var(--accent);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            display: block;
        }

        .data-label {
            font-weight: 600;
            color: var(--text-muted);
            min-width: 120px;
            display: inline-block;
        }

        .data-value {
            color: var(--text-main);
            font-weight: 500;
        }

        /* Code and original content */
        .prose pre {
            background: #f8fafc;
            border: 1px solid var(--border);
            padding: 1.25rem;
            border-radius: 8px;
            font-family: "JetBrains Mono", "Cascadia Code", "Courier New", monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            box-decoration-break: clone;
        }

        .prose blockquote {
            border-left: 4px solid var(--primary-light);
            background: rgba(59, 130, 246, 0.03);
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            font-style: italic;
        }

        /* --- Table design (Professional Tables) --- */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            table-layout: auto;
        }

        th {
            background-color: #f1f5f9;
            color: var(--text-main);
            font-weight: 700;
            padding: 1rem;
            border-bottom: 2px solid var(--primary);
            text-align: left;
        }

        td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }

        /* Zebra striping */
        tr:nth-child(even) { background-color: #f8fafc; }

        /* Page break protection */
        tr { page-break-inside: avoid; break-inside: avoid; }

        /* Optimize styling for numeric columns */
        .num {
            font-family: "JetBrains Mono", monospace;
            text-align: right;
            white-space: nowrap;
        }

        /* --- Chart container --- */
        .chart-wrapper {
            margin: 2.5rem 0;
            text-align: center;
            page-break-inside: avoid;
        }

        .chart-wrapper img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }

        .chart-caption {
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        /* --- Footer --- */
        .footer-banner {
            text-align: center;
            padding: 3rem 1.5rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin: 2rem 1.5rem 0;
        }

        /* Print mode override */
        @media print {
            body { background-color: white; }
            .report-section { box-shadow: none; margin: 0; padding: 2rem 0; border-radius: 0; }
            .report-section + .report-section { border-top: 1px solid var(--border); }
        }
    </style>
</head>
<body>
    <header class="report-cover">
        <div class="cover-content">
            <h1 class="cover-title">{{ report_title | default('Financial Analysis Report') }}</h1>
            <div class="cover-meta">
                <span>Generated Time: {{ generation_time | default('N/A') }}</span>
                {% if symbol %}
                <div class="symbol-badge">SYMBOL: {{ symbol }}</div>
                {% endif %}
            </div>
        </div>
    </header>

    <main>
        {% if analysis_content %}
        <section class="report-section">
            <h2 class="section-header">Executive Summary</h2>
            <article class="prose">
                {{ analysis_content | safe }}
            </article>
        </section>
        {% endif %}

        {% if charts %}
        <section class="report-section">
            <h2 class="section-header">Technical Analysis</h2>
            {% for chart in charts %}
            <figure class="chart-wrapper">
                <figcaption class="chart-caption">{{ chart.title | default('Technical Chart ' + loop.index|string) }}</figcaption>
                <img src="{{ chart.path }}" alt="{{ chart.title }}">
            </figure>
            {% endfor %}
        </section>
        {% endif %}

        {% if summary_data %}
        <section class="report-section">
            <h2 class="section-header">Key Indicators</h2>
            <table>
                <thead>
                    <tr>
                        {% for key in summary_data.keys() %}
                        <th>{{ key }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        {% for value in summary_data.values() %}
                        <td class="{% if value is number %}num{% endif %}">{{ value }}</td>
                        {% endfor %}
                    </tr>
                </tbody>
            </table>
        </section>
        {% endif %}

        {% if additional_sections %}
        {% for section in additional_sections %}
        <section class="report-section">
            <h2 class="section-header">{{ section.title }}</h2>
            <article class="prose">
                {{ section.content | safe }}
            </article>
        </section>
        {% endfor %}
        {% endif %}
    </main>

    <footer class="footer-banner">
        <p>This report is automatically generated by Finance Tool Universe</p>
        <p>© {{ generation_time[:4] if generation_time else '2024' }} Financial Technology Lab. Confidential.</p>
    </footer>
</body>
</html>
"""


def _render_html(
    llm_json: Dict[str, Any],
    chart_paths: List[str],
    template_path: Optional[str] = None
) -> str:
    """
    Orchestration Layer: Use Jinja2 to combine LLM text analysis + generated image paths into HTML

    Args:
        llm_json: LLM output JSON report data (dict)
        chart_paths: List of generated chart file paths
        template_path: Custom template path (optional)

    Returns:
        Rendered HTML string
    """
    if Environment is None or Template is None:
        raise ImportError("jinja2 is not installed")

    # Ensure llm_json is a dictionary type
    if not isinstance(llm_json, dict):
        _print(f"[WARN] _render_html Received llm_json is not a dictionary type: {type(llm_json)}, convert to dictionary")
        llm_json = {
            'title': 'Report Content',
            'analysis': str(llm_json),
            'content': str(llm_json),
            'text': str(llm_json)
        }

    # Smart extract and analyze content: try multiple possible field names
    analysis_content = ''

    # First check if there is a results array (common in API response format)
    if 'results' in llm_json and isinstance(llm_json['results'], list) and len(llm_json['results']) > 0:
        first_result = llm_json['results'][0]
        if isinstance(first_result, dict):
            # Extract content from the first result
            if 'final_answer' in first_result:
                analysis_content = first_result['final_answer']
            elif 'analysis' in first_result:
                analysis_content = first_result['analysis']
            elif 'content' in first_result:
                analysis_content = first_result['content']
            elif 'text' in first_result:
                analysis_content = first_result['text']
            elif 'message' in first_result:
                analysis_content = first_result['message']
            elif 'result' in first_result:
                analysis_content = first_result['result']

    # If no content is found, try extracting from top-level fields
    if not analysis_content:
        if 'final_answer' in llm_json:
            analysis_content = llm_json['final_answer']
        elif 'analysis' in llm_json:
            analysis_content = llm_json['analysis']
        elif 'content' in llm_json:
            analysis_content = llm_json['content']
        elif 'text' in llm_json:
            analysis_content = llm_json['text']
        elif 'message' in llm_json:
            analysis_content = llm_json['message']
        elif 'result' in llm_json:
            analysis_content = llm_json['result']

    # If still no content, try looking for a nested structure containing "final_answer"
    if not analysis_content:
        def _extract_from_nested(obj, target_key='final_answer', max_depth=3, current_depth=0):
            """Recursively search for the target key in nested dictionaries"""
            if current_depth >= max_depth:
                return None
            if isinstance(obj, dict):
                if target_key in obj:
                    value = obj[target_key]
                    if value and isinstance(value, str) and value.strip():
                        return value
                for value in obj.values():
                    result = _extract_from_nested(value, target_key, max_depth, current_depth + 1)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = _extract_from_nested(item, target_key, max_depth, current_depth + 1)
                    if result:
                        return result
            return None

        nested_content = _extract_from_nested(llm_json, 'final_answer')
        if nested_content:
            analysis_content = nested_content
        else:
            # Finally, try converting the entire dictionary into formatted text (but exclude some metadata fields)
            filtered_dict = {k: v for k, v in llm_json.items()
                           if k not in ['timestamp', 'total_queries', 'success_count', 'overall_recall', 'tools', 'arguments']}
            try:
                analysis_content = json.dumps(filtered_dict, ensure_ascii=False, indent=2)
            except:
                analysis_content = str(llm_json)

    # Ensure analysis_content is of string type
    if not isinstance(analysis_content, str):
        analysis_content = str(analysis_content)

    # Extract title - prioritize extraction from top level
    report_title = llm_json.get('title') or llm_json.get('report_title')

    # Parse the analysis content into structured data (supporting JSON strings or Python single-quoted dictionaries)
    analysis_data = _safe_load_json(analysis_content)

    # If the analysis content itself is structured and there is no title at the top level, try extracting the title from the analysis content
    if not report_title and isinstance(analysis_data, dict):
        report_title = analysis_data.get('report_title') or analysis_data.get('title')

    # If still no title, use a default value
    if not report_title:
        report_title = 'Financial Analysis Report'

    if not isinstance(report_title, str):
        report_title = str(report_title)

    # Format the analysis content into a clean and readable formatHTML
    # If it's already structured data, directly call _format_json_to_html
    if isinstance(analysis_data, (dict, list)):
        analysis_content_html = _format_json_to_html(analysis_data)
    else:
        analysis_content_html = _format_analysis_content(analysis_content)

    # Extract symbol
    symbol = llm_json.get('symbol')
    if not symbol and isinstance(analysis_data, dict):
        # Try extracting the symbol from the analysis result
        symbol = analysis_data.get('symbol') or analysis_data.get('ticker') or analysis_data.get('symbols')
        # If symbols is a list, convert it to a string
        if isinstance(symbol, list):
            symbol = ", ".join([str(s) for s in symbol])

    if symbol is not None and not isinstance(symbol, str):
        symbol = str(symbol)

    # Extract summary_data and ensure it is of dictionary type
    summary_data = llm_json.get('summary', {})
    if not isinstance(summary_data, dict):
        if summary_data is not None:
            summary_data = {'Summary': str(summary_data)}
        else:
            summary_data = {}

    # Extract additional_sections and ensure it is of list type
    additional_sections = llm_json.get('sections', [])
    if not isinstance(additional_sections, list):
        if additional_sections is not None:
            additional_sections = [{'title': 'Additional Content', 'content': str(additional_sections)}]
        else:
            additional_sections = []

    # Extract chart_titles and ensure it is of dictionary type
    chart_titles = llm_json.get('chart_titles', {})
    if not isinstance(chart_titles, dict):
        chart_titles = {}

    # Prepare template data
    template_data = {
        'report_title': report_title,
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'analysis_content': analysis_content_html,
        'summary_data': summary_data,
        'charts': [],
        'additional_sections': additional_sections
    }

    # Process chart paths
    for idx, chart_path in enumerate(chart_paths):
        # Extract the filename from the chart file path (without extension) as the title
        chart_filename = os.path.basename(chart_path)  # Get the filename, for example "chart_qcom_stock_2017_2019_3.png"
        chart_title = os.path.splitext(chart_filename)[0]  # Remove the extension, for example "chart_qcom_stock_2017_2019_3"

        # If a custom title is provided in llm_json, prioritize using the custom title
        custom_title = llm_json.get('chart_titles', {}).get(str(idx))
        if custom_title:
            chart_title = custom_title

        chart_info = {
            'path': chart_path,
            'title': chart_title
        }
        template_data['charts'].append(chart_info)

    # Load template
    if template_path and os.path.exists(template_path):
        # Use file template
        template_dir = os.path.dirname(template_path)
        template_name = os.path.basename(template_path)
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)
    else:
        # Use default template
        template = Template(_get_default_template())

    # Render HTML
    html_content = template.render(**template_data)

    return html_content


def _generate_pdf(html_content: str, output_path: str) -> Optional[str]:
    """
    Rendering Layer: Use WeasyPrint to render HTML into final PDF(Otherwise, only save HTML)

    Args:
        html_content: HTML content string
        output_path: PDF output path

    Returns:
        Generated PDF Or HTML file path; if WeasyPrint is unavailable, save as a .html file with the same name and return the path, ensuring the caller receives the output path.
    """
    _ensure_dir(os.path.dirname(output_path))

    if not WEASYPRINT_AVAILABLE or HTML is None:
        # WeasyPrint or underlying libraries (such as libgobject) missing: no PDF generated, saving as alternative HTML
        html_path = output_path.rstrip(".pdf") if output_path.lower().endswith(".pdf") else output_path
        if not html_path.lower().endswith(".html"):
            html_path += ".html"
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            _print(f"[WARN] WeasyPrint Unavailable, PDF generation skipped, report saved as HTML: {html_path}")
            return html_path
        except Exception as e:
            _print(f"[ERROR] Failed to save HTML report: {e}")
            return None

    try:
        # Use WeasyPrint to convert HTML to PDF
        HTML(string=html_content, base_url=os.path.dirname(output_path)).write_pdf(output_path)
        _print(f"[SUCCESS] PDF report generated: {output_path}")
        return output_path
    except Exception as e:
        _print(f"[ERROR] PDF generation failed: {e}")
        # Downgraded: saving HTML for user to still view
        html_path = output_path.rstrip(".pdf") if output_path.lower().endswith(".pdf") else output_path
        if not html_path.lower().endswith(".html"):
            html_path += ".html"
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            _print(f"[WARN] Downgraded and saved as HTML: {html_path}")
            return html_path
        except Exception as e2:
            _print(f"[ERROR] Failed to save downgraded HTML file: {e2}")
            raise


def run_generate_report(args) -> None:
    """
    Main function: Generate complete PDF report

    Required parameters:
        llm_json: LLM output JSON report data (dict)
        output_path: PDF output path

    Optional parameters:
        data_files: List of data file paths (CSV/JSON) for generating charts
        chart_config: Chart configuration dictionary
        template_path: Custom HTML template path
        report_dir: Report output directory (default: configured output/report)
    """
    llm_json = getattr(args, "llm_json", None)
    output_path = getattr(args, "output_path", None)
    data_files = getattr(args, "data_files", None) or []
    chart_config = getattr(args, "chart_config", None)
    template_path = getattr(args, "template_path", None)
    # Note: getattr's default only takes effect when the attribute does not exist; if the upstream explicitly passes report_dir=None,
    # getattr it will still return None, leading to _ensure_dir(None) -> os.path.expanduser(None) error.
    # Here we use `or DEFAULT_REPORT_DIR` as fallback to ensure report_dir is always a valid string path.
    report_dir = getattr(args, "report_dir", None) or DEFAULT_REPORT_DIR

    if llm_json is None:
        raise ValueError("llm_json parameter is required")
    if not output_path:
        raise ValueError("output_path parameter is required")

    # Compatible with llm_json: supports file paths, JSON strings, plain strings, dictionaries, and other formats
    original_llm_json = llm_json
    llm_json_dict = None

    if isinstance(llm_json, dict):
        # Already a dictionary, using directly
        llm_json_dict = llm_json
    elif isinstance(llm_json, str):
        # String type: could be a file path, JSON string, or plain text
        llm_json_str = llm_json.strip()

        # First check if it's a file path
        if os.path.exists(llm_json_str) and os.path.isfile(llm_json_str):
            _print(f"[INFO] llm_json It's a file path, attempting to read the file: {llm_json_str}")
            try:
                with open(llm_json_str, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                # Attempting to parse as JSON
                try:
                    llm_json_dict = json.loads(file_content)
                    _print(f"[INFO] Successfully read from file and parsed as JSON")
                except json.JSONDecodeError:
                    # File content is not JSON, treating as text
                    _print(f"[WARN] File content is not valid JSON, will be displayed as text")
                    llm_json_dict = {
                        'title': 'Report Content',
                        'analysis': file_content,
                        'content': file_content,
                        'text': file_content
                    }
            except Exception as e:
                _print(f"[WARN] Failed to read file: {e}, using original string as text content")
                llm_json_dict = {
                    'title': 'Report Content',
                    'analysis': llm_json_str,
                    'content': llm_json_str,
                    'text': llm_json_str
                }
        else:
            # Not a file path, attempting to parse as JSON
            try:
                llm_json_dict = json.loads(llm_json_str)
                _print(f"[INFO] Successfully parsed JSON string")
            except json.JSONDecodeError:
                # Not valid JSON, treating as text content
                _print(f"[WARN] llm_json Not valid JSON, will be displayed as text content")
                llm_json_dict = {
                    'title': 'Report Content',
                    'analysis': llm_json_str,
                    'content': llm_json_str,
                    'text': llm_json_str
                }
    else:
        # Other types (such as list, int, float, etc.) converted to string for display
        _print(f"[WARN] llm_json Type is {type(llm_json)}, converting to text content for display")
        content_str = str(llm_json)
        llm_json_dict = {
            'title': 'Report Content',
            'analysis': content_str,
            'content': content_str,
            'text': content_str
        }

    # Ensure llm_json_dict is of dictionary type
    if not isinstance(llm_json_dict, dict):
        _print(f"[WARN] llm_json If it's still not a dictionary after processing, forcibly convert it to a dictionary")
        llm_json_dict = {
            'title': 'Report Content',
            'analysis': str(llm_json_dict),
            'content': str(llm_json_dict),
            'text': str(llm_json_dict)
        }

    # Use the processed dictionary
    llm_json = llm_json_dict

    # Ensure output directory exists
    if not isinstance(report_dir, str) or not report_dir.strip():
        # Double safety: prevent exceptions from invalid inputs expanduser(None)
        report_dir = DEFAULT_REPORT_DIR
    _ensure_dir(report_dir)

    # If output_path is relative, base it on report_dir
    if not os.path.isabs(output_path):
        output_path = os.path.join(report_dir, output_path)

    # Ensure output path ends with .pdf
    if not output_path.endswith('.pdf'):
        output_path += '.pdf'

    _print(f"[INFO] Starting report generation...")
    _print(f"[INFO] Output PDF path: {output_path}")

    # 2. Generate charts (if data files provided)
    chart_paths = []
    if data_files:
        # Validate and filter data files before generating charts
        valid_data_files = []
        invalid_count = 0
        for data_file in data_files:
            is_valid, error_msg = _validate_data_file(data_file)
            if is_valid:
                valid_data_files.append(data_file)
            else:
                _print(f"[WARN] Skipping invalid data file: {error_msg}")
                invalid_count += 1

        if invalid_count > 0:
            _print(f"[WARN] Skipped {invalid_count} invalid data file(s) out of {len(data_files)} total files")

        if valid_data_files:
            chart_output_dir = os.path.join(report_dir, "charts")
            _ensure_dir(chart_output_dir)

            chart_paths = _generate_charts(
                data_files=valid_data_files,
                output_dir=chart_output_dir,
                chart_config=chart_config
            )
            _print(f"[INFO] Successfully generated {len(chart_paths)} charts from {len(valid_data_files)} valid data files")
        else:
            _print(f"[WARN] No valid data files found, skipping chart generation")

    # 3. Render HTML
    html_content = _render_html(
        llm_json=llm_json,
        chart_paths=chart_paths,
        template_path=template_path
    )
    _print(f"[INFO] Successfully rendered HTML")

    # 4. Generate PDF(If WeasyPrint is unavailable, save only HTML without error)
    pdf_path = _generate_pdf(
        html_content=html_content,
        output_path=output_path
    )

    if pdf_path:
        _print(f"[SUCCESS] Report generation completed: {pdf_path}")
    else:
        _print(f"[WARN] Report generation finished but no output file was written (PDF skipped, HTML save failed).")
