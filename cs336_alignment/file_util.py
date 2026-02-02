from typing import Tuple
import json
import pandas as pd


def load_prompt_response_data_jsonl(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a JSONL file containing prompt and response fields.

    Args:
        file_path: Path to the JSONL file
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append((item[prompt_column], item[response_column]))
    return data


def load_prompt_response_data_parquet(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a Parquet file containing prompt and response fields.

    Args:
        file_path: Path to the Parquet file
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)
    """

    df = pd.read_parquet(file_path)
    data = [
        (str(row[prompt_column]), str(row[response_column])) for _, row in df.iterrows()
    ]
    return data


def load_prompt_response_data(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a file containing prompt and response fields.

    Supports both JSONL and Parquet file formats. The format is determined by the file extension.

    Args:
        file_path: Path to the data file (.jsonl or .parquet)
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)

    Raises:
        ValueError: If the file extension is not supported
    """
    if file_path.endswith(".jsonl"):
        return load_prompt_response_data_jsonl(
            file_path, prompt_column, response_column
        )
    elif file_path.endswith(".parquet"):
        return load_prompt_response_data_parquet(
            file_path, prompt_column, response_column
        )
    else:
        raise ValueError(
            f"Unsupported file format. File must be .jsonl or .parquet, got: {file_path}"
        )
