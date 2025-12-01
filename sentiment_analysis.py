import os
import pandas as pd
import glob
from typing import List, Dict, Union, Optional, Callable
from tqdm.auto import tqdm
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== CONFIGURATION ======================
DEFAULT_CONFIG = {
    "input_file": "all_extracted_triplets.csv",
    "output_dir": "./output",
    "rows_per_file": 1000,
    "date_column": "Date",
    "text_column": "Sentences",
    "model_provider": "openai",  # openai, anthropic, or perplexity
    "model_name": "gpt-4o-2024-08-06", # For OpenAI
    # "model_name": "claude-3-sonnet-20240229", # For Anthropic
    # "model_name": "sonar-small-online", # For Perplexity
    "batch_size": 100,  # Number of items to process before saving progress
    "result_column": "sentiment_classification",
    "resume_from_checkpoint": True
}

# ====================== CATEGORIES AND DEFINITIONS ======================
# Using exact variable names from original script for consistency
categories = ["1: geopolitical headwinds", "2: geopolitical tailwinds", "3: geopolitical choices", "4: no geopolitical sentiment", "5: holding strong"]

definitions = """
0: no geopolitical sentiment. The text does not contain a clear sentiment towards geopolitics. Only use this category as an absolute last resort.)\\\\
1: geopolitical headwinds. The text flags geopolitics and geopolitical factors such as war, sanctions, supply chain upheaval, and regulatory uncertainty as general headwinds that make the business environment more challenging, volatile, and uncertain. This category is about general challenges without specific actions connected to them (Example sentence: And I think that's where the closure of courts and some of the uncertainty still lingering post COVID continues to drive that uncertainty. What I would also say is that we do think that the geopolitical shifts that we're seeing occurring right in front of our eyes, as we look at today, continue to drive this focus around the uncertainty underlying these loss cost trends. So that's broadly what I would say to you.)\\\\
2: geopolitical tailwinds. The text flags geopolitics and geopolitical factors such as war, sanctions, supply chain upheaval, and regulatory uncertainty as general opportunities or tailwinds. This category is about general opportunities and benefits without specific actions connected to them (Example sentence: Today, global defense spending is on the rise, driven by the Ukraine war, shifts in geopolitical dynamics and the U.S. Department of Defense modernization priorities. Allison is poised to capture growth in this cycle by continuing our long-standing partnership with the U.S. Department of Defense.)\\\\
3: geopolitical choices. The text flags geopolitics and geopolitical factors such as war, sanctions, supply chain upheaval, and regulatory uncertainty as reasons directly imacting business strategy and investment decisions. Such decisions and strategies include, but are not limited to: supply chain diversification, de-risking, stockpiling, reshoring, nearshoring, risk monitoring, market entry/exit, complying with local content requirements, and inventory management (Example sentence: The geopolitical tensions may amplify the supply chain challenges mentioned above, which we address with our supplier risk mitigation strategy, buying key products from multiple regions and manufacturers.)\\\\
4: holding strong. The text states that geopolitics and geopolitical factors such as war, sanctions, supply chain upheaval, and regulatory uncertainty do not affect business performance or strategy, or that business performance has remained strong in spite of them (Example sentence: we delivered another quarter of strong financial results despite market concerns about slowing demand, broader macroeconomic challenges and the various global geopolitical issues. In fact, indicators of demand both from customers and in the market generally remain healthy.)\\\\
"""

# ====================== API CLIENT SETUP ======================
def setup_api_client(provider: str) -> object:
    """Set up and return the appropriate API client based on the provider."""
    if provider.lower() == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    elif provider.lower() == "anthropic":
        import anthropic
        return anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    elif provider.lower() == "perplexity":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("PERPLEXITY_API_KEY", "YOUR_PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# ====================== OPENAI FUNCTIONS ======================
def send_prompt_with_context_4(model: str, 
                             messages: List[Dict],
                             client,
                             max_tokens: int = 0) -> str:
    """
    Send prompt to OpenAI API - EXACT match to original implementation.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        seed=42,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Return raw content without stripping or processing
    return response.choices[0].message.content

def predict_sentiment_openai(review: str, model: str, client) -> str:
    """
    OpenAI sentiment prediction.
    """
    # Preserve exact format, whitespace, and indentation from original code
    system_msg = f"""
                    You are a research assistant to a social scientist. You will be provided with texts from corporate earnings calls that mention 'geopolitics'. \\
                    Classify the following text into one of the given categories: {categories}\\n{definitions} \\
                    Only include the number of the selected category in your response and no further text.
                    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": review}
    ]
    return send_prompt_with_context_4(model, messages, client)

# ====================== ANTHROPIC FUNCTIONS ======================
def send_prompt_with_context_claude(model: str, messages: list[dict], client) -> str:
    """Send prompt to Anthropic's Claude API"""
    message = client.messages.create(
        model=model,
        system=f"""
        You are a research assistant to a social scientist. You will be provided with texts from corporate earnings calls that mention 'geopolitics'.\\ 
        Classify the following text into one of the given categories: {categories}\\n{definitions}\\
        Only respond with the chosen category number and no further text.""",
        max_tokens=1500,
        temperature=0,
        messages=messages
    )
    
    # Extract the text from the response - using original handling
    if hasattr(message.content[0], 'text'):
        response_text = message.content[0].text
    else:
        response_text = str(message.content[0])
    
    # Extract first digit as in original code
    category = ''.join(filter(str.isdigit, response_text))[:1]
    return category if category else "Error: No valid category found"

def predict_sentiment_claude(review: str, model: str, client) -> str:
    """Claude sentiment prediction"""
    messages = [
        {"role": "user", "content": review}
    ]
    return send_prompt_with_context_claude(model, messages, client)

# ====================== PERPLEXITY FUNCTIONS ======================
def send_prompt_with_context_pplx(model: str, messages: List[Dict], client) -> str:
    """Send prompt to Perplexity API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=42,
            temperature=0 ## add more hyperparameters when needed
        )
        # Return raw content as in original implementation
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Perplexity API call: {e}")
        return "4"  # Default category in original code

def predict_sentiment_pplx(review: str, model: str, client) -> str:
    """Perplexity sentiment prediction"""
    system_msg = f"""
    You are a research assistant classifying corporate earnings call texts that mention geopolitics.  
    Your task is to assign exactly **one** of the following categories: {categories}.  

    {definitions}  

    ### **Instructions**  
    1. **Think step by step and reason through the classification internally.**  
    2. **Then, output only the category number (1, 2, 3, 4 or 5). No explanations.**  
    3. **Output nothing except the number.**  
    - If uncertain, follow the definitions strictly and default to 4 only as a last resort.
    """
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": review}
    ]
    
    return send_prompt_with_context_pplx(model, messages, client)

# ====================== CSV PROCESSING FUNCTIONS ======================
def split_and_sort_csv(input_file_path: str, output_dir: str, rows_per_file: int = 1000, date_column: str = "Date") -> List[str]:
    """
    Split a CSV file into smaller chunks and sort by date.
    
    Args:
        input_file_path: Path to the input CSV file
        output_dir: Directory to save the split files
        rows_per_file: Number of rows per split file
        date_column: Column to sort by (should be a date)
        
    Returns:
        List of paths to the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading CSV file: {input_file_path}")
    # Load the CSV file
    df = pd.read_csv(input_file_path)
    
    # Check if date_column exists
    if date_column in df.columns:
        # Convert the date column to datetime format (if not already)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Sort the DataFrame by the date column in ascending order
        df = df.sort_values(by=date_column)
        logger.info(f"Sorted data by {date_column}")
    else:
        logger.warning(f"Date column '{date_column}' not found. Proceeding without sorting.")
    
    # Calculate the number of files needed
    num_files = len(df) // rows_per_file + (1 if len(df) % rows_per_file else 0)
    logger.info(f"Splitting into {num_files} files with {rows_per_file} rows each")
    
    # Get the base filename without extension
    base_filename = os.path.basename(input_file_path).rsplit('.', 1)[0]
    
    split_files = []
    
    # Split and save the files
    for i in range(num_files):
        start_row = i * rows_per_file
        end_row = start_row + rows_per_file
        df_slice = df.iloc[start_row:end_row]
        
        # Generate the output file path
        output_file_path = os.path.join(output_dir, f"{base_filename}_split_{i+1}.csv")
        split_files.append(output_file_path)
        
        # Save the slice to a new CSV file
        df_slice.to_csv(output_file_path, index=False)
        logger.info(f"Saved split file {i+1}/{num_files}: {output_file_path}")
    
    return split_files

def get_predict_function(model_provider: str, model_name: str, client) -> Callable:
    """Get the appropriate prediction function based on the provider."""
    if model_provider.lower() == "openai":
        return lambda text: predict_sentiment_openai(text, model_name, client)
    elif model_provider.lower() == "anthropic":
        return lambda text: predict_sentiment_claude(text, model_name, client)
    elif model_provider.lower() == "perplexity":
        return lambda text: predict_sentiment_pplx(text, model_name, client)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

def process_csv_with_model(
    file_path: str, 
    predict_func: Callable, 
    text_column: str = "Sentences", 
    result_column: str = "sentiment_classification",
    batch_size: int = 10,
    resume: bool = True
) -> pd.DataFrame:
    """
    Process a CSV file with the selected model.
    
    Args:
        file_path: Path to the CSV file
        predict_func: Function to predict sentiment
        text_column: Column containing the text to analyze
        result_column: Column name to store the results
        batch_size: Number of rows to process before saving progress
        resume: Whether to resume from a checkpoint if available
        
    Returns:
        DataFrame with the results
    """
    # Create checkpoint path
    checkpoint_path = f"{file_path}.checkpoint.csv"
    
    # Check if checkpoint exists and we should resume
    if os.path.exists(checkpoint_path) and resume:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        df = pd.read_csv(checkpoint_path)
        # Count how many records already have results
        processed_count = df[result_column].notna().sum()
        logger.info(f"Found {processed_count} already processed records")
    else:
        logger.info(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        # If the result column doesn't exist, create it as empty
        if result_column not in df.columns:
            df[result_column] = None
        processed_count = 0
    
    # Check if text_column exists
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found in the CSV file.")
        return df
    
    total_rows = len(df)
    logger.info(f"Processing {total_rows - processed_count} remaining records in {file_path}")
    
    # Process rows that don't have results yet
    tq = tqdm(total=total_rows, initial=processed_count, desc="Processing texts")
    save_counter = 0
    
    for i in range(processed_count, total_rows):
        # Skip if already processed
        if pd.notna(df.loc[i, result_column]):
            continue
        
        text = df.loc[i, text_column]
        if pd.isna(text) or text == "":
            df.loc[i, result_column] = "4"  # Default to no sentiment if text is empty
            continue
        
        # Predict sentiment
        try:
            result = predict_func(text)
            df.loc[i, result_column] = result
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            df.loc[i, result_column] = "Error"
        
        # Update progress
        tq.update(1)
        save_counter += 1
        
        # Save checkpoint periodically
        if save_counter >= batch_size:
            df.to_csv(checkpoint_path, index=False)
            logger.info(f"Saved checkpoint after processing {i+1}/{total_rows} rows")
            save_counter = 0
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    tq.close()
    
    # Save final results
    df.to_csv(file_path, index=False)
    # Remove checkpoint file as it's no longer needed
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    logger.info(f"Completed processing: {file_path}")
    return df

def combine_csv_files(directory_path: str, pattern: str, output_file_path: str) -> None:
    """
    Combine multiple CSV files into one.
    
    Args:
        directory_path: Directory containing the CSV files
        pattern: Glob pattern to match files (e.g., '*_split_*.csv')
        output_file_path: Path to save the combined file
    """
    # Create a pattern to match all CSV files in the directory
    full_pattern = os.path.join(directory_path, pattern)
    
    # Get a list of all matching files and sort them
    csv_files = sorted(glob.glob(full_pattern))
    
    if not csv_files:
        logger.error(f"No files found matching pattern: {full_pattern}")
        return
    
    logger.info(f"Combining {len(csv_files)} CSV files")
    
    # Read and concatenate all the CSV files
    dfs = []
    for file in tqdm(csv_files, desc="Reading files"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    if not dfs:
        logger.error("No valid dataframes to combine")
        return
    
    # Concatenate all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    df_combined.to_csv(output_file_path, index=False)
    logger.info(f"Combined file saved as: {output_file_path}")

# ====================== MAIN PIPELINE ======================
def run_sentiment_analysis_pipeline(config: Dict) -> None:
    """
    Run the full sentiment analysis pipeline.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    input_file = config["input_file"]
    output_dir = config["output_dir"]
    rows_per_file = config["rows_per_file"]
    date_column = config["date_column"]
    text_column = config["text_column"]
    model_provider = config["model_provider"]
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    resume = config["resume_from_checkpoint"]
    
    # Generate result column name based on model name (replace hyphens with underscores)
    result_column = model_name.replace("-", "_")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the API client
    logger.info(f"Setting up {model_provider} client for model: {model_name}")
    client = setup_api_client(model_provider)
    
    # Get the prediction function
    predict_func = get_predict_function(model_provider, model_name, client)
    
    # Step 1: Split the input file
    logger.info("Step 1: Splitting input file")
    split_files = split_and_sort_csv(
        input_file_path=input_file,
        output_dir=output_dir,
        rows_per_file=rows_per_file,
        date_column=date_column
    )
    
    # Step 2: Process each split file
    logger.info(f"Step 2: Processing split files with model, results will be in column: {result_column}")
    for file_path in split_files:
        logger.info(f"Processing file: {file_path}")
        process_csv_with_model(
            file_path=file_path,
            predict_func=predict_func,
            text_column=text_column,
            result_column=result_column,
            batch_size=batch_size,
            resume=resume
        )
    
    # Step 3: Combine the results
    logger.info("Step 3: Combining results")
    base_filename = os.path.basename(input_file).rsplit('.', 1)[0]
    
    # Simplified output filename with "_annotated" suffix
    output_file = os.path.join(output_dir, f"{base_filename}_annotated.csv")
    
    combine_csv_files(
        directory_path=output_dir,
        pattern=f"{base_filename}_split_*.csv",
        output_file_path=output_file
    )
    
    logger.info(f"Pipeline completed. Final output: {output_file}")

# ====================== COMMAND LINE INTERFACE ======================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Geopolitical Sentiment Analysis Pipeline")
    
    parser.add_argument("--input-file", type=str, help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, help="Directory for output files")
    parser.add_argument("--rows-per-file", type=int, help="Number of rows per split file")
    parser.add_argument("--date-column", type=str, help="Column to sort by (should be a date)")
    parser.add_argument("--text-column", type=str, help="Column containing text to analyze")
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic", "perplexity"], 
                        help="LLM provider to use")
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--batch-size", type=int, help="Number of records to process before saving progress")
    parser.add_argument("--result-column", type=str, help="Column name for results")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoints")
    
    return parser.parse_args()

# ====================== ENTRY POINT ======================
if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Update config with command line args if provided
    for key, value in vars(args).items():
        if value is not None:
            # Convert --no-resume to the opposite of resume_from_checkpoint
            if key == "no_resume":
                config["resume_from_checkpoint"] = not value
            else:
                # Replace dashes with underscores in key names
                config_key = key.replace("-", "_")
                config[config_key] = value
    
    # Run the pipeline
    run_sentiment_analysis_pipeline(config)