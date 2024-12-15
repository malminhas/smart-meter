"""
Energy Advisor - Energy Usage Analysis and Reporting Tool

Copyright (c) 2024 Mal Minhas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Security Considerations:
    - API Key: Stored in environment variable to prevent exposure
    - Input Validation: CSV file path should be validated
    - HTML Output: Content should be escaped to prevent XSS
    - File Operations: Paths should be sanitized
    - Rate Limiting: Consider adding for API calls
    - Error Handling: Sensitive information is logged safely

Dependencies:
    - pandas: Data manipulation
    - openai: GPT-4 API integration
    - groq: Groq API integration for Mixtral and Llama2 models
    - matplotlib: Graph generation
    - jinja2: HTML template rendering
    - docopt: Command line argument parsing
    - requests: HTTP client for Ollama API

Version History:
    0.5 - Current (March 19, 2024)
        - Added support for Groq LLM models (Mixtral and Llama2)
        - Improved error handling and logging
        - Updated cost calculation for all models
    0.4 - (March 19, 2024)
        - Added type hint support
        - Refactored JSON handling in prompts
        - Added personalization via context file input (-i option)
        - Improved output file handling
        - Structured JSON responses for insights and recommendations
    0.3 - (December 12, 2024)
        - Added support for multiple LLM models (GPT-4, GPT-3.5-turbo, Ollama)
        - Improved output formatting for insights and recommendations
        - Added model selection via command line
    0.2 - (December 11, 2024)
        - Switched to docopt for CLI
        - Added verbose logging option
        - Added version display
    0.1 - Initial release (December 10, 2024)
        - Basic energy consumption analysis
        - GPT-4 powered insights
        - HTML report generation
        - Graph visualization

TODO:
    - Add support for multiple data formats
    - Implement rate limiting for API calls
    - Add data export functionality
    - Add year-over-year comparison
    - Implement caching for API responses
    - Add unit tests
    - Add CI/CD pipeline
    - Add configuration file support
"""

VERSION = "0.5"
AUTHOR = "Mal Minhas with AI helpers"
RELEASE_DATE = "December 15, 2024"

import pandas as pd # type: ignore
import openai # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import webbrowser
import logging
from docopt import docopt # type: ignore
from jinja2 import Template # type: ignore
from pathlib import Path
import requests # type: ignore
import json
import html
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from groq import Groq

# Security enhancement: Validate file paths
def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and sanitize file paths to prevent directory traversal attacks.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist
        
    Returns:
        A sanitized Path object
        
    Raises:
        ValueError: If the path is invalid or suspicious
    """
    try:
        path = Path(file_path).resolve()
        if must_exist and not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        # Ensure the path doesn't contain suspicious patterns
        if any(part.startswith('.') for part in path.parts):
            raise ValueError("Suspicious file path detected")
        return path
    except Exception as e:
        logger.error(f"Invalid file path: {str(e)}")
        raise ValueError(f"Invalid file path: {str(e)}")

# Security enhancement: Sanitize HTML content
def sanitize_html_content(content: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks.
    
    Args:
        content: Raw HTML content
        
    Returns:
        Sanitized HTML content
    """
    # Allow only specific HTML tags
    allowed_tags = ['b', 'i', 'u', 'p', 'br', 'li', 'ul', 'ol']
    # Basic HTML escaping
    content = html.escape(content)
    # Re-enable allowed tags
    for tag in allowed_tags:
        content = content.replace(f'&lt;{tag}&gt;', f'<{tag}>')
        content = content.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
    return content

# HTML template for the report
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Advisor Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            background-color: #f7f8fa;
            color: #333;
        }
        header {
            background-color: #0033a0;
            padding: 20px 10px;
            text-align: center;
            width: 100%;
        }
        header h1 {
            color: #FF4D1F;
            margin: 0;
            padding: 0;
        }
        .timestamp {
            color: #FF4D1F;
            text-align: center;
            font-style: italic;
            margin: 10px 0;
        }
        .cost-info {
            color: #FF4D1F;
            text-align: center;
            font-size: 0.9em;
            margin: 5px 0;
        }
        section {
            padding: 30px;
            margin: 20px auto;
            width: 90%;
            max-width: 1400px;
            background: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #0033a0;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        ol {
            padding-left: 20px;
            width: 95%;
            margin: 0 auto;
        }
        li {
            margin-bottom: 10px;
            max-width: none;
        }
        strong {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <h1>Energy Advisor Insights and Recommendations</h1>
        <div class="timestamp">Generated on {{ timestamp }}</div>
        <div class="cost-info">Analysis cost: {{ total_cost }} (using {{ model }})</div>
    </header>
    <section>
        <h2>Key Insights</h2>
        <ol>
            {% for insight in insights_list %}
            <li>{{ insight }}</li>
            {% endfor %}
        </ol>
    </section>
    <section>
        <h2>Energy Consumption Graph</h2>
        <img src="graph.png" alt="Energy Consumption Graph">
    </section>
    <section>
        <h2>Recommendations</h2>
        <ol>
            {% for recommendation in recommendations_list %}
            <li>{{ recommendation }}</li>
            {% endfor %}
        </ol>
    </section>
</body>
</html>
"""

def generate_graph(data: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Generate energy consumption visualization graph.
    
    Args:
        data: Energy consumption data
        output_path: Path to save the generated graph
        
    Raises:
        ValueError: If data format is invalid
        IOError: If unable to save the graph
    """
    logger.info("Generating energy consumption graph")
    try:
        # Validate output path
        output_path = validate_file_path(output_path, must_exist=False)
        
        # Validate required columns
        required_columns = ['Timestamp', 'Electricity consumption (kWh)', 'Gas consumption (kWh)']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")

        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%Y')
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data['Timestamp'], data['Electricity consumption (kWh)'], 
                color='blue', marker='o', label='Electricity Consumption (kWh)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Electricity Consumption (kWh)', color='green')
        
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_xticks(data['Timestamp'])
        ax1.set_xticklabels(data['Timestamp'].dt.strftime('%b %y'), rotation=90)

        ax2 = ax1.twinx()
        ax2.plot(data['Timestamp'], data['Gas consumption (kWh)'], 
                color='green', marker='x', label='Gas Consumption (kWh)')
        ax2.set_ylabel('Gas Consumption (kWh)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title('Electricity and Gas Consumption Over Time')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Graph saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error generating graph: {str(e)}")
        raise


def generate_model_response(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Generate response using OpenAI, Groq, or Ollama models.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'llama2', 'mixtral-8x7b-32768')
        messages: List of message dictionaries
        
    Returns:
        Model response content
        
    Raises:
        Exception: If API call fails
    """
    try:
        if model.startswith('gpt-'):
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                #response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        elif model.startswith('mixtral-') or model.startswith('llama2-'):
            # Groq API handling
            client = Groq(api_key=get_groq_api_key())
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        else:
            # Ollama API endpoint
            url = f"http://localhost:11434/api/chat"
            
            # Format messages for Ollama
            data = {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json"  # Request JSON response
            }
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                content = response.json()['message']['content']
                # Try to parse and re-serialize to ensure valid JSON
                try:
                    parsed = json.loads(content)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    # If the response isn't valid JSON, try to extract JSON part
                    import re
                    json_match = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', content)
                    if json_match:
                        return json_match.group(0)
                    raise Exception("Could not extract valid JSON from response")
            else:
                raise Exception(f"Ollama API error: {response.text}")
    except Exception as e:
        logger.error(f"Error in model response: {str(e)}")
        raise

def read_user_context(context_file: Union[str, Path]) -> str:
    """
    Read user context from a text file.
    
    Args:
        context_file: Path to the context file
        
    Returns:
        User context as a string
        
    Raises:
        ValueError: If file cannot be read
    """
    try:
        path = validate_file_path(context_file, must_exist=True)
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading context file: {str(e)}")
        raise

def calculate_costs(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of LLM inference based on model and token counts.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'mixtral-8x7b-32768')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Cost in USD
    """
    costs: Dict[str, Dict[str, float]] = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
        'mixtral-8x7b-32768': {'input': 0.0007, 'output': 0.0007},  # Groq pricing
        'llama2-70b-4096': {'input': 0.0007, 'output': 0.0007}      # Groq pricing
    }
    
    if model not in costs:
        return 0.0  # No cost for local models like Ollama
        
    model_costs = costs[model]
    input_cost = (input_tokens / 1000) * model_costs['input']
    output_cost = (output_tokens / 1000) * model_costs['output']
    
    return input_cost + output_cost

def format_costs(cost: float) -> str:
    """
    Format the cost for display.
    
    Args:
        cost: Cost in USD
        
    Returns:
        Formatted cost string
    """
    if cost == 0:
        return "No cost (using local model)"
    return f"${cost:.4f}"

def generate_insights(data: pd.DataFrame, model: str, user_context: Optional[str] = None) -> Tuple[str, float]:
    """
    Generate insights using OpenAI GPT-4.
    
    Args:
        data: Energy consumption data
        model: Model name (e.g., 'gpt-4' or 'llama2')
        user_context: Additional user context for personalization
        
    Returns:
        Tuple of (HTML-formatted insights, cost of inference)
        
    Raises:
        openai.error.OpenAIError: If API call fails
        ValueError: If data format is invalid
    """
    logger.info("Generating insights using LLM")
    try:
        description = data.describe(include='all').to_string()
        context_str = f"\nAdditional context about the user and their situation: {user_context}" if user_context else ""
        prompt = f"""
        The following is a dataset description: {description}{context_str}.
        Please provide detailed insights into trends, outliers, and patterns.
        You must return a JSON array containing exactly 5 insights.
        Each insight must be a JSON object with exactly two fields:
        1. "heading": A single sentence summarizing the insight
        2. "insight": A detailed analysis of the insight
        
        Requirements:
        - No markdown or HTML formatting
        - Every number must have a unit (£ prefix for costs, kWh suffix for energy)
        - Numbers must have 2 decimal places
        - Make insights personalized if context is provided
        
        Example of required JSON format:
        {{
            "insights": [
                {{
                    "heading": "Peak Usage Occurs in Winter Months",
                    "insight": "Analysis shows highest consumption of 450.00 kWh in December"
                }},
                {{
                    "heading": "Another Insight",
                    "insight": "Another detailed analysis"
                }}
            ]
        }}
        """
        
        print(f"========== Insights Prompt ({len(prompt)} tokens) ==========\n{prompt}")
        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split()) + len(str(description).split())
        if user_context:
            input_tokens += len(user_context.split())
            
        response = generate_model_response(model, [
            {"role": "system", "content": "You are a data analyst. Return only valid JSON with exactly 5 insights."},
            {"role": "user", "content": prompt}
        ])
        
        print(f"========== Insights Response ({len(response)} tokens) ==========\n{response}")
        # Parse JSON response
        response_json = json.loads(response)
        insights_list = response_json.get('insights', [])
        
        # Estimate output tokens
        output_tokens = len(response.split())
        
        # Calculate cost
        cost = calculate_costs(model, input_tokens, output_tokens)
        logger.info(f"Insight generation cost: {format_costs(cost)}")
        
        # Convert to HTML format
        insights_html = ""
        for item in insights_list:
            insights_html += f'<b>{item["heading"]}</b>: {item["insight"]}\n\n'
            
        logger.info(f"Successfully generated insights:\n{insights_html}")
        return insights_html, cost
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise

def generate_recommendations(data: pd.DataFrame, model: str, user_context: Optional[str] = None) -> Tuple[str, float]:
    """
    Generate energy usage recommendations using OpenAI GPT-4.
    
    Args:
        data: Energy consumption data
        model: Model name (e.g., 'gpt-4' or 'llama2')
        user_context: Additional user context for personalization
        
    Returns:
        Tuple of (HTML-formatted recommendations, cost of inference)
        
    Raises:
        openai.error.OpenAIError: If API call fails
        ValueError: If data format is invalid
    """
    logger.info("Generating recommendations using GPT-4")
    try:
        context_str = f"\nAdditional context about the user and their situation: {user_context}" if user_context else ""
        prompt = f"""
        Based on the following dataset trends, provide actionable recommendations to lower electricity and gas usage.{context_str}
        You must return a JSON array containing exactly 10 recommendations.
        Each recommendation must be a JSON object with exactly two fields:
        1. "heading": A single sentence summarizing the recommendation
        2. "recommendation": Detailed actionable advice
        
        Requirements:
        - No markdown or HTML formatting
        - Every number must have a unit (£ prefix for costs, kWh suffix for energy)
        - Numbers must have 2 decimal places
        - Make recommendations personalized if context is provided
        
        Example of required JSON format:
        {{
            "recommendations": [
                {{
                    "heading": "Install LED Lighting",
                    "recommendation": "Replace all traditional bulbs with LED alternatives to save 100.00 kWh annually"
                }},
                {{
                    "heading": "Another Recommendation",
                    "recommendation": "Another detailed recommendation"
                }}
            ]
        }}
        """
        prompt += data.describe(include='all').to_string()
        print(f"========== Recommendations Prompt ({len(prompt)} tokens) ==========\n{prompt}")

        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split())
        if user_context:
            input_tokens += len(user_context.split())
            
        response = generate_model_response(model, [
            {"role": "system", "content": "You are an energy efficiency expert. Return only valid JSON with exactly 10 recommendations."},
            {"role": "user", "content": prompt}
        ])
        
        print(f"========== Recommendations Response ({len(response)} tokens) ==========\n{response}")
        # Parse JSON response
        response_json = json.loads(response)
        recommendations_list = response_json.get('recommendations', [])
        
        # Estimate output tokens
        output_tokens = len(response.split())
        
        # Calculate cost
        cost = calculate_costs(model, input_tokens, output_tokens)
        logger.info(f"Recommendation generation cost: {format_costs(cost)}")
        
        # Convert to HTML format
        recommendations_html = ""
        for item in recommendations_list:
            recommendations_html += f'<b>{item["heading"]}</b>: {item["recommendation"]}\n\n'
            
        logger.info(f"Successfully generated recommendations:\n{recommendations_html}")
        return recommendations_html, cost
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def split_into_list(text: str) -> List[str]:
    """
    Split numbered insights/recommendations into a list.
    
    Args:
        text: Text to split
        
    Returns:
        List of split text items
    """
    return [line.strip() for line in text.split("\n") if line.strip()]

def generate_report(data: pd.DataFrame, graph_path: Union[str, Path], 
                   insights: str, recommendations: str, 
                   output_path: Union[str, Path], total_cost: float = 0.0,
                   model: str = "gpt-4") -> None:
    """
    Generate HTML report with insights and recommendations.
    
    Args:
        data: Energy consumption data
        graph_path: Path to the generated graph
        insights: HTML-formatted insights
        recommendations: HTML-formatted recommendations
        output_path: Path to save the HTML report
        total_cost: Total cost of LLM inference
        model: Model used for generation
        
    Raises:
        IOError: If unable to write the report
        jinja2.TemplateError: If template rendering fails
    """
    logger.info(f"Generating HTML report to {output_path}")
    try:
        # Validate paths
        graph_path = validate_file_path(graph_path, must_exist=True)
        output_path = validate_file_path(output_path, must_exist=False)
        
        # Sanitize content
        insights = sanitize_html_content(insights)
        recommendations = sanitize_html_content(recommendations)
        
        # Get current timestamp in human readable format
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        template = Template(html_template)
        insights_list = split_into_list(insights)
        recommendations_list = split_into_list(recommendations)
        html_content = template.render(
            insights_list=insights_list,
            recommendations_list=recommendations_list,
            timestamp=timestamp,
            total_cost=format_costs(total_cost),
            model=model
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        logger.info("HTML report generated successfully")
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise

def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variables with fallback options.
    
    Returns:
        OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    # Try different common environment variable names
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set either OPENAI_API_KEY "
            "environment variable with your API key. You can get your API key from: "
            "https://platform.openai.com/api-keys"
        )
    
    return api_key

def get_groq_api_key() -> str:
    """
    Get Groq API key from environment variables.
    
    Returns:
        Groq API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Groq API key not found. Please set GROQ_API_KEY "
            "environment variable with your API key. You can get your API key from "
            "https://console.groq.com/keys"
        )
    
    return api_key

def main() -> None:
    """Main execution function for the Energy Advisor tool."""
    # Define help text separately
    help_text = """
Usage:
    energy-advisor.py [-v] [-m MODEL] [-i CONTEXT] <csv_file>
    energy-advisor.py (-h | --help)
    energy-advisor.py (-V | --version)

Options:
    -h --help           Show this help message
    -v --verbose        Enable verbose logging output
    -V --version        Show version and author information
    -m --model MODEL    Model to use for analysis [default: gpt-4]
                        Can be 'gpt-4', 'gpt-3.5-turbo', 'mixtral-8x7b-32768',
                        'llama2-70b-4096', or 'llama3.2' for local Ollama model
    -i --input CONTEXT  Path to text file containing user context for personalization

Arguments:
    csv_file            Path to CSV file containing energy data
"""
    try:
        # Parse arguments using docopt with separate help text
        arguments = docopt(help_text, version=f'Energy Advisor v{VERSION}\nAuthor: {AUTHOR}\nDate: {RELEASE_DATE}')
        
        # Configure logging based on verbose flag
        if arguments['--verbose']:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('energy_oracle.log'),
                    logging.StreamHandler()
                ]
            )
        else:
            # Suppress all logging unless verbose
            logging.basicConfig(level=logging.WARNING)
            
        global logger
        logger = logging.getLogger(__name__)
        
        logger.debug("Starting energy usage report generation")
        logger.debug(f"Arguments: {arguments}")

        # Get CSV file path and model
        csv_file = arguments['<csv_file>']
        model = arguments['--model'] or 'gpt-4'
        
        # Read user context if provided
        user_context = None
        if arguments['--input']:
            user_context = read_user_context(arguments['--input'])
            logger.info(f"Loaded user context from {arguments['--input']}")
        
        # Validate input file
        input_path = validate_file_path(csv_file)
        
        # Set API key based on model type
        if model.startswith('gpt-'):
            logger.info(f"Using OpenAI API for LLM model={model}")
            print("Using OpenAI API")
            openai.api_key = get_openai_api_key()
        elif model.startswith('mixtral-') or model.startswith('llama2-'):
            logger.info(f"Using Groq API for LLM model={model}")
            os.environ["GROQ_API_KEY"] = get_groq_api_key()
        else:
            logger.info(f"Using Ollama API for LLM model={model}")
            print("Using Ollama API")

        # Load data
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        logger.info("Data loaded successfully")

        # Generate graph
        graph_path = "graph.png"
        generate_graph(data, graph_path)

        # Generate insights and track cost
        insights, insights_cost = generate_insights(data, model, user_context)

        # Generate recommendations and track cost
        recommendations, recommendations_cost = generate_recommendations(data, model, user_context)

        # Calculate total cost
        total_cost = insights_cost + recommendations_cost

        # Generate HTML report with cost information
        report_path = "report.html"
        generate_report(data, graph_path, insights, recommendations, report_path, total_cost, model)

        # Open the report in the browser
        logger.info("Opening report in web browser")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        logger.info(f"Report generation completed successfully. Total cost: {format_costs(total_cost)}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
