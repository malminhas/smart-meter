"""
Smart Meter Advisor - Smart Meter Analysis and Reporting Tool

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
    - anthropic: Claude API integration for Sonnet models
    - matplotlib: Graph generation
    - jinja2: HTML template rendering
    - docopt: Command line argument parsing
    - requests: HTTP client for Ollama API
    - arrow: Date and time parsing and formatting
    - typing: Type hints for better code readability
    - BeautifulSoup: HTML parsing

Version History:
    0.1 - Initial release (December 20, 2024)
        - End to end smart meter data analysis and reporting tool
        - LLM powered insights and recommendations
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

VERSION = "0.1"
AUTHOR = "Mal Minhas with AI helpers"
RELEASE_DATE = "December 20, 2024"

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
import pandas as pd # type: ignore
import arrow # type: ignore
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from groq import Groq # type: ignore
from anthropic import Anthropic # type: ignore
try: 
    from BeautifulSoup import BeautifulSoup # type: ignore
except ImportError:
    from bs4 import BeautifulSoup # type: ignore

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
    <title>Smart Meter Advisor Report</title>
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
        <h1>Smart Meter Advisor Insights and Recommendations</h1>
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

# Add this HTML template near the top with the other templates
smart_meter_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Meter Report</title>
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
            font-size: 24px;
        }
        section {
            padding: 30px;
            margin: 20px auto;
            width: 90%;
            max-width: 1400px;
            background: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h2, h3.section-heading {
            color: #0033a0;
            font-size: 20px;
            font-weight: bold;
            border-bottom: 2px solid #0033a0;
            padding-bottom: 10px;
            margin-top: 30px;
            font-family: Arial, sans-serif;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            margin: 10px 0;
            border: 1px solid #e9ecef;
            line-height: 1.6;
        }
        .device-info {
            margin-left: 20px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Smart Meter Status Report</h1>
    </header>
    <section>
        {{ content }}
    </section>
</body>
</html>
"""

def generate_graph(data: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Generate electricity consumption visualization graph.
    
    Args:
        data: Energy consumption data with 'Timestamp' and consumption columns
        output_path: Path to save the generated graph
        
    Raises:
        ValueError: If data format is invalid
        IOError: If unable to save the graph
    """
    logger.info("Generating electricity consumption graph")
    try:
        # Validate output path
        output_path = validate_file_path(output_path, must_exist=False)
        
        # Validate required columns
        required_columns = ['Timestamp', 'Electricity Consumption (kWh)', 'Electricity cost (£)']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")

        # Create a copy of the data
        monthly_data = data.copy()
        
        # Convert MM/YYYY string to datetime
        monthly_data['Timestamp'] = pd.to_datetime(monthly_data['Timestamp'].apply(lambda x: f"01/{x}"), format='%d/%m/%Y')
        
        # Create the figure and axis with extra top margin
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot electricity consumption
        line = ax.plot(monthly_data['Timestamp'], monthly_data['Electricity Consumption (kWh)'], 
                color='green',
                marker='o',
                label='Monthly Electricity Consumption (kWh)')
        
        # Add extra headroom above highest point and below lowest point
        max_value = monthly_data['Electricity Consumption (kWh)'].max()
        min_value = monthly_data['Electricity Consumption (kWh)'].min()
        delta = (max_value - min_value)/10
        ax.set_ylim(min_value - delta, max_value + delta)  # 10% padding on both ends
        
        # Add cost annotations with dynamic positioning
        values = monthly_data['Electricity Consumption (kWh)'].tolist()
        timestamps = monthly_data['Timestamp'].tolist()
        costs = monthly_data['Electricity cost (£)'].tolist()
        
        for i in range(len(values)):
            # Get current point value
            y = values[i]
            x = timestamps[i]
            cost = costs[i]
            
            # Determine if point is a peak or valley
            is_peak = False
            if i > 0 and i < len(values) - 1:  # Skip first and last points
                prev_val = values[i-1]
                next_val = values[i+1]
                is_peak = y > prev_val and y > next_val
            
            # Position annotation above for peaks, below for valleys
            y_offset = 5 if is_peak else -10
            
            ax.annotate(f'£{cost}',  # Text to display
                       (x, y),       # Point to annotate
                       textcoords="offset points",  # How to position the text
                       xytext=(5, y_offset),  # Distance from point
                       ha='left',      # Left alignment for better spacing
                       fontsize=10,    # Font size
                       color='green')  # Color
        
        # Customize the plot
        ax.set_xlabel('Month')
        ax.set_ylabel('Electricity Consumption (kWh)', color='green')
        ax.tick_params(axis='y', labelcolor='#0033a0')
        
        # Format x-axis
        ax.set_xticks(monthly_data['Timestamp'])
        ax.set_xticklabels(monthly_data['Timestamp'].dt.strftime('%m/%Y'), rotation=45)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        plt.title('Electricity Consumption Over Time')
        
        # Add extra margin at the top
        plt.subplots_adjust(top=0.85)
        
        # Adjust layout to prevent label cutoff while maintaining top margin
        plt.tight_layout()
        
        # Save the graph
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Graph saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error generating graph: {str(e)}")
        raise

def generate_model_response(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Generate response using OpenAI, Groq, Anthropic, or Ollama models.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'llama2', 'mixtral-8x7b-32768', 'claude-3-sonnet')
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
        elif model.startswith('claude-'):
            # Anthropic API handling
            client = Anthropic(api_key=get_anthropic_api_key())
            # Convert messages to Anthropic format
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
            
            prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                system="Return response in JSON format."
            )
            return response.content[0].text
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
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'mixtral-8x7b-32768', 'claude-3-sonnet')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Cost in USD
    """
    # Define pricing for different models.  Costs are per 1000 tokens
    # OpenAI model pricing: https://openai.com/api/pricing/ 
    # 1. gpt-4: input=$30.00/1M tokens, output=$60.00/1M tokens
    # 2. gpt-3.5-turbo: input=$1.50/1M tokens, output=$2.00/1M tokens
    # Groq pricing: https://groq.com/pricing/
    # 3. mixtral-8x7b-32768: input=	$0.24/1M tokens, output=$0.24/1M tokens
    # 4. llama2-70b-4096: input=$0.59/1M tokens, output=$0.79/1M tokens
    # Anthropic pricing: https://www.anthropic.com/pricing#anthropic-api
    # 5. claude-3-5-sonnet-20241022: input=$3.75/1M tokens, output=$15/1M tokens
    costs: Dict[str, Dict[str, float]] = {
        'gpt-4': {'input': 0.03, 'output': 0.06},                        # OpenAI
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},             # OpenAI
        'mixtral-8x7b-32768': {'input': 0.00024, 'output': 0.00024},     # Groq
        'llama2-70b-4096': {'input': 0.00059, 'output': 0.00079},        # Groq
        'claude-3-5-sonnet-20241022': {'input': 0.0375, 'output': 0.015} # Anthropic pricing
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

def generate_insights(data: pd.DataFrame, model: str, user_context: Optional[str] = None, 
                      dump_insights: bool = False) -> Tuple[str, float]:
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
        - Every number must have a unit (use a £ prefix with all costs, kWh suffix with all energy)
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
                    "insight": "Another detailed analysis with total cost of £100.01"
                }}
            ]
        }}
        """
        
        if dump_insights:
            print(f"========== Insights Prompt ({len(prompt)} tokens) ==========\n{prompt}")
        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split()) + len(str(description).split())
        if user_context:
            input_tokens += len(user_context.split())
            
        response = generate_model_response(model, [
            {"role": "system", "content": "You are a data analyst. Return only valid JSON with exactly 5 insights."},
            {"role": "user", "content": prompt}
        ])
        
        if dump_insights:
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

def generate_recommendations(data: pd.DataFrame, model: str, user_context: Optional[str] = None, 
                             dump_recommendations: bool = False) -> Tuple[str, float]:
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
        - Every number must have a unit (use a £ prefix with all costs, kWh suffix with all energy)
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
                    "insight": "Another detailed analysis with total cost of £100.01"
                }}
            ]
        }}
        """
        prompt += data.describe(include='all').to_string()
        if dump_recommendations:
            print(f"========== Recommendations Prompt ({len(prompt)} tokens) ==========\n{prompt}")

        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split())
        if user_context:
            input_tokens += len(user_context.split())
            
        response = generate_model_response(model, [
            {"role": "system", "content": "You are an energy efficiency expert. Return only valid JSON with exactly 10 recommendations."},
            {"role": "user", "content": prompt}
        ])
        
        if dump_recommendations:
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
        text: Text containing numbered items to split
        
    Returns:
        List of split text items with numbering removed
        
    Example:
        Input: "1. First item\n2. Second item"
        Output: ["First item", "Second item"]
    """
    return [item.strip() for item in text.split('\n') if item.strip()]

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

def get_anthropic_api_key() -> str:
    """
    Get Anthropic API key from environment variables.
    
    Returns:
        Anthropic API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Please set ANTHROPIC_API_KEY "
            "environment variable with your API key. You can get your API key from "
            "https://console.anthropic.com/settings/keys"
        )
    
    return api_key

def get_mpan() -> str:
    """
    Get MPAN from environment variables.
    
    Returns:
        MPAN
    """
    return os.getenv("MPAN")

def get_ihd() -> str:
    """
    Get IHD from environment variables.
    
    Returns:
        IHD
    """
    return os.getenv("IHD_MAC")

def get_house_number() -> str:
    """
    Get house number from environment variables.
    
    Returns:
        House number
    """
    return os.getenv("HOUSE_NUMBER")

def get_postcode() -> str:
    """
    Get postcode from environment variables.
    
    Returns:
        Postcode
    """
    return os.getenv("POSTCODE")

def checkMeter(mpan: str, house: str, postcode: str, confirm: str = 'on') -> str:
    """
    Check meter status and save raw HTML response from n3rgy API.
    
    Args:
        mpan: Meter Point Administration Number
        house: House number or name
        postcode: UK postal code
        confirm: Confirmation parameter for API (default: 'on')
        
    Returns:
        Raw HTML response from the meter check
        
    Raises:
        AssertionError: If API response is not 200
        IOError: If unable to save response to file
    """
    logger.info(f"Checking meter for house {house} in postcode {postcode}")
    url = f"https://homebrew.n3rgy.com/cgi-bin/n3rgy-checkmeter.py?house={house}&postcode={postcode}&confirm={confirm}"
    headers = {'Authorization': mpan}
    r = requests.get(url=url, headers=headers)
    assert(r.status_code == 200)
    
    # Save raw HTML response
    with open('raw_meter_check.html', 'w') as f:
        f.write(r.text)
    logger.info("Raw meter check response saved to raw_meter_check.html")
    
    return r.text

def generateSmartMeterReport(html: str) -> str:
    """
    Generate formatted HTML report from meter check response.
    
    Args:
        html: Raw HTML response from meter check
        
    Returns:
        Formatted HTML content for the report
        
    Raises:
        BeautifulSoup.ParserError: If HTML parsing fails
        IOError: If unable to save report
        jinja2.TemplateError: If template rendering fails
    """
    logger.info("Generating smart meter report")
    parsed_html = BeautifulSoup(html, features="html.parser")
    title = parsed_html.find_all("body")[1].find("h2").string
    raw_headings = parsed_html.find_all("body")[1].find_all("pre")[0].find_all("b")
    headings = [heading.string for heading in raw_headings]
    values = str(parsed_html.find_all("body")[1].find_all("pre")[0]).split('\n')
    
    # Generate formatted content with reduced spacing
    content = f"<h2>{title}</h2>\n"
    i = 0
    for v in values:
        if v not in ['<pre>','</pre>']:
            if v.find('<b>') == 0:
                if i > 0:  # Close previous pre if it exists
                    content += "</pre>\n"
                content += f"<h3 class='section-heading' style='margin-bottom: 0.5em; margin-top: 1em'>{headings[i]}</h3>\n<pre class='device-info' style='margin-top: 0'>"
                i += 1
            else:
                if len(v.strip()) > 0 and 'href' not in v:
                    content += f"{v}\n" 
        if v == '</pre>':
            content += "</pre>\n"
    
    print(content)
    # Render template
    template = Template(smart_meter_template)
    html_content = template.render(content=content)
    
    # Save formatted report
    report_path = "smart_meter_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Smart meter report saved to {report_path}")
    
    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(report_path)}")
    
    return html_content

def getDate(date_str: str = '', dayOffset: int = 0, monthOffset: int = 0) -> str:
    """
    Get formatted date with optional offsets.
    
    Args:
        date_str: Starting date in DD.MM.YYYY format (default: today)
        dayOffset: Number of days to shift (default: 0)
        monthOffset: Number of months to shift (default: 0)
        
    Returns:
        Date string in DD.MM.YYYY format
        
    Example:
        >>> getDate('01.01.2024', dayOffset=1, monthOffset=2)
        '02.03.2024'
    """
    logger.info(f"Getting date for {date_str} with dayOffset={dayOffset} and monthOffset={monthOffset}")
    start = arrow.now().floor('day')
    if date_str: 
        start = arrow.get(date_str, "DD.MM.YYYY")
    return start.shift(days=dayOffset, months=monthOffset).format("DD.MM.YYYY")

def isDateLessThanToday(date_str: str) -> bool:
    """
    Check if given date is earlier than today.
    
    Args:
        date_str: Date to check in DD.MM.YYYY format
        
    Returns:
        True if date is earlier than today, False otherwise
        
    Example:
        >>> isDateLessThanToday('01.01.2024')
        True  # Assuming today is after Jan 1, 2024
    """
    logger.info(f"Checking if date {date_str} is earlier than today's date")
    given_date = arrow.get(date_str, "DD.MM.YYYY")
    today = arrow.now().floor('day')
    return given_date < today

def isDateLessThanEnd(date_str: str, end: str) -> bool:
    """
    Check if first date is earlier than second date.
    
    Args:
        date_str: First date in DD.MM.YYYY format
        end: Second date in DD.MM.YYYY format
        
    Returns:
        True if first date is earlier than second date, False otherwise
        
    Example:
        >>> isDateLessThanEnd('01.01.2024', '02.01.2024')
        True
    """
    logger.info(f"Checking if date {date_str} is earlier than end date {end}")
    given_date = arrow.get(date_str, "DD.MM.YYYY")
    end_date = arrow.get(end, "DD.MM.YYYY")
    return given_date < end_date

def getAvailableRange(start: str, end: str, ihd_mac: str, 
                     source: str = 'electricity', 
                     category: str = 'consumption', 
                     fmt: str = 'json') -> Tuple[str, str]:
    """
    Get available date range for smart meter data from n3rgy API.
    
    Args:
        start: Start date in DD.MM.YYYY format
        end: End date in DD.MM.YYYY format
        ihd_mac: In-Home Display MAC address
        source: Data source (default: 'electricity')
        category: Data category (default: 'consumption')
        fmt: Response format (default: 'json')
        
    Returns:
        Tuple of (start_date, end_date) in DD.MM.YYYY format
        
    Raises:
        AssertionError: If API response is not 200 or 206
    """
    logger.info(f"Getting available range for {start} to {end} from {source}:{category}:{fmt}")
    reformatDate = lambda s: ''.join(s.split('.')[::-1])
    start = reformatDate(start)
    end = reformatDate(end)
    url = f"https://consumer-api.data.n3rgy.com/{source}/{category}/1?start={start}&end={end}&output={format}"
    headers = {'Authorization': ihd_mac}
    r = requests.get(url=url, headers=headers)
    if r.status_code == 206:
        logger.warning("Data is not available for the whole requested range.")
    assert(r.status_code in [200, 206])
    logger.info(f"{r.json()}")
    d = r.json().get('availableCacheRange')
    start = d.get('start')
    start = f"{start[6:8]}.{start[4:6]}.{start[:4]}"
    end = d.get('end')
    end = f"{end[6:8]}.{end[4:6]}.{end[:4]}"
    return start, end

def getDataInRange(start: str, end: str, ihd_mac: str,
                  source: str = 'electricity',
                  category: str = 'consumption',
                  fmt: str = 'json') -> Dict:
    """
    Get smart meter data for specified date range from n3rgy API.
    
    Args:
        start: Start date in DD.MM.YYYY format
        end: End date in DD.MM.YYYY format
        ihd_mac: In-Home Display MAC address
        source: Data source (default: 'electricity')
        category: Data category (default: 'consumption')
        fmt: Response format (default: 'json')
        
    Returns:
        Dictionary containing smart meter data
        
    Raises:
        AssertionError: If API response is not 200
    """
    logger.info(f"Getting data in range {start} to {end} from {source} {category} {fmt}")
    reformatDate = lambda s: ''.join(s.split('.')[::-1])
    start = reformatDate(start)
    end = reformatDate(end)
    url = f"https://consumer-api.data.n3rgy.com/{source}/{category}/1?start={start}&end={end}&output={format}"
    headers = {'Authorization': ihd_mac}
    r = requests.get(url=url, headers=headers)
    assert(r.status_code == 200)
    return r.json()

def generateDateBlocks(start: str, end: str, moffset: int = 3) -> Tuple[List[str], List[str]]:
    """
    Generate list of date ranges with specified month offset.
    
    Args:
        start: Start date in DD.MM.YYYY format
        end: End date in DD.MM.YYYY format
        moffset: Month offset for each block (default: 3)
        
    Returns:
        Tuple of (start_dates, end_dates) where each is a list of dates in DD.MM.YYYY format
        
    Example:
        >>> starts, ends = generateDateBlocks('01.01.2024', '01.07.2024', moffset=3)
        >>> starts
        ['01.01.2024', '01.04.2024']
        >>> ends
        ['31.03.2024', '01.07.2024']
    """
    logger.info(f"Generating date blocks from {start} to {end} with month offset={moffset}")
    starts = []
    ends = []
    offset = start
    while isDateLessThanEnd(offset, end):
        starts.append(offset)
        noffset = getDate(offset, dayOffset=-1, monthOffset=moffset)
        if isDateLessThanToday(noffset):    
            offset = getDate(noffset, dayOffset=1)
        else:
            noffset = end
            offset = end
        ends.append(noffset)
    return starts, ends

def main() -> None:
    """Main execution function for the Smart Meter Advisor tool."""
    # Define help text separately
    help_text = """
Usage:
    smart-meter-advisor.py [-v] [-m MODEL] [-i CONTEXT] --command CMD
    smart-meter-advisor.py (-h | --help)
    smart-meter-advisor.py (-V | --version)

Commands:
    --command CMD       Command to execute
                        - dump-meter
                        - get-smart-meter-data
                        - generate-report

Options:
    -h --help           Show this help message
    -v --verbose        Enable verbose logging output
    -V --version        Show version and author information
    -m --model MODEL    Model to use for analysis [default: gpt-4]
                        - OpenAI: 'gpt-4', 'gpt-3.5-turbo'
                        - Groq: 'mixtral', 'llama2'
                        - Claude: 'claude'
                        - Ollama: 'llama3.2'
    -i --input CONTEXT  Path to text file containing user context for personalization

"""
    try:
        # Parse arguments using docopt with separate help text
        arguments = docopt(help_text, version=f'Smart Meter Advisor v{VERSION}\nAuthor: {AUTHOR}\nDate: {RELEASE_DATE}')
        
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

        # Get MPAN and IHD values
        mpan = get_mpan()
        ihd_mac = get_ihd()
        house_number = get_house_number()
        postcode = get_postcode()
        
        if arguments['--command'] == 'dump-meter':
            logger.debug(f"MPAN: {mpan}, IHD: {ihd_mac}, House Number: {house_number}, Postcode: {postcode}")
            assert(mpan and ihd_mac and house_number and postcode)
            html = checkMeter(mpan, house_number, postcode, confirm='on')
            generateSmartMeterReport(html)  # This will now save and open the formatted report

        elif arguments['--command'] == 'get-smart-meter-data':
            today = getDate()
            yesterday = getDate(dayOffset=-1)
            threeMonthsAgo = getDate(monthOffset=-3)
            aYearAgo = getDate(monthOffset=-12)
            assert(isDateLessThanEnd(aYearAgo, today))
            assert(isDateLessThanEnd(threeMonthsAgo, today))
        
            start, end = getAvailableRange(threeMonthsAgo, today, ihd_mac)
            logger.info(f"Today is {today} and a year ago was {aYearAgo}.")
            logger.info(f"We can get energy data from start={start} to end={end}")
        
            starts,ends = generateDateBlocks(start,end,moffset=3)
            logger.info(f"Start dates: {starts}")
            logger.info(f"End dates  : {ends}")
            
            results = []
            for s,e in zip(starts, ends):
                records = getDataInRange(s, e, ihd_mac, source='electricity')
                logger.info(f"{len(records.get('values'))} records found between {s} and {e}")
                results.append(records)

            d = results[0]
            logger.info(f"keys = {list(d.keys())}")
            logger.info(f"resource = {d.get('resource')}")
            logger.info(f"responseTimestamp = {d.get('responseTimestamp')}")
            logger.info(f"start = {d.get('start')}")
            logger.info(f"end = {d.get('end')}")
            logger.info(f"granularity = {d.get('granularity')}")
            logger.info(f"{len(d.get('values'))} values")
            logger.info(f"availableCacheRange = {d.get('availableCacheRange')}")
            logger.info(f"unit = {d.get('unit')}")

            standing_charge = 0.71  # Daily standing charge in £
            kWh_unit_charge = 0.259  # Average cost per kWh in £
            
            # Collect all values
            all_values = []
            for d in results:
                all_values += d.get('values')
                
            # Create DataFrame
            df = pd.DataFrame(all_values)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by day and calculate daily sums
            daily_df = df.groupby(df['timestamp'].dt.date).agg({
                'value': 'sum'
            }).reset_index()
            
            # Rename columns for daily data
            daily_df = daily_df.rename(columns={
                'timestamp': 'Timestamp',
                'value': 'Electricity Consumption (kWh)'
            })
            
            # Round electricity consumption to 3 decimal places
            daily_df['Electricity Consumption (kWh)'] = daily_df['Electricity Consumption (kWh)'].round(3)
            
            # Calculate daily cost: (consumption * unit rate) + standing charge
            daily_df['Electricity cost (£)'] = (daily_df['Electricity Consumption (kWh)'] * kWh_unit_charge) + standing_charge
            
            # Format the cost to 2 decimal places
            daily_df['Electricity cost (£)'] = daily_df['Electricity cost (£)'].round(2)
            
            # Convert Timestamp back to datetime (as groupby converts it to date)
            daily_df['Timestamp'] = pd.to_datetime(daily_df['Timestamp'])
            
            # Format the date to DD/MM/YYYY for daily data
            daily_df['Timestamp'] = daily_df['Timestamp'].dt.strftime('%d/%m/%Y')
            
            # Save daily data to CSV
            daily_df.to_csv('smart_meter_data_daily.csv', index=False)
            logger.info(f"Daily smart meter data saved to smart_meter_data_daily.csv")
            
            # Create monthly aggregation
            # First, convert back to datetime for grouping
            monthly_df = pd.DataFrame(daily_df)
            monthly_df['Timestamp'] = pd.to_datetime(monthly_df['Timestamp'], format='%d/%m/%Y')
            
            # Group by month and calculate monthly sums
            monthly_df = monthly_df.groupby(monthly_df['Timestamp'].dt.to_period('M')).agg({
                'Electricity Consumption (kWh)': 'sum',
                'Electricity cost (£)': 'sum'
            }).reset_index()
            
            # Convert Period to datetime
            monthly_df['Timestamp'] = monthly_df['Timestamp'].dt.to_timestamp()
            
            # Format the date to MM/YYYY for monthly data
            monthly_df['Timestamp'] = monthly_df['Timestamp'].dt.strftime('%m/%Y')
            
            # Round values
            monthly_df['Electricity Consumption (kWh)'] = monthly_df['Electricity Consumption (kWh)'].round(3)
            monthly_df['Electricity cost (£)'] = monthly_df['Electricity cost (£)'].round(2)
            
            # Save monthly data to CSV
            monthly_df.to_csv('smart_meter_data_monthly.csv', index=False)
            logger.info(f"Monthly smart meter data saved to smart_meter_data_monthly.csv")

        elif arguments['--command'] == 'generate-report':
            # Get model
            model = arguments['--model'] or 'gpt-4'
            
            # Read user context if provided
            user_context = None
            if arguments['--input']:
                user_context = read_user_context(arguments['--input'])
                logger.info(f"Loaded user context from {arguments['--input']}")
            
            # Validate input file
            csv_file = 'smart_meter_data_monthly.csv'
            logger.info(f"Using monthly data from {csv_file}")
            input_path = validate_file_path(csv_file)
            
            # Set API key based on model type
            if model.startswith('gpt-'):
                using_model = 'gpt-4'
                str_model = f"Using OpenAI API for LLM model '{model}'='{using_model}'"
                print(str_model)
                logger.info(str_model)
                openai.api_key = get_openai_api_key()
            elif model.startswith('mixtral'):
                using_model = 'mixtral-8x7b-32768'
                str_model = f"Using Groq API for LLM model '{model}'='{using_model}'"
                print(str_model)
                logger.info(str_model)
                model = using_model
                os.environ["GROQ_API_KEY"] = get_groq_api_key()
            elif model.startswith('llama2'):
                using_model = 'llama2-70b-4096'
                str_model = f"Using Groq API for LLM model '{model}'='{using_model}'"
                print(str_model)
                logger.info(str_model)
                model = using_model
                os.environ["GROQ_API_KEY"] = get_groq_api_key()
            elif model.startswith('claude'):
                using_model = 'claude-3-5-sonnet-20241022'
                str_model = f"Using Anthropic API for LLM model '{model}'='{using_model}'"
                print(str_model)
                logger.info(str_model)
                model = using_model
                os.environ["ANTHROPIC_API_KEY"] = get_anthropic_api_key()
            else:
                logger.info(f"Using Ollama API for LLM model='{model}'")
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
            report_path = "smart_meter_usage_report.html"
            generate_report(data, graph_path, insights, recommendations, report_path, total_cost, model)

            # Open the report in the browser
            logger.info("Opening report in web browser")
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            logger.info(f"Report generation completed successfully. Total cost: {format_costs(total_cost)}")
        else:
            logger.error(f"Invalid command: {arguments['--command']}")
            raise ValueError(f"Invalid command: {arguments['--command']}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
