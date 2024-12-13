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
    - matplotlib: Graph generation
    - jinja2: HTML template rendering
    - docopt: Command line argument parsing

Version History:
    0.3 - Current (December 13, 2024)
        - Restructured prompt for insights and recommendations
        - Added support for multiple LLM models (GPT-4, GPT-3.5-turbo, Ollama)
        - Improved output formatting for insights and recommendations
        - Added model selection via command line
    0.2 - Previous (December 12, 2024)
        - Switched to docopt for CLI
        - Added verbose logging option
        - Added version display
    0.1 - Initial release (December 11, 2024)
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

VERSION = "0.3"
AUTHOR = "Mal Minhas with AI helpers"

import pandas as pd
import openai
import matplotlib.pyplot as plt
import os
import webbrowser
import logging
from docopt import docopt
from jinja2 import Template
from pathlib import Path
import html
import sys
from datetime import datetime
import requests
import json

# Security enhancement: Validate file paths
def validate_file_path(file_path):
    """
    Validate and sanitize file paths to prevent directory traversal attacks.
    
    Args:
        file_path (str): The file path to validate
        
    Returns:
        Path: A sanitized Path object
        
    Raises:
        ValueError: If the path is invalid or suspicious
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        # Ensure the path doesn't contain suspicious patterns
        if any(part.startswith('.') for part in path.parts):
            raise ValueError("Suspicious file path detected")
        return path
    except Exception as e:
        logger.error(f"Invalid file path: {str(e)}")
        raise ValueError(f"Invalid file path: {str(e)}")

# Security enhancement: Sanitize HTML content
def sanitize_html_content(content):
    """
    Sanitize HTML content to prevent XSS attacks.
    
    Args:
        content (str): Raw HTML content
        
    Returns:
        str: Sanitized HTML content
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
            color: #FF4D1F;  /* EDF Orange */
            margin: 0;
            padding: 0;
        }
        .timestamp {
            color: #FF4D1F;
            text-align: center;
            font-style: italic;
            margin: 10px 0;
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

def generate_graph(data, output_path):
    """
    Generate energy consumption visualization graph.
    
    Args:
        data (pandas.DataFrame): Energy consumption data
        output_path (str): Path to save the generated graph
        
    Raises:
        ValueError: If data format is invalid
        IOError: If unable to save the graph
    """
    logger.info("Generating energy consumption graph")
    try:
        # Validate output path
        output_path = validate_file_path(output_path)
        
        # Validate required columns
        required_columns = ['Timestamp', 'Electricity consumption (kWh)', 'Gas consumption (kWh)']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")

        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%Y')
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data['Timestamp'], data['Electricity consumption (kWh)'], color='blue', marker='o', label='Electricity Consumption (kWh)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Electricity Consumption (kWh)', color='green')
        
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_xticks(data['Timestamp'])
        ax1.set_xticklabels(data['Timestamp'].dt.strftime('%b %y'), rotation=90)

        ax2 = ax1.twinx()
        ax2.plot(data['Timestamp'], data['Gas consumption (kWh)'], color='green', marker='x', label='Gas Consumption (kWh)')
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

def generate_model_response(model, messages):
    """
    Generate response using either OpenAI or Ollama models.
    
    Args:
        model (str): Model name (e.g., 'gpt-4' or 'llama2')
        messages (list): List of message dictionaries
        
    Returns:
        str: Model response content
        
    Raises:
        Exception: If API call fails
    """
    if model.startswith('gpt-'):
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    else:
        # Ollama API endpoint
        url = f"http://localhost:11434/api/chat"
        
        # Format messages for Ollama
        data = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            raise Exception(f"Ollama API error: {response.text}")

def generate_insights(data, model):
    """
    Generate insights using OpenAI GPT-4.
    
    Args:
        data (pandas.DataFrame): Energy consumption data
        model (str): Model name (e.g., 'gpt-4' or 'llama2')
        
    Returns:
        str: HTML-formatted insights
        
    Raises:
        openai.error.OpenAIError: If API call fails
        ValueError: If data format is invalid
    """
    logger.info("Generating insights using GPT-4")
    try:
        description = data.describe(include='all').to_dict()
        prompt = f"""
        The following is a dataset description: {description}
        Please provide detailed insights into trends, outliers, and patterns.
        Return the response as a JSON list of dictionaries.
        Each dictionary should have two keys: "heading" and "insight".
        The heading should be a single sentence that summarizes the insight.
        The insight should provide a paragraph of detailed analysis.
        No markdown or HTML formatting should be used.
        Ensure that every number is supplied with the right unit.
        Costs should have a £ sign preceding the number.
        kWh should have a kWh suffix.
        Numbers should be formatted as a number with 2 decimal places.
        Generate at least 5 insights.
        
        Example format:
        [
            {{"heading": "Peak Usage Occurs in Winter Months", "insight": "Analysis shows highest consumption of 450.00 kWh in December"}},
            {{"heading": "Another Insight", "insight": "Another detailed analysis"}}
        ]
        """
        response = generate_model_response(model, [
            {"role": "system", "content": "You are a data analyst providing a set of insights derived from the dataset. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ])
        
        # Parse JSON response
        insights_list = json.loads(response)
        
        # Convert to HTML format
        insights_html = ""
        for item in insights_list:
            insights_html += f'<b>{item["heading"]}</b>: {item["insight"]}\n\n'
       
        print(f"Insights:\n{insights_html}")
        logger.info(f"Successfully generated insights:\n{insights_html}")
        return insights_html
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise

def generate_recommendations(data, model):
    """
    Generate energy usage recommendations using OpenAI GPT-4.
    
    Args:
        data (pandas.DataFrame): Energy consumption data
        model (str): Model name (e.g., 'gpt-4' or 'llama2')
        
    Returns:
        str: HTML-formatted recommendations
        
    Raises:
        openai.error.OpenAIError: If API call fails
        ValueError: If data format is invalid
    """
    logger.info("Generating recommendations using GPT-4")
    try:
        prompt = """
        Based on the following dataset trends, provide actionable recommendations to lower electricity and gas usage.
        Return the response as a JSON list of dictionaries.
        Each dictionary should have two keys: "heading" and "recommendation".
        The heading should be a single sentence that summarizes the recommendation.
        The recommendation should provide a paragraph of rationale and detailed actionable advice.
        A specific call to action should be included at the end of each recommendation.
        No markdown or HTML formatting should be used.
        Ensure that every number is supplied with the right unit.
        Costs should have a £ sign preceding the number.
        kWh should have a kWh suffix.
        Numbers should be formatted as a number with 2 decimal places.
        Generate at least 10 recommendations.
        
        Example format:
        [
            {{"heading": "Install LED Lighting", "recommendation": "Replace all traditional bulbs with LED alternatives to save 100.00 kWh annually"}},
            {{"heading": "Another Recommendation", "recommendation": "Another detailed recommendation"}}
        ]
        """
        prompt += data.describe(include='all').to_string()
        response = generate_model_response(model, [
            {"role": "system", "content": "You are an energy efficiency expert providing recommendations. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ])
        
        # Parse JSON response
        recommendations_list = json.loads(response)
        
        # Convert to HTML format
        recommendations_html = ""
        for item in recommendations_list:
            recommendations_html += f'<b>{item["heading"]}</b>: {item["recommendation"]}\n\n'
            
        logger.info(f"Successfully generated recommendations:\n{recommendations_html}")
        print(f"Recommendations:\n{recommendations_html}")
        return recommendations_html
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def split_into_list(text):
    """Split numbered insights/recommendations into a list."""
    return [line.strip() for line in text.split("\n") if line.strip()]

def generate_report(data, graph_path, insights, recommendations, output_path):
    """
    Generate HTML report with insights and recommendations.
    
    Args:
        data (pandas.DataFrame): Energy consumption data
        graph_path (str): Path to the generated graph
        insights (str): HTML-formatted insights
        recommendations (str): HTML-formatted recommendations
        output_path (str): Path to save the HTML report
        
    Raises:
        IOError: If unable to write the report
        jinja2.TemplateError: If template rendering fails
    """
    logger.info(f"Generating HTML report to {output_path}")
    try:
        # Validate paths
        graph_path = validate_file_path(graph_path)
        output_path = validate_file_path(output_path)
        
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
            timestamp=timestamp
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        logger.info("HTML report generated successfully")
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise

def get_api_key():
    """
    Get OpenAI API key from environment variables with fallback options.
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    # Try different common environment variable names
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set either OPENAI_API_KEY or OPEN_API_KEY "
            "environment variable with your API key. You can get your API key from "
            "https://platform.openai.com/api-keys"
        )
    
    return api_key

def main():
    """Main execution function for the Energy Advisor tool."""
    # Define help text separately
    help_text = """
Usage:
    energy-advisor.py [-v] [-m MODEL] <csv_file>
    energy-advisor.py (-h | --help)
    energy-advisor.py (-V | --version)

Options:
    -h --help           Show this help message
    -v --verbose        Enable verbose logging output
    -V --version        Show version and author information
    -m --model MODEL    Model to use for analysis [default: gpt-4]
                        Can be 'gpt-4', 'gpt-3.5-turbo', 'llama3.2', etc.

Arguments:
    csv_file            Path to CSV file containing energy data
"""
    try:
        # Parse arguments using docopt with separate help text
        arguments = docopt(help_text, version=f'Energy Advisor v{VERSION}\nAuthor: {AUTHOR}')
        
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
        
        # Validate input file
        input_path = validate_file_path(csv_file)
        
        # Set API key with better error handling if using GPT models
        if model.startswith('gpt-'):
            openai.api_key = get_api_key()

        # Load data
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        logger.info("Data loaded successfully")

        # Generate graph
        graph_path = "graph.png"
        generate_graph(data, graph_path)

        # Generate insights
        insights = generate_insights(data, model)

        # Generate recommendations
        recommendations = generate_recommendations(data, model)

        # Generate HTML report
        report_path = "report.html"
        generate_report(data, graph_path, insights, recommendations, report_path)

        # Open the report in the browser
        logger.info("Opening report in web browser")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        logger.info("Report generation completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
