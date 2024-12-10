import pandas as pd
import openai
import matplotlib.pyplot as plt
import os
import webbrowser
import logging
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_oracle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("OPEN_API_KEY")

# HTML template for the report
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Usage Report</title>
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
            color: white;
            padding: 20px 10px;
            text-align: center;
        }
        section {
            padding: 20px;
            margin: 20px auto;
            max-width: 800px;
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
        }
        ol {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        strong {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <h1>Energy Usage Report</h1>
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
    """Generate the graph and save as an image."""
    logger.info("Generating energy consumption graph")
    try:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%Y')
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data['Timestamp'], data['Electricity consumption (kWh)'], color='blue', marker='o', label='Electricity Consumption (kWh)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Electricity Consumption (kWh)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(data['Timestamp'])
        ax1.set_xticklabels(data['Timestamp'].dt.strftime('%b %y'), rotation=90)

        ax2 = ax1.twinx()
        ax2.plot(data['Timestamp'], data['Gas consumption (kWh)'], color='green', marker='x', label='Gas Consumption (kWh)')
        ax2.set_ylabel('Gas Consumption (kWh)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title('Electricity and Gas Consumption Over Time')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Graph saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error generating graph: {str(e)}")
        raise

def generate_insights(data):
    """Generate key insights using OpenAI GPT-4."""
    logger.info("Generating insights using GPT-4")
    try:
        description = data.describe(include='all').to_dict()
        prompt = f"""
        The following is a dataset description: {description}
        Please provide detailed insights into trends, outliers, and patterns. The dataset columns are: {', '.join(data.columns)}.
        Each insight should be a separate block of text as follows:
        <b>heading</b>: insight;&nbsp
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst providing a list of insights derived from the dataset."},
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Successfully generated insights")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise

def generate_recommendations(data):
    """Generate recommendations for lowering energy usage."""
    logger.info("Generating recommendations using GPT-4")
    try:
        prompt = """
        Based on the following dataset trends, provide actionable recommendations to lower electricity and gas usage.
        Each recommendation should be a separate block of text as follows:
        <b>heading</b>: recommendation;&nbsp
        """
        prompt += data.describe(include='all').to_string()
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an energy efficiency expert providing a list of recommendations."},
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Successfully generated recommendations")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def split_into_list(text):
    """Split numbered insights/recommendations into a list."""
    return [line.strip() for line in text.split("\n") if line.strip()]

def generate_report(data, graph_path, insights, recommendations, output_path):
    """Generate an HTML report."""
    logger.info(f"Generating HTML report to {output_path}")
    try:
        template = Template(html_template)
        insights_list = split_into_list(insights)
        recommendations_list = split_into_list(recommendations)
        html_content = template.render(insights_list=insights_list, recommendations_list=recommendations_list)
        with open(output_path, 'w') as f:
            f.write(html_content)
        logger.info("HTML report generated successfully")
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise

def main():
    logger.info("Starting energy usage report generation")
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Generate an energy usage report.")
        parser.add_argument("csv_file", help="Path to the CSV file containing energy data.")
        args = parser.parse_args()

        # Load data
        logger.info(f"Loading data from {args.csv_file}")
        data = pd.read_csv(args.csv_file)
        logger.info("Data loaded successfully")

        # Generate graph
        graph_path = "graph.png"
        generate_graph(data, graph_path)

        # Generate insights
        insights = generate_insights(data)

        # Generate recommendations
        recommendations = generate_recommendations(data)

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
