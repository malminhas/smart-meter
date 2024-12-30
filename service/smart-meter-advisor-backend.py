"""
Smart Meter Advisor Backend Service
---------------------------------

A FastAPI-based backend service that provides smart meter data analysis and reporting.
This service interfaces with the n3rgy API to fetch smart meter data and provides
endpoints for data retrieval, analysis, and report generation.

Features:
- Smart meter validation
- Data retrieval with automatic date range handling
- Statistical analysis and insight generation
- Graph data generation for frontend visualization
- RESTful API endpoints
- Error handling and logging
- CORS support for frontend integration

Version History:
    1.0.5 - (December 30, 2024)
        - Added validation checks for IHD_MAC and MPAN
        - Added check for valid house number and postcode
        - Added check for valid start and end date
        - Updated Docker and nginx configuration for production
    1.0.4 - (December 28, 2024)
        - Added deployment to docker with nginx, frontend and backend
    1.0.3 - (December 27, 2024)
        - Adjusted CSS to make it more readable and consistent
        - Added download energy data feature
        - Re-ran all tests
    1.0.2 - (December 26, 2024)
        - Updated names of endpoints
        - Updated smart meter info endpoint
    1.0.1 - (December 25, 2024)
        - Added version endpoint
        - Fixed date range handling using availableCacheRange
        - Improved error handling and logging
    1.0.0 - Initial Release (December 24, 2024)
        - FastAPI backend implementation
        - Smart meter validation endpoint
        - Meter data retrieval with date block handling
        - Report generation with statistics and insights
        - Graph data generation for frontend
        - Integration with n3rgy API
        - Error handling and logging
        - CORS support
        - JSON-based data exchange

Todo:
- Add a simple database for storing meter information from users.

Author: Mal Minhas with AI help.
License: MIT
"""

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import Optional, Dict, Any, List
import pandas as pd  # type: ignore
import logging
import os
import requests  # type: ignore
from datetime import datetime, timedelta
import arrow  # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json
from bs4 import BeautifulSoup # type: ignore
import re

VERSION = "1.0.5"
DATE = "30.12.2024"

# Create FastAPI app with the correct base path
app = FastAPI(
    title="Smart Meter Advisor API",
    description="API for analyzing smart meter data",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    root_path="/energy-assistant/api"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smartmeteradvisor.uk", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartMeterRequest(BaseModel):
    mpan: str
    ihd_mac: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class SmartMeterValidation(BaseModel):
    mpan: str
    house_number: str
    postcode: str

class ReportRequest(BaseModel):
    """Request model for report generation."""
    mpan: str
    ihd_mac: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    output_dir: Optional[str] = "reports"

class DateRange(BaseModel):
    """Response model for date range."""
    start_date: str
    end_date: str

class MeterRequest(BaseModel):
    """Request model for meter validation."""
    mpan: str
    ihd_mac: str
    house_number: Optional[str] = None
    postcode: Optional[str] = None

def get_date(date_str: str = '', dayOffset: int = 0, monthOffset: int = 0) -> str:
    """Get formatted date with optional offsets."""
    start = arrow.now().floor('day')
    if date_str: 
        start = arrow.get(date_str, "DD.MM.YYYY")
    return start.shift(days=dayOffset, months=monthOffset).format("DD.MM.YYYY")

def validate_smart_meter(mpan: str, house_number: str, postcode: str) -> Dict[str, Any]:
    """Validate smart meter details with n3rgy API."""
    try:
        url = f"https://homebrew.n3rgy.com/cgi-bin/n3rgy-checkmeter.py?house={house_number}&postcode={postcode}&confirm=on"
        headers = {'Authorization': mpan}
        response = requests.get(url=url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid meter details")
            
        return {"status": "valid", "message": "Smart meter validated successfully"}
    except Exception as e:
        logger.error(f"Error validating smart meter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_smart_meter_data(mpan: str, ihd_mac: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Get smart meter data from n3rgy API."""
    try:
        # Format dates for API
        start = ''.join(start_date.split('.')[::-1])
        end = ''.join(end_date.split('.')[::-1])
        
        # Get electricity consumption
        url = f"https://consumer-api.data.n3rgy.com/electricity/consumption/1?start={start}&end={end}&output=json"
        headers = {'Authorization': ihd_mac}
        response = requests.get(url=url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to fetch smart meter data")
            
        data = response.json()
        
        # Process data into daily and monthly formats
        values = data.get('values', [])
        df = pd.DataFrame(values)
        
        if df.empty:
            return {"status": "error", "message": "No data available"}
            
        # Process data similar to original script
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Daily aggregation
        daily_data = df.groupby(df['timestamp'].dt.date).agg({
            'value': 'sum'
        }).reset_index()
        
        # Monthly aggregation
        monthly_data = df.groupby(df['timestamp'].dt.to_period('M')).agg({
            'value': 'sum'
        }).reset_index()
        monthly_data['timestamp'] = monthly_data['timestamp'].dt.strftime('%m/%Y')
        
        return {
            "status": "success",
            "daily_data": daily_data.to_dict('records'),
            "monthly_data": monthly_data.to_dict('records'),
            "unit": data.get('unit')
        }
    except Exception as e:
        logger.error(f"Error fetching smart meter data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_date_blocks(start_date: str, end_date: str, max_days: int = 90) -> List[Dict[str, str]]:
    """
    Generate blocks of dates that are within the maximum allowed range.
    
    Args:
        start_date: Start date in DD.MM.YYYY format
        end_date: End date in DD.MM.YYYY format
        max_days: Maximum number of days per block (default 90 for n3rgy API)
        
    Returns:
        List of dictionaries containing start and end dates for each block
    """
    logger.info(f"Generating date blocks from {start_date} to {end_date}")
    try:
        # Convert string dates to arrow objects
        start = arrow.get(start_date, "DD.MM.YYYY")
        end = arrow.get(end_date, "DD.MM.YYYY")
        
        blocks = []
        current_start = start
        
        while current_start < end:
            # Calculate block end date
            block_end = min(
                current_start.shift(days=max_days-1),  # -1 because range is inclusive
                end
            )
            
            blocks.append({
                'start_date': current_start.format("DD.MM.YYYY"),
                'end_date': block_end.format("DD.MM.YYYY")
            })
            
            # Move to next block
            current_start = block_end.shift(days=1)
            
        logger.info(f"Generated {len(blocks)} date blocks")
        return blocks
        
    except Exception as e:
        logger.error(f"Error generating date blocks: {str(e)}")
        raise ValueError(f"Error generating date blocks: {str(e)}")

async def get_block_data(mpan: str, ihd_mac: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Get smart meter data for a specific date block."""
    try:
        # Format dates for API (DD.MM.YYYY to YYYYMMDD)
        start = datetime.strptime(start_date, '%d.%m.%Y').strftime('%Y%m%d')
        end = datetime.strptime(end_date, '%d.%m.%Y').strftime('%Y%m%d')
        
        # Get electricity consumption using correct URL format
        url = f"https://consumer-api.data.n3rgy.com/electricity/consumption/1?start={start}&end={end}&output=json"
        headers = {
            'Authorization': ihd_mac
        }
        
        logger.info(f"Fetching block data from: {url}")
        response = requests.get(url=url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to fetch smart meter data. Status: {response.status_code}"
            )
            
        return response.json()
        
    except Exception as e:
        logger.error(f"Error fetching block data: {str(e)}")
        raise ValueError(f"Error fetching block data: {str(e)}")

@app.post("/validate-meter")
async def validate_meter(request: SmartMeterValidation):
    """Endpoint to validate smart meter details."""
    return validate_smart_meter(request.mpan, request.house_number, request.postcode)

@app.post("/meter-data")
async def get_meter_data(request: SmartMeterRequest) -> Dict[str, Any]:
    """Get smart meter data, handling large date ranges in blocks."""
    logger.info(f"Getting meter data for MPAN: {request.mpan}")
    try:
        # If dates not provided, get available range
        if not request.start_date or not request.end_date:
            try:
                range_data = await get_available_range(request.mpan, request.ihd_mac)
                request.start_date = range_data.start_date
                request.end_date = range_data.end_date
            except HTTPException as e:
                logger.error(f"Error getting date range: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="Unable to determine date range. Please provide start_date and end_date."
                )
        
        # Generate date blocks
        try:
            date_blocks = generate_date_blocks(request.start_date, request.end_date)
        except ValueError as e:
            logger.error(f"Error generating date blocks: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid date format or range")
        
        # Collect data for all blocks
        all_values = []
        for block in date_blocks:
            logger.info(f"Fetching data for block: {block['start_date']} to {block['end_date']}")
            try:
                block_data = await get_block_data(
                    request.mpan,
                    request.ihd_mac,
                    block['start_date'],
                    block['end_date']
                )
                all_values.extend(block_data.get('values', []))
            except ValueError as e:
                logger.error(f"Error fetching block data: {str(e)}")
                continue
        
        if not all_values:
            return {"status": "error", "message": "No data available"}
        
        # Process the data
        try:
            df = pd.DataFrame(all_values)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Daily aggregation
            daily_data = df.groupby(df['timestamp'].dt.date).agg({
                'value': 'sum'
            }).reset_index()
            
            # Monthly aggregation
            monthly_data = df.groupby(df['timestamp'].dt.to_period('M')).agg({
                'value': 'sum'
            }).reset_index()
            monthly_data['timestamp'] = monthly_data['timestamp'].dt.strftime('%m/%Y')
            
            return {
                "status": "success",
                "daily_data": daily_data.to_dict('records'),
                "monthly_data": monthly_data.to_dict('records'),
                "unit": "kWh"
            }
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing meter data")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching meter data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-range")
async def get_available_range(mpan: str, ihd_mac: str) -> DateRange:
    """Get available date range for smart meter data."""
    logger.info(f"Getting available range for MPAN: {mpan} and IHD_MAC: {ihd_mac}")
    try:
        # Query n3rgy API using correct URL format
        url = "https://consumer-api.data.n3rgy.com/electricity/consumption/1?output=json"
        headers = {
            'Authorization': ihd_mac
        }
        
        logger.info(f"Querying consumption data for range from: {url}")
        response = requests.get(url=url, headers=headers)
        
        logger.info(f"Response status code: {response.status_code}")
        if response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="Unauthorized In-Home Display (IDH) code provided for meter. Please use the 16 character GUID on the bottom of your IDH."
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Unable to fetch date range. Status: {response.status_code}, Response: {response.text}"
            )
            
        data = response.json()
        logger.info(f"Raw API response: {data}")
        
        # Extract start and end dates from availableCacheRange
        cache_range = data.get('availableCacheRange', {})
        if not cache_range:
            raise HTTPException(
                status_code=400,
                detail="No available cache range found in response"
            )
            
        start = str(cache_range.get('start', ''))[:8]  # Take first 8 characters (YYYYMMDD)
        end = str(cache_range.get('end', ''))[:8]      # Take first 8 characters (YYYYMMDD)
        
        logger.info(f"Extracted dates from cache range - start: {start}, end: {end}")
        
        if not start or not end or len(start) != 8 or len(end) != 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format received from smart meter"
            )
        
        # Convert from YYYYMMDD to DD.MM.YYYY
        try:
            start_date = f"{start[6:8]}.{start[4:6]}.{start[:4]}"
            end_date = f"{end[6:8]}.{end[4:6]}.{end[:4]}"
            
            logger.info(f"Converted dates - start_date: {start_date}, end_date: {end_date}")
            
            # Validate the converted dates
            datetime.strptime(start_date, '%d.%m.%Y')
            datetime.strptime(end_date, '%d.%m.%Y')
            
            return DateRange(start_date=start_date, end_date=end_date)
            
        except ValueError as e:
            logger.error(f"Error validating converted dates: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error validating date format: {str(e)}"
            )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to n3rgy API: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to smart meter service"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available range: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_graph_data(monthly_data: pd.DataFrame, daily_data: pd.DataFrame) -> Dict[str, Any]:
    """Generate graph data in a format suitable for frontend charting libraries."""
    try:
        # Monthly graph data
        monthly_graph = {
            "labels": monthly_data['timestamp'].tolist(),
            "datasets": [{
                "label": "Monthly Consumption (kWh)",
                "data": monthly_data['value'].tolist(),
                "type": "line",
                "fill": False,
                "borderColor": "rgb(75, 192, 192)",
                "tension": 0.1
            }]
        }

        # Daily graph data
        daily_graph = {
            "labels": [str(date) for date in daily_data['timestamp'].tolist()],
            "datasets": [{
                "label": "Daily Consumption (kWh)",
                "data": daily_data['value'].tolist(),
                "type": "line",
                "fill": False,
                "borderColor": "rgb(54, 162, 235)",
                "tension": 0.1
            }]
        }

        # Calculate moving average for trend line (7-day)
        if len(daily_data) >= 7:
            moving_avg = daily_data['value'].rolling(window=7).mean()
            daily_graph["datasets"].append({
                "label": "7-day Moving Average",
                "data": moving_avg.tolist(),
                "type": "line",
                "fill": False,
                "borderColor": "rgb(255, 99, 132)",
                "borderDash": [5, 5],
                "tension": 0.1
            })

        return {
            "monthly": monthly_graph,
            "daily": daily_graph,
            "config": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Consumption (kWh)"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Time Period"
                        }
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating graph data: {str(e)}")
        raise ValueError(f"Error generating graph data: {str(e)}")

@app.post("/generate-energy-report")
async def generate_energy_report(request: ReportRequest) -> Dict[str, Any]:
    """Generate energy usage report for the specified date range."""
    logger.info(f"Generating report data for MPAN: {request.mpan}")
    try:
        # Get meter data
        meter_data = await get_meter_data(request)
        
        if not meter_data or meter_data.get('status') != 'success':
            raise HTTPException(
                status_code=400, 
                detail="Failed to fetch meter data"
            )

        # Convert data to DataFrame for analysis
        monthly_data = pd.DataFrame(meter_data['monthly_data'])
        daily_data = pd.DataFrame(meter_data['daily_data'])
        
        # Calculate statistics
        stats = {
            "monthly": {
                "average": float(monthly_data['value'].mean()),
                "maximum": float(monthly_data['value'].max()),
                "minimum": float(monthly_data['value'].min()),
                "total": float(monthly_data['value'].sum())
            },
            "daily": {
                "average": float(daily_data['value'].mean()),
                "maximum": float(daily_data['value'].max()),
                "minimum": float(daily_data['value'].min()),
                "total": float(daily_data['value'].sum())
            }
        }
        
        # Generate insights
        insights = [
            f"Average monthly consumption: {stats['monthly']['average']:.2f} kWh",
            f"Highest monthly consumption: {stats['monthly']['maximum']:.2f} kWh",
            f"Lowest monthly consumption: {stats['monthly']['minimum']:.2f} kWh",
            f"Total consumption: {stats['monthly']['total']:.2f} kWh",
            f"Average daily consumption: {stats['daily']['average']:.2f} kWh"
        ]
        
        # Identify trends
        if len(monthly_data) >= 2:
            last_month = monthly_data['value'].iloc[-1]
            prev_month = monthly_data['value'].iloc[-2]
            percent_change = ((last_month - prev_month) / prev_month) * 100
            trend_message = (
                f"Consumption {'increased' if percent_change > 0 else 'decreased'} "
                f"by {abs(percent_change):.1f}% compared to previous month"
            )
            insights.append(trend_message)

        # Generate graph data
        graph_data = generate_graph_data(monthly_data, daily_data)
        
        return {
            "status": "success",
            "report_data": {
                "statistics": stats,
                "insights": insights,
                "consumption_data": {
                    "monthly": meter_data['monthly_data'],
                    "daily": meter_data['daily_data']
                },
                "graph_data": graph_data,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "mpan": request.mpan,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "unit": meter_data.get('unit', 'kWh')
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating report data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/version")
async def get_version() -> Dict[str, str]:
    """Get the current version of the service."""
    return {
        "version": VERSION,
        "date": DATE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-meter-report")
async def generate_meter_report(request: MeterRequest) -> Dict[str, Any]:
    """Get smart meter report details using the n3rgy homebrew API."""
    logger.info(f"Received request: {request}")
    
    # Validate required fields
    if not request.house_number or not request.postcode:
        raise HTTPException(
            status_code=400,
            detail="House number and postcode are required"
        )
    
    try:
        # First validate the meter
        logger.info("Validating smart meter...")
        validation = await validate_meter(request)
        logger.info(f"Validation result: {validation}")
        
        if validation['status'] != 'valid':
            logger.error(f"Validation failed: {validation['message']}")
            raise HTTPException(status_code=400, detail=validation['message'])
            
        # Get meter info using checkMeter function logic
        logger.info("Fetching smart meter info...")

        url = f"https://homebrew.n3rgy.com/cgi-bin/n3rgy-checkmeter.py"
        params = {
            'house': request.house_number,
            'postcode': request.postcode.upper().replace(' ', ''),  # Format postcode
            'confirm': 'on'
        }
        headers = {'Authorization': request.mpan}
        
        logger.info(f"Making request to {url} with params: {params}")
        response = requests.get(url=url, params=params, headers=headers)
        logger.info(f"Got response with status: {response.status_code}")
        logger.debug(f"Response content: {response.text[:50]}")  # Log first 500 chars of response
  
        if response.status_code != 200:
            logger.error(f"API error: {response.text}")
            raise HTTPException(
                status_code=400,
                detail=f"Unable to fetch meter info. Status: {response.status_code}"
            )

        # Parse the HTML response
        html = response.text
        logger.debug(f"Raw HTML response: {html}")  # Add this for debugging
        
        if "Error finding address" in html:
            raise HTTPException(
                status_code=404,
                detail="Invalid address or postcode provided for meter."
            )

        try:
            parsed_html = BeautifulSoup(html, features="html.parser")
            # More defensive parsing
            body_elements = parsed_html.find_all("body")
            if not body_elements or len(body_elements) < 2:
                logger.error("Unexpected HTML structure - not enough body elements")
                raise ValueError("Unexpected HTML structure")
                
            main_body = body_elements[1]
            title = main_body.find("h2")
            if not title:
                logger.error("Could not find title (h2) element")
                raise ValueError("Could not find title")
                
            pre_elements = main_body.find_all("pre")
            if not pre_elements:
                logger.error("Could not find pre elements")
                raise ValueError("Could not find pre elements")
                
            # Extract sections
            sections = {}
            current_section = None
            current_content = []
            
            # Split the pre content by lines
            lines = str(pre_elements[0]).split('\n')
            for line in lines:
                if '<b>' in line:
                    # If we have a previous section, save it
                    if current_section:
                        sections[current_section] = current_content
                        current_content = []
                    # Extract new section name without HTML tags
                    section_match = re.search(r'<b>(.*?)</b>', line)
                    if section_match:
                        current_section = section_match.group(1).replace('<u>', '').replace('</u>', '')
                elif line and line not in ['<pre>', '</pre>']:
                    current_content.append(line.strip())
            
            # Add the last section
            if current_section:
                sections[current_section] = current_content
            
            # Get available date range with error handling
            date_range = await get_available_range(request.mpan, request.ihd_mac)
            
            return {
                'status': 'success',
                'title': title.string if title else 'Smart Meter Report',
                'meter_info': {
                    'mpan': request.mpan,
                    'sections': sections
                },
                'date_range': date_range
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error parsing meter info: {str(e)}")
            logger.exception(e)  # This will log the full stack trace
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing meter information: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meter report: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000) 