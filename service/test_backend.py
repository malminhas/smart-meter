import os
import pytest # type: ignore
import requests # type: ignore
import json
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv # type: ignore

# Load environment variables from local.env
env_path = os.path.join(os.path.dirname(__file__), '../local.env')
if not load_dotenv(env_path):
    raise RuntimeError(f"Could not load environment variables from {env_path}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BACKEND_URL = "http://localhost:8000"
MPAN = os.getenv('MPAN')
IHD_MAC = os.getenv('IHD_MAC')
HOUSE_NUMBER = os.getenv('HOUSE_NUMBER')
POSTCODE = os.getenv('POSTCODE')

if not all([MPAN, IHD_MAC, HOUSE_NUMBER, POSTCODE]):
    raise RuntimeError("MPAN, IHD_MAC, HOUSE_NUMBER and POSTCODE must be set in local.env")

# Fixtures
@pytest.fixture
def test_credentials():
    """Fixture to provide test credentials."""
    logger.info(f"Using test credentials - MPAN: {MPAN}, IHD_MAC: {IHD_MAC}, HOUSE_NUMBER: {HOUSE_NUMBER}, POSTCODE: {POSTCODE}")
    return {'mpan': MPAN, 'ihd_mac': IHD_MAC, 'house_number': HOUSE_NUMBER, 'postcode': POSTCODE}

@pytest.fixture
def test_dates():
    """Fixture to provide test dates."""
    return {
        'start_date': "01.03.2024",
        'end_date': "19.03.2024"
    }

def test_version():
    """Test the version endpoint."""
    logger.info("Testing version endpoint...")
    
    response = requests.get(f"{BACKEND_URL}/version")
    assert response.status_code == 200
    
    result = response.json()
    logger.info(f"Version response: {json.dumps(result, indent=2)}")
    
    assert 'version' in result
    assert 'timestamp' in result
    assert isinstance(result['version'], str)
    assert isinstance(result['timestamp'], str)

def test_validate_meter(test_credentials):
    """Test the validate-meter endpoint."""
    logger.info("Testing validate-meter endpoint...")
    
    payload = test_credentials
    response = requests.post(
        f"{BACKEND_URL}/validate-meter",
        headers={'Content-Type': 'application/json'},
        json=payload
    )
    
    assert response.status_code == 200
    result = response.json()
    logger.info(f"Validate meter response: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'valid'
    assert 'message' in result

def test_available_range(test_credentials):
    """Test the available-range endpoint."""
    logger.info("Testing available-range endpoint...")
    
    response = requests.get(
        f"{BACKEND_URL}/available-range",
        params=test_credentials
    )
    
    assert response.status_code == 200
    result = response.json()
    logger.info(f"Available range response: {json.dumps(result, indent=2)}")
    
    assert 'start_date' in result
    assert 'end_date' in result
    
    # Validate date formats
    for date_str in [result['start_date'], result['end_date']]:
        datetime.strptime(date_str, '%d.%m.%Y')

def test_meter_data_without_dates(test_credentials):
    """Test the meter-data endpoint without providing dates."""
    logger.info("Testing meter-data endpoint without dates...")
    
    response = requests.post(
        f"{BACKEND_URL}/meter-data",
        headers={'Content-Type': 'application/json'},
        json=test_credentials
    )
    
    assert response.status_code in [200, 400]  # 400 is acceptable if dates are required
    result = response.json()
    logger.info(f"Meter data response: {json.dumps(result, indent=2)}")

def test_meter_data_with_dates(test_credentials, test_dates):
    """Test the meter-data endpoint with specific dates."""
    logger.info("Testing meter-data endpoint with dates...")
    
    payload = {**test_credentials, **test_dates}
    response = requests.post(
        f"{BACKEND_URL}/meter-data",
        headers={'Content-Type': 'application/json'},
        json=payload
    )
    
    assert response.status_code == 200
    result = response.json()
    logger.info(f"Meter data response: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'success'
    assert 'daily_data' in result
    assert 'monthly_data' in result
    assert 'unit' in result

def test_meter_report(test_credentials):
    """Test the generate-meter-report endpoint."""
    logger.info("Testing generate-meter-report endpoint...")
    
    payload = {
        'mpan': test_credentials['mpan'],
        'ihd_mac': test_credentials['ihd_mac'],
        'house_number': test_credentials['house_number'],
        'postcode': test_credentials['postcode']
    }
    
    response = requests.post(
        f"{BACKEND_URL}/generate-meter-report",
        headers={'Content-Type': 'application/json'},
        json=payload
    )
    
    if response.status_code != 200:
        logger.error(f"Error response: {response.text}")
    
    assert response.status_code == 200
    result = response.json()
    logger.info(f"Meter report response: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'success'
    assert 'meter_info' in result
    assert 'sections' in result['meter_info']
    assert 'date_range' in result

def test_energy_report(test_credentials):
    """Test the generate-energy-report endpoint."""
    logger.info("Testing generate-energy-report endpoint...")
    
    # First get available date range
    response = requests.get(
        f"{BACKEND_URL}/available-range",
        params={
            'mpan': test_credentials['mpan'],
            'ihd_mac': test_credentials['ihd_mac']
        }
    )
    
    assert response.status_code == 200
    date_range = response.json()
    
    # Now generate the report
    payload = {
        'mpan': test_credentials['mpan'],
        'ihd_mac': test_credentials['ihd_mac'],
        'start_date': date_range['start_date'],
        'end_date': date_range['end_date']
    }
    
    response = requests.post(
        f"{BACKEND_URL}/generate-energy-report",
        headers={'Content-Type': 'application/json'},
        json=payload
    )
    
    if response.status_code != 200:
        logger.error(f"Error response: {response.text}")
    
    assert response.status_code == 200
    result = response.json()
    logger.info(f"Energy report response: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'success'
    assert 'report_data' in result
    assert 'consumption_data' in result['report_data']
    assert 'insights' in result['report_data'] 