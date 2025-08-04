#!/usr/bin/env python3
"""
Automated Test Suite for Travel API
Tests all endpoints with various scenarios including happy path, edge cases, and error conditions.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import your API app
from api import app

# Create test client
client = TestClient(app)

###############################
# Test Data & Fixtures        #
###############################

@pytest.fixture
def sample_transactions():
    """Sample bank transactions for testing categorization."""
    return {
        "transactions": [
            {"Description": "STARBUCKS COFFEE"},
            {"Description": "UBER RIDE"},
            {"Description": "AMAZON.COM PURCHASE"},
            {"Description": "GAS STATION FUEL"},
            {"Description": "RESTAURANT DINNER"}
        ]
    }

@pytest.fixture
def sample_hotel_search():
    """Sample hotel search request."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    return {
        "checkin_date": tomorrow,
        "checkout_date": day_after,
        "location": "Tokyo",
        "adults": 2,
        "children": 1,
        "infants": 0,
        "room_type": "standard",
        "amenities": ["wifi", "breakfast"],
        "fetch_mode": "fallback",
        "limit": 3,
        "debug": False
    }

@pytest.fixture
def sample_flight_search():
    """Sample flight search request."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "date": tomorrow,
        "from_airport": "TPE",
        "to_airport": "NRT",
        "trip": "one-way",
        "seat": "economy",
        "adults": 2,
        "children": 0,
        "infants_in_seat": 0,
        "infants_on_lap": 0,
        "fetch_mode": "local"
    }

@pytest.fixture
def sample_round_trip_flight():
    """Sample round-trip flight search request."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=8)).strftime("%Y-%m-%d")
    return {
        "date": tomorrow,
        "from_airport": "TPE",
        "to_airport": "NRT",
        "trip": "round-trip",
        "seat": "economy",
        "adults": 2,
        "children": 0,
        "infants_in_seat": 0,
        "infants_on_lap": 0,
        "fetch_mode": "local",
        "return_date": next_week
    }

@pytest.fixture
def sample_trip_plan():
    """Sample trip planning request."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=8)).strftime("%Y-%m-%d")
    return {
        "origin": "TPE",
        "destination": "NRT",
        "depart_date": tomorrow,
        "return_date": next_week,
        "adults": 2,
        "children": 0,
        "infants": 0,
        "hotel_preferences": {
            "star_rating": 3,
            "max_price_per_night": 200,
            "amenities": ["Free Wi-Fi"]
        },
        "room_type": "standard",
        "amenities": ["wifi"],
        "max_total_budget": 3000
    }

@pytest.fixture
def sample_airbnb_search():
    """Sample Airbnb search request."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    return {
        "check_in": tomorrow,
        "check_out": day_after,
        "ne_lat": 35.7,
        "ne_long": 139.8,
        "sw_lat": 35.6,
        "sw_long": 139.7,
        "zoom_value": 2,
        "price_min": 50,
        "price_max": 200,
        "place_type": "Entire home/apt",
        "amenities": [1, 2, 3],
        "currency": "USD",
        "language": "en",
        "proxy_url": "",
        "limit": 5
    }

@pytest.fixture
def sample_airbnb_details():
    """Sample Airbnb details request."""
    return {
        "room_id": 12345678,
        "currency": "USD",
        "language": "en",
        "proxy_url": ""
    }

###############################
# Health & Basic Endpoints    #
###############################

@pytest.mark.basic
@pytest.mark.quick
def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    # Verify timestamp is valid ISO format
    datetime.fromisoformat(data["timestamp"])

@pytest.mark.basic
@pytest.mark.quick
def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Travel API is running" in data["message"]
    assert "docs" in data
    assert "health" in data

###############################
# Transaction Categorization  #
###############################

@pytest.mark.categorization
@pytest.mark.unit
def test_categorize_transactions_success(sample_transactions):
    """Test successful transaction categorization."""
    response = client.post("/categorize", json=sample_transactions)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == len(sample_transactions["transactions"])
    
    # Verify each result has required fields
    for result in data["results"]:
        assert "index" in result
        assert "Description" in result
        assert "Category" in result
        assert "Sub_Category" in result
        assert "Category_Confidence" in result
        assert "Sub_Category_Confidence" in result
        assert isinstance(result["Category_Confidence"], float)
        assert isinstance(result["Sub_Category_Confidence"], float)
        assert 0 <= result["Category_Confidence"] <= 1
        assert 0 <= result["Sub_Category_Confidence"] <= 1

@pytest.mark.categorization
@pytest.mark.unit
def test_categorize_empty_transactions():
    """Test categorization with empty transaction list."""
    response = client.post("/categorize", json={"transactions": []})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "No transactions provided" in data["detail"]

@pytest.mark.categorization
@pytest.mark.unit
def test_categorize_missing_transactions():
    """Test categorization with missing transactions field."""
    response = client.post("/categorize", json={})
    assert response.status_code == 422  # Validation error

###############################
# Hotel Search Tests          #
###############################

@pytest.mark.hotels
@pytest.mark.unit
@patch('api.get_hotels')
def test_hotel_search_success(mock_get_hotels, sample_hotel_search):
    """Test successful hotel search."""
    # Mock the hotel search response
    mock_hotel = MagicMock()
    mock_hotel.name = "Test Hotel"
    mock_hotel.price = 150.0
    mock_hotel.rating = 4.5
    mock_hotel.url = "https://example.com/hotel"
    mock_hotel.amenities = ["wifi", "breakfast"]
    
    mock_result = MagicMock()
    mock_result.hotels = [mock_hotel]
    mock_result.lowest_price = 150.0
    mock_result.current_price = 150.0
    
    mock_get_hotels.return_value = mock_result
    
    response = client.post("/hotels/search", json=sample_hotel_search)
    assert response.status_code == 200
    data = response.json()
    assert "hotels" in data
    assert len(data["hotels"]) == 1
    assert data["hotels"][0]["name"] == "Test Hotel"
    assert data["hotels"][0]["price"] == 150.0
    assert data["lowest_price"] == 150.0

@pytest.mark.hotels
@pytest.mark.unit
def test_hotel_search_invalid_dates():
    """Test hotel search with invalid dates."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    invalid_request = {
        "checkin_date": yesterday,  # Past date
        "checkout_date": tomorrow,
        "location": "Tokyo",
        "adults": 2,
        "children": 0,
        "infants": 0
    }
    
    response = client.post("/hotels/search", json=invalid_request)
    assert response.status_code == 422  # Validation error

@pytest.mark.hotels
@pytest.mark.unit
def test_hotel_search_invalid_adults():
    """Test hotel search with invalid number of adults."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    
    invalid_request = {
        "checkin_date": tomorrow,
        "checkout_date": day_after,
        "location": "Tokyo",
        "adults": 0,  # Invalid: must be >= 1
        "children": 0,
        "infants": 0
    }
    
    response = client.post("/hotels/search", json=invalid_request)
    assert response.status_code == 422  # Validation error

###############################
# Flight Search Tests         #
###############################

@patch('api.get_flights')
@patch('api.resolve_to_iata')
def test_flight_search_success(mock_resolve_iata, mock_get_flights, sample_flight_search):
    """Test successful flight search."""
    # Mock IATA resolution
    mock_resolve_iata.side_effect = lambda x: x if x in ["TPE", "NRT"] else None
    
    # Mock flight search response
    mock_flight = MagicMock()
    mock_flight.name = "Test Airline"
    mock_flight.departure = "10:00 AM"
    mock_flight.arrival = "2:00 PM"
    mock_flight.price = 500.0
    mock_flight.stops = 1
    mock_flight.duration = "4h 30m"
    mock_flight.is_best = True
    
    mock_result = MagicMock()
    mock_result.flights = [mock_flight]
    mock_result.current_price = "500"
    
    mock_get_flights.return_value = mock_result
    
    response = client.post("/flights/search", json=sample_flight_search)
    assert response.status_code == 200
    data = response.json()
    assert "flights" in data
    assert len(data["flights"]) == 1
    assert data["flights"][0]["name"] == "Test Airline"
    assert data["flights"][0]["price"] == 500.0
    assert data["current_price"] == "500"

@patch('api.resolve_to_iata')
def test_flight_search_invalid_airport(mock_resolve_iata, sample_flight_search):
    """Test flight search with invalid airport codes."""
    # Mock IATA resolution to return None (invalid airport)
    mock_resolve_iata.return_value = None
    
    response = client.post("/flights/search", json=sample_flight_search)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Could not resolve" in data["detail"]

def test_flight_search_missing_return_date():
    """Test round-trip flight search without return date."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    invalid_request = {
        "date": tomorrow,
        "from_airport": "TPE",
        "to_airport": "NRT",
        "trip": "round-trip",  # Round-trip but no return_date
        "seat": "economy",
        "adults": 2,
        "children": 0,
        "infants_in_seat": 0,
        "infants_on_lap": 0,
        "fetch_mode": "local"
    }
    
    response = client.post("/flights/search", json=invalid_request)
    # The API returns 400 for this validation error, not 422
    assert response.status_code == 400  # Bad Request error

def test_flight_search_invalid_return_date():
    """Test round-trip flight search with return date before departure."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    invalid_request = {
        "date": tomorrow,
        "from_airport": "TPE",
        "to_airport": "NRT",
        "trip": "round-trip",
        "seat": "economy",
        "adults": 2,
        "children": 0,
        "infants_in_seat": 0,
        "infants_on_lap": 0,
        "fetch_mode": "local",
        "return_date": yesterday  # Before departure date
    }
    
    response = client.post("/flights/search", json=invalid_request)
    assert response.status_code == 422  # Validation error

###############################
# Trip Planning Tests         #
###############################

@patch('api.get_flights')
@patch('api.get_hotels')
@patch('api.resolve_to_iata')
def test_trip_plan_success(mock_resolve_iata, mock_get_hotels, mock_get_flights, sample_trip_plan):
    """Test successful trip planning."""
    # Mock IATA resolution
    mock_resolve_iata.side_effect = lambda x: x if x in ["TPE", "NRT"] else None
    
    # Mock flight search response
    mock_flight = MagicMock()
    mock_flight.name = "Test Airline"
    mock_flight.departure = "10:00 AM"
    mock_flight.arrival = "2:00 PM"
    mock_flight.price = 500.0
    mock_flight.stops = 1
    mock_flight.duration = "4h 30m"
    mock_flight.is_best = True
    
    mock_flight_result = MagicMock()
    mock_flight_result.flights = [mock_flight]
    
    # Mock hotel search response
    mock_hotel = MagicMock()
    mock_hotel.name = "Test Hotel"
    mock_hotel.price = 150.0
    mock_hotel.rating = 4.0
    mock_hotel.url = "https://example.com/hotel"
    mock_hotel.amenities = ["wifi"]
    
    mock_hotel_result = MagicMock()
    mock_hotel_result.hotels = [mock_hotel]
    
    # Mock the blocking calls
    mock_get_flights.return_value = mock_flight_result
    mock_get_hotels.return_value = mock_hotel_result
    
    response = client.post("/trip/plan", json=sample_trip_plan)
    assert response.status_code == 200
    data = response.json()
    assert "best_outbound_flight" in data
    assert "best_hotel" in data
    assert "total_estimated_cost" in data
    assert "breakdown" in data
    assert data["best_outbound_flight"]["name"] == "Test Airline"
    assert data["best_hotel"]["name"] == "Test Hotel"

def test_trip_plan_invalid_dates():
    """Test trip planning with invalid dates."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    invalid_request = {
        "origin": "TPE",
        "destination": "NRT",
        "depart_date": yesterday,  # Past date
        "return_date": tomorrow,
        "adults": 2,
        "children": 0,
        "infants": 0
    }
    
    response = client.post("/trip/plan", json=invalid_request)
    assert response.status_code == 422  # Validation error

###############################
# Airbnb Search Tests         #
###############################

@patch('api.pyairbnb.search_all')
def test_airbnb_search_success(mock_search_all, sample_airbnb_search):
    """Test successful Airbnb search."""
    # Mock Airbnb search response
    mock_airbnb_item = {
        "room_id": 12345678,
        "name": "Cozy Tokyo Apartment",
        "title": "Beautiful apartment in central Tokyo",
        "price": {
            "unit": {
                "amount": "150.00",
                "qualifier": "for 2 nights"
            }
        },
        "rating": {
            "value": 4.8,
            "reviewCount": 25
        },
        "images": [{"url": "https://example.com/image.jpg"}],
        "amenities": ["wifi", "kitchen"],
        "coordinates": {
            "latitude": 35.6762,
            "longitude": 139.6503
        },
        "badges": ["Superhost"]
    }
    
    mock_search_all.return_value = [mock_airbnb_item]
    
    response = client.post("/airbnbs/search", json=sample_airbnb_search)
    assert response.status_code == 200
    data = response.json()
    assert "airbnbs" in data
    assert len(data["airbnbs"]) == 1
    assert data["airbnbs"][0]["name"] == "Cozy Tokyo Apartment"
    assert data["airbnbs"][0]["price"] == 150.0
    assert data["airbnbs"][0]["per_night"] == 75.0
    assert data["lowest_price"] == 150.0

@patch('api.pyairbnb.get_details')
def test_airbnb_details_success(mock_get_details, sample_airbnb_details):
    """Test successful Airbnb details fetch."""
    # Mock Airbnb details response
    mock_details = {
        "name": "Cozy Tokyo Apartment",
        "title": "Beautiful apartment in central Tokyo",
        "rating": {
            "value": 4.8,
            "reviewCount": 25
        },
        "amenities": [
            {"title": "Wi-Fi"},
            {"title": "Kitchen"},
            {"title": "Air conditioning"}
        ],
        "images": [{"url": "https://example.com/image.jpg"}],
        "description": "A beautiful apartment in the heart of Tokyo",
        "coordinates": {
            "latitude": 35.6762,
            "longitude": 139.6503
        },
        "host": {
            "name": "John Doe"
        },
        "is_super_host": True
    }
    
    mock_get_details.return_value = mock_details
    
    response = client.post("/airbnbs/details", json=sample_airbnb_details)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Cozy Tokyo Apartment"
    assert data["rating"] == 4.8
    assert data["review_count"] == 25
    assert "Wi-Fi" in data["amenities"]
    assert data["host_name"] == "John Doe"
    assert data["is_superhost"] is True

###############################
# Itinerary Tests             #
###############################

@patch('requests.get')
def test_itinerary_success(mock_get):
    """Test successful itinerary fetch."""
    # Mock Foursquare API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "fsq_place_id": "test_id_1",
                "name": "Test Restaurant",
                "categories": [{"name": "Restaurant", "icon": {"prefix": "https://", "suffix": ".png"}}],
                "location": {"formatted_address": "123 Test St"},
                "distance": 500,
                "rating": 4.5,
                "price": 2,
                "hours": {"open_now": True},
                "latitude": 35.6762,
                "longitude": 139.6503,
                "website": "https://example.com"
            }
        ]
    }
    mock_get.return_value = mock_response
    
    # Mock PredictHQ events
    with patch('api.phq.events.search') as mock_phq:
        mock_event = MagicMock()
        mock_event.id = "event_1"
        mock_event.title = "Test Event"
        mock_event.category = "concert"
        mock_event.rank = 0.8
        mock_event.start = datetime.now()
        mock_event.end = datetime.now() + timedelta(hours=2)
        mock_event.location = {"lat": 35.6762, "lng": 139.6503}
        mock_phq.return_value = [mock_event]
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-01-01",
            "end_date": "2025-01-03",
            "radius": 10,
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "itinerary" in data
        assert "Food" in data["itinerary"]
        assert "Events" in data["itinerary"]
        assert len(data["itinerary"]["Food"]) == 1
        assert len(data["itinerary"]["Events"]) == 1

def test_itinerary_missing_params():
    """Test itinerary with missing required parameters."""
    response = client.get("/itinerary", params={
        "lat": 35.6762,
        # Missing lng, start_date, end_date
        "radius": 10
    })
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Missing" in data["detail"]

def test_itinerary_invalid_dates():
    """Test itinerary with invalid date format."""
    response = client.get("/itinerary", params={
        "lat": 35.6762,
        "lng": 139.6503,
        "start_date": "invalid-date",
        "end_date": "2025-01-03",
        "radius": 10
    })
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "YYYY-MM-DD format" in data["detail"]

###############################
# Error Handling Tests        #
###############################

def test_404_endpoint():
    """Test non-existent endpoint returns 404."""
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_invalid_json():
    """Test sending invalid JSON."""
    response = client.post("/categorize", data="invalid json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422

def test_missing_content_type():
    """Test request without proper content type."""
    response = client.post("/categorize", data='{"transactions": []}')
    assert response.status_code == 422

###############################
# Performance Tests           #
###############################

def test_health_check_performance():
    """Test health check endpoint performance."""
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1.0  # Should respond within 1 second

def test_root_endpoint_performance():
    """Test root endpoint performance."""
    start_time = time.time()
    response = client.get("/")
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1.0  # Should respond within 1 second

###############################
# PredictHQ Tests             #
###############################

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_success(mock_phq_search):
    """Test successful PredictHQ events search."""
    # Mock PredictHQ events
    mock_event1 = MagicMock()
    mock_event1.id = "event_1"
    mock_event1.title = "Summer Music Festival"
    mock_event1.category = "concert"
    mock_event1.rank = 0.85
    mock_event1.start = datetime(2025, 6, 15, 19, 0, 0)
    mock_event1.end = datetime(2025, 6, 15, 23, 0, 0)
    mock_event1.location = {"lat": 35.6762, "lng": 139.6503}
    mock_event1.place_hierarchies = ["Tokyo", "Japan"]
    mock_event1.description = "Annual summer music festival"
    mock_event1.labels = ["music", "outdoor"]
    mock_event1.timezone = "Asia/Tokyo"
    mock_event1.phq_attendance = 5000
    mock_event1.phq_rank = 0.9
    
    mock_event2 = MagicMock()
    mock_event2.id = "event_2"
    mock_event2.title = "Food Festival"
    mock_event2.category = "food"
    mock_event2.rank = 0.7
    mock_event2.start = datetime(2025, 6, 16, 12, 0, 0)
    mock_event2.end = datetime(2025, 6, 16, 18, 0, 0)
    mock_event2.location = {"lat": 35.6762, "lng": 139.6503}
    mock_event2.place_hierarchies = ["Tokyo", "Japan"]
    mock_event2.description = "Local food festival"
    mock_event2.labels = ["food", "local"]
    mock_event2.timezone = "Asia/Tokyo"
    mock_event2.phq_attendance = 2000
    mock_event2.phq_rank = 0.6
    
    mock_phq_search.return_value = [mock_event1, mock_event2]
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 10,
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "itinerary" in data
        assert "Events" in data["itinerary"]
        events = data["itinerary"]["Events"]
        assert len(events) == 2
        
        # Verify first event
        event1 = events[0]
        assert event1["id"] == "event_1"
        assert event1["title"] == "Summer Music Festival"
        assert event1["category"] == "concert"
        assert event1["rank"] == 0.85
        assert event1["start"] == "2025-06-15T19:00:00"
        assert event1["end"] == "2025-06-15T23:00:00"
        assert event1["location"] == {"lat": 35.6762, "lng": 139.6503}
        assert event1["place_hierarchies"] == ["Tokyo", "Japan"]
        assert event1["description"] == "Annual summer music festival"
        assert event1["labels"] == ["music", "outdoor"]
        assert event1["timezone"] == "Asia/Tokyo"
        assert event1["phq_attendance"] == 5000
        assert event1["phq_rank"] == 0.9

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_no_results(mock_phq_search):
    """Test PredictHQ events search with no results."""
    # Mock empty PredictHQ results
    mock_phq_search.return_value = []
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 10,
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "itinerary" in data
        # Events should not be in the response when no events found
        assert "Events" not in data["itinerary"]

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_api_error(mock_phq_search):
    """Test PredictHQ events search when API fails."""
    # Mock PredictHQ API error
    mock_phq_search.side_effect = Exception("PredictHQ API Error")
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 10,
            "limit": 10
        })
        
        # Should still return 200 even if PredictHQ fails
        assert response.status_code == 200
        data = response.json()
        assert "itinerary" in data
        # Events should not be in the response when PredictHQ fails
        assert "Events" not in data["itinerary"]

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_missing_attributes(mock_phq_search):
    """Test PredictHQ events with missing optional attributes."""
    # Mock PredictHQ event with missing attributes
    mock_event = MagicMock()
    mock_event.id = "event_1"
    mock_event.title = "Test Event"
    mock_event.category = "concert"
    mock_event.rank = 0.8
    mock_event.start = datetime(2025, 6, 15, 19, 0, 0)
    # No end time - explicitly set to None
    mock_event.end = None
    mock_event.location = {"lat": 35.6762, "lng": 139.6503}
    # Explicitly set missing attributes to None
    mock_event.place_hierarchies = None
    mock_event.description = None
    mock_event.labels = None
    mock_event.timezone = None
    mock_event.phq_attendance = None
    mock_event.phq_rank = None
    
    mock_phq_search.return_value = [mock_event]
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 10,
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "itinerary" in data
        assert "Events" in data["itinerary"]
        events = data["itinerary"]["Events"]
        assert len(events) == 1
        
        event = events[0]
        assert event["id"] == "event_1"
        assert event["title"] == "Test Event"
        assert event["start"] == "2025-06-15T19:00:00"
        assert event["end"] is None  # Should be None when missing
        # The API uses getattr(event, 'attribute', None) which returns None for missing attributes
        assert event["place_hierarchies"] is None
        assert event["description"] is None
        assert event["labels"] is None
        assert event["timezone"] is None
        assert event["phq_attendance"] is None
        assert event["phq_rank"] is None

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_parameter_formatting(mock_phq_search):
    """Test that PredictHQ parameters are formatted correctly."""
    # Mock PredictHQ events
    mock_event = MagicMock()
    mock_event.id = "event_1"
    mock_event.title = "Test Event"
    mock_event.category = "concert"
    mock_event.rank = 0.8
    mock_event.start = datetime(2025, 6, 15, 19, 0, 0)
    mock_event.end = datetime(2025, 6, 15, 23, 0, 0)
    mock_event.location = {"lat": 35.6762, "lng": 139.6503}
    
    mock_phq_search.return_value = [mock_event]
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 5000,  # 5000m = 5km radius
            "limit": 10
        })
        
        assert response.status_code == 200
        
        # Verify PredictHQ was called with correct parameters
        mock_phq_search.assert_called_once()
        call_args = mock_phq_search.call_args
        
        # Check 'within' parameter format: '5km@35.6762,139.6503'
        assert call_args[1]['within'] == '5km@35.6762,139.6503'
        
        # Check 'start' parameter format
        start_param = call_args[1]['start']
        assert start_param['gte'] == '2025-06-15'
        assert start_param['lte'] == '2025-06-16'

@pytest.mark.predicthq
@pytest.mark.unit
@patch('api.phq.events.search')
@patch.dict('os.environ', {'FOURSQUARE_API_KEY': 'test_key'})
def test_predicthq_events_large_radius(mock_phq_search):
    """Test PredictHQ events with large radius (>1000km)."""
    # Mock PredictHQ events
    mock_event = MagicMock()
    mock_event.id = "event_1"
    mock_event.title = "Test Event"
    mock_event.category = "concert"
    mock_event.rank = 0.8
    mock_event.start = datetime(2025, 6, 15, 19, 0, 0)
    mock_event.end = datetime(2025, 6, 15, 23, 0, 0)
    mock_event.location = {"lat": 35.6762, "lng": 139.6503}
    
    mock_phq_search.return_value = [mock_event]
    
    # Mock Foursquare API
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client.get("/itinerary", params={
            "lat": 35.6762,
            "lng": 139.6503,
            "start_date": "2025-06-15",
            "end_date": "2025-06-16",
            "radius": 1500000,  # 1500000m = 1500km radius
            "limit": 10
        })
        
        assert response.status_code == 200
        
        # Verify PredictHQ was called with correct parameters
        mock_phq_search.assert_called_once()
        call_args = mock_phq_search.call_args
        
        # Check 'within' parameter format: '1500km@35.6762,139.6503'
        assert call_args[1]['within'] == '1500km@35.6762,139.6503'

###############################
# Airport Tests               #
###############################

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_success(mock_get_airports):
    """Test successful nearest airport search."""
    # Mock airport data
    mock_airports = [
        {
            'code': 'TPE',
            'name': 'Taipei Taoyuan International Airport',
            'latitude': '25.0777',
            'longitude': '121.2328',
            'country': 'Taiwan'
        },
        {
            'code': 'NRT',
            'name': 'Narita International Airport',
            'latitude': '35.7720',
            'longitude': '140.3929',
            'country': 'Japan'
        },
        {
            'code': 'HND',
            'name': 'Tokyo Haneda Airport',
            'latitude': '35.5494',
            'longitude': '139.7798',
            'country': 'Japan'
        }
    ]
    mock_get_airports.return_value = mock_airports
    
    # Test coordinates near Taipei
    response = client.get("/airports/nearest", params={"lat": 25.0330, "lng": 121.5654})
    assert response.status_code == 200
    data = response.json()
    assert "iata" in data
    assert "name" in data
    assert "latitude" in data
    assert "longitude" in data
    assert "country" in data
    assert "distance_km" in data
    assert data["iata"] == "TPE"  # Should find Taipei airport
    assert isinstance(data["distance_km"], float)
    assert data["distance_km"] > 0

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_prefers_international(mock_get_airports):
    """Test that the endpoint prefers international airports when available."""
    # Mock airport data with both international and regional airports
    mock_airports = [
        {
            'code': 'TPE',
            'name': 'Taipei Taoyuan International Airport',
            'latitude': '25.0777',
            'longitude': '121.2328',
            'country': 'Taiwan'
        },
        {
            'code': 'TSA',
            'name': 'Taipei Songshan Airport',
            'latitude': '25.0697',
            'longitude': '121.5525',
            'country': 'Taiwan'
        }
    ]
    mock_get_airports.return_value = mock_airports
    
    # Test coordinates near both airports
    response = client.get("/airports/nearest", params={"lat": 25.0737, "lng": 121.3466})
    assert response.status_code == 200
    data = response.json()
    # Should prefer the international airport (TPE) over the regional one (TSA)
    assert data["iata"] == "TPE"

@pytest.mark.airports
@pytest.mark.unit
def test_nearest_airport_missing_coordinates():
    """Test nearest airport with missing coordinates."""
    response = client.get("/airports/nearest", params={})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Missing or invalid lat/lng" in data["detail"]

@pytest.mark.airports
@pytest.mark.unit
def test_nearest_airport_invalid_coordinates():
    """Test nearest airport with invalid coordinates."""
    response = client.get("/airports/nearest", params={"lat": "invalid", "lng": "invalid"})
    assert response.status_code == 422  # Validation error

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_no_airports_found(mock_get_airports):
    """Test nearest airport when no airports are found."""
    # Mock empty airport data
    mock_get_airports.return_value = []
    
    response = client.get("/airports/nearest", params={"lat": 25.0330, "lng": 121.5654})
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "No airport found" in data["detail"]

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_invalid_airport_data(mock_get_airports):
    """Test nearest airport with invalid airport data (missing coordinates)."""
    # Mock airport data with invalid coordinates
    mock_airports = [
        {
            'code': 'TPE',
            'name': 'Taipei Taoyuan International Airport',
            'latitude': '0',  # Invalid latitude
            'longitude': '0',  # Invalid longitude
            'country': 'Taiwan'
        },
        {
            'code': 'NRT',
            'name': 'Narita International Airport',
            'latitude': 'invalid',  # Invalid latitude
            'longitude': 'invalid',  # Invalid longitude
            'country': 'Japan'
        }
    ]
    mock_get_airports.return_value = mock_airports
    
    response = client.get("/airports/nearest", params={"lat": 25.0330, "lng": 121.5654})
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "No airport found" in data["detail"]

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_missing_iata_code(mock_get_airports):
    """Test nearest airport with airports missing IATA codes."""
    # Mock airport data without IATA codes
    mock_airports = [
        {
            'code': '',  # Missing IATA code
            'name': 'Some Airport',
            'latitude': '25.0777',
            'longitude': '121.2328',
            'country': 'Taiwan'
        }
    ]
    mock_get_airports.return_value = mock_airports
    
    response = client.get("/airports/nearest", params={"lat": 25.0330, "lng": 121.5654})
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "No airport found" in data["detail"]

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_api_error(mock_get_airports):
    """Test nearest airport when the airports API fails."""
    # Mock API error
    mock_get_airports.side_effect = Exception("API Error")
    
    response = client.get("/airports/nearest", params={"lat": 25.0330, "lng": 121.5654})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Failed to find nearest airport" in data["detail"]

@pytest.mark.airports
@pytest.mark.unit
@patch('api.get_airports')
def test_nearest_airport_distance_calculation(mock_get_airports):
    """Test that distance calculation works correctly."""
    # Mock airport data with known coordinates
    mock_airports = [
        {
            'code': 'TPE',
            'name': 'Taipei Taoyuan International Airport',
            'latitude': '25.0777',
            'longitude': '121.2328',
            'country': 'Taiwan'
        }
    ]
    mock_get_airports.return_value = mock_airports
    
    # Test coordinates very close to the airport
    response = client.get("/airports/nearest", params={"lat": 25.0777, "lng": 121.2328})
    assert response.status_code == 200
    data = response.json()
    assert data["iata"] == "TPE"
    # Distance should be very close to 0 km
    assert data["distance_km"] < 1.0

###############################
# Integration Tests           #
###############################

def test_api_documentation_accessible():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_openapi_schema_accessible():
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data

###############################
# Test Runner                 #
###############################

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 