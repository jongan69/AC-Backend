###############################
# Standard Library Imports    #
###############################
import os
import re
import io
import csv
import math
import json
import time
import logging
import asyncio
from datetime import datetime
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

###############################
# Third-Party Imports         #
###############################
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from predicthq import Client
import nltk

###############################
# Local Application Imports   #
###############################
import utils.nltk_bootstrap # noqa
from utils.api_predict import BankTransactionCategorizer
from fast_hotels.hotels_impl import HotelData, Guests
from fast_hotels import get_hotels
from fast_flights import FlightData, Passengers, get_flights, create_filter
import pyairbnb
from utils.airports import CITY_TO_IATA

load_dotenv()

# Initialize PredictHQ client
phq = Client(access_token=os.environ.get("PREDICTHQ_ACCESS_TOKEN"))

###############################
# Constants & Configuration   #
###############################

AIRPORTS_CSV_URL = "https://raw.githubusercontent.com/lxndrblz/Airports/refs/heads/main/airports.csv"

# Helper functions for parsing price and stops and run blocking operations
def _parse_price(price):
    """Parse a price string or number and return a float value, or None if invalid."""
    if isinstance(price, str):
        # Remove any non-numeric characters except dot and comma
        match = re.search(r"[\d,.]+", price)
        if match:
            # Remove commas and convert to float
            return float(match.group(0).replace(",", ""))
        return None
    return price

def _parse_stops(stops):
    """Parse the number of stops from various input types and return as int or None."""
    try:
        if stops is None:
            return None
        if isinstance(stops, int):
            return stops
        # Try to convert to int if it's a string representation of a number
        return int(stops)
    except (ValueError, TypeError):
        return None
    
def _run_blocking(func, *args, **kwargs):
    """Run a blocking function synchronously (for use in async context)."""
    return func(*args, **kwargs)

async def run_blocking(func, *args, **kwargs):
    """Run a blocking function in an executor, returning the result asynchronously."""
    loop = asyncio.get_running_loop()
    # Use the default executor (None) for more flexibility
    return await loop.run_in_executor(None, _run_blocking, func, *args, **kwargs)

app = FastAPI(title="Travel API", description="Plan your trip with the best flight and hotel options")

logging.basicConfig(level=logging.INFO)

###############################
# Helper Functions            #
###############################

# Helper to log before/after each blocking task and enforce a hard timeout

def call_with_timeout(func, timeout, *args, **kwargs):
    """Run a function in a separate process with a timeout. Raise TimeoutError if exceeded."""
    def target(q, *args, **kwargs):
        import logging
        try:
            logging.info(f"{func.__name__} started with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            q.put(result)
            logging.info(f"{func.__name__} finished successfully")
        except Exception as e:
            logging.error(f"{func.__name__} failed: {e}")
            q.put(e)
    q = Queue()
    p = Process(target=target, args=(q,)+args, kwargs=kwargs)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        logging.error(f"{func.__name__} timed out after {timeout} seconds")
        raise TimeoutError(f"{func.__name__} timed out after {timeout} seconds")
    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        raise TimeoutError(f"{func.__name__} did not return a result")

async def run_blocking_with_log(label, func, *args, **kwargs):
    """Run a blocking function asynchronously with logging before and after execution."""
    logging.info(f"Task {label} started")
    try:
        result = await run_blocking(func, *args, **kwargs)
        logging.info(f"Task {label} finished successfully")
        return result
    except Exception as e:
        logging.error(f"Task {label} failed: {e}")
        raise

###############################
# Data Models (Pydantic)      #
###############################

class Transaction(BaseModel):
    Description: str

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

class CategorizedTransaction(BaseModel):
    index: int
    Description: str
    Category: str
    Sub_Category: str
    Category_Confidence: float
    Sub_Category_Confidence: float

class CategorizedResponse(BaseModel):
    results: List[CategorizedTransaction]
    
# Hotel search models
class HotelSearchRequest(BaseModel):
    checkin_date: str = Field(..., description="Check-in date in YYYY-MM-DD format", examples=["2025-06-23"])
    checkout_date: str = Field(..., description="Check-out date in YYYY-MM-DD format", examples=["2025-06-25"])
    location: str = Field(..., description="City or location to search hotels in", examples=["Tokyo"])
    adults: int = Field(..., ge=1, description="Number of adult guests", examples=[2])
    children: int = Field(0, ge=0, description="Number of child guests", examples=[1])
    infants: int = Field(0, ge=0, description="Number of infant guests", examples=[0])
    room_type: str = Field("standard", description="Room type (e.g., standard, deluxe)", examples=["standard"])
    amenities: list[str] = Field(default_factory=list, description="List of required amenities", examples=[["wifi", "breakfast"]])
    fetch_mode: str = Field("fallback", description="'common', 'fallback', 'force-fallback', or 'local'", examples=["fallback"])
    limit: int = Field(3, ge=1, description="Maximum number of hotel results to return", examples=[3])
    debug: bool = Field(False, description="Enable debug mode for scraping", examples=[False])
    
    @field_validator('checkin_date')
    def checkin_date_not_in_past(cls, v):
        import datetime
        checkin = datetime.datetime.strptime(v, "%Y-%m-%d").date()
        today = datetime.date.today()
        if checkin < today:
            raise ValueError("checkin_date cannot be in the past.")
        return v

class HotelInfo(BaseModel):
    name: str = Field(..., description="Hotel name")
    price: Optional[float] = Field(None, description="Price per night in USD")
    rating: Optional[float] = Field(None, description="Hotel rating (e.g., 4.5)")
    url: Optional[str] = Field(None, description="URL to the hotel page")
    amenities: Optional[List[str]] = Field(None, description="List of hotel amenities")

class HotelSearchResponse(BaseModel):
    hotels: List[HotelInfo] = Field(..., description="List of hotel results")
    lowest_price: Optional[float] = Field(None, description="Lowest price among the results")
    current_price: Optional[float] = Field(None, description="Current price for the search")

# Flight search models
class FlightSearchRequest(BaseModel):
    date: str = Field(..., description="Flight date in YYYY-MM-DD format", examples=["2025-01-01"])
    from_airport: str = Field(..., description="IATA code of departure airport", examples=["TPE"])
    to_airport: str = Field(..., description="IATA code of arrival airport", examples=["MYJ"])
    trip: str = Field("one-way", description="Trip type: 'one-way' or 'round-trip'", examples=["one-way"])
    seat: str = Field("economy", description="Seat class: 'economy', 'business', etc.", examples=["economy"])
    adults: int = Field(..., ge=1, description="Number of adult passengers", examples=[2])
    children: int = Field(0, ge=0, description="Number of child passengers", examples=[1])
    infants_in_seat: int = Field(0, ge=0, description="Number of infants in seat", examples=[0])
    infants_on_lap: int = Field(0, ge=0, description="Number of infants on lap", examples=[0])
    fetch_mode: str = Field("fallback", description="Fetch mode: 'fallback', etc.", examples=["fallback"])
    return_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (required for round-trip)", examples=["2025-01-10"])

    @field_validator('return_date')
    def return_date_valid_for_round_trip(cls, v, info):
        import datetime
        trip = info.data.get('trip')
        if trip == 'round-trip':
            if not v:
                raise ValueError("return_date is required for round-trip flights.")
            return_dt = datetime.datetime.strptime(v, "%Y-%m-%d").date()
            depart_dt = datetime.datetime.strptime(info.data['date'], "%Y-%m-%d").date()
            if return_dt < depart_dt:
                raise ValueError("return_date cannot be before depart date.")
        return v

class FlightInfo(BaseModel):
    name: str = Field(..., description="Airline or flight name")
    departure: str = Field(..., description="Departure time and date")
    arrival: str = Field(..., description="Arrival time and date")
    arrival_time_ahead: Optional[str] = Field(None, description="Arrival time ahead (e.g., '+1' for next day)")
    duration: Optional[str] = Field(None, description="Flight duration (e.g., '4h 30m')")
    stops: Optional[int] = Field(None, description="Number of stops (0 for nonstop)")
    delay: Optional[str] = Field(None, description="Delay information if available")
    price: Optional[float] = Field(None, description="Flight price in USD")
    is_best: Optional[bool] = Field(None, description="Whether this is the best flight option")
    url: Optional[str] = Field(None, description="URL to view this flight search")

class FlightSearchResponse(BaseModel):
    flights: Optional[List[FlightInfo]] = Field(None, description="List of flight results (for one-way)")
    outbound_flights: Optional[List[FlightInfo]] = Field(None, description="List of outbound flight results (for round-trip)")
    return_flights: Optional[List[FlightInfo]] = Field(None, description="List of return flight results (for round-trip)")
    current_price: Optional[str] = Field(None, description="Current price for the search")

class HotelPreferences(BaseModel):
    star_rating: Optional[int] = Field(None, description="Minimum hotel star rating", examples=[3])
    max_price_per_night: Optional[float] = Field(None, description="Maximum price per night in USD", examples=[150])
    amenities: Optional[list[str]] = Field(None, description="Required hotel amenities", examples=[["Free Wi-Fi", "Breakfast ($)"]])

class TripPlanRequest(BaseModel):
    origin: str = Field(..., description="IATA code of origin airport", examples=["LHR"])
    destination: str = Field(..., description="IATA code of destination airport", examples=["CDG"])
    depart_date: str = Field(..., description="Departure date in YYYY-MM-DD format", examples=["2025-06-24"])
    return_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)", examples=["2025-06-30"])
    adults: int = Field(..., ge=1, description="Number of adult travelers", examples=[1])
    children: int = Field(0, ge=0, description="Number of child travelers", examples=[0])
    infants: int = Field(0, ge=0, description="Number of infant travelers", examples=[0])
    hotel_preferences: Optional[HotelPreferences] = Field(None, description="Hotel preferences (optional)")
    room_type: str = Field("standard", description="Room type (e.g., standard, deluxe)", examples=["standard"])
    amenities: Optional[List[str]] = Field(None, description="List of required hotel amenities", examples=[["wifi", "breakfast"]])
    max_total_budget: Optional[float] = Field(None, description="Maximum total budget for the trip in USD", examples=[3000])

    @field_validator('depart_date')
    def depart_date_not_in_past(cls, v):
        import datetime
        depart = datetime.datetime.strptime(v, "%Y-%m-%d").date()
        today = datetime.date.today()
        if depart < today:
            raise ValueError("depart_date cannot be in the past.")
        return v

class TripPlanResponse(BaseModel):
    best_outbound_flight: Optional[FlightInfo] = Field(None, description="Best outbound flight option")
    best_return_flight: Optional[FlightInfo] = Field(None, description="Best return flight option (if applicable)")
    best_hotel: Optional[HotelInfo] = Field(None, description="Best hotel option")
    total_estimated_cost: Optional[float] = Field(None, description="Total estimated cost for the trip in USD")
    per_person_per_day: Optional[float] = Field(None, description="Estimated cost per person per day in USD")
    breakdown: Dict[str, Any] = Field(..., description="Breakdown of costs and trip details")
    suggestions: Optional[str] = Field(None, description="Suggestions for optimizing the trip or saving money")
    warning: Optional[str] = Field(None, description="Warning or error messages about partial results or timeouts")

class AirbnbSearchRequest(BaseModel):
    check_in: str = Field(..., description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(..., description="Check-out date in YYYY-MM-DD format")
    ne_lat: float = Field(..., description="North-East latitude")
    ne_long: float = Field(..., description="North-East longitude")
    sw_lat: float = Field(..., description="South-West latitude")
    sw_long: float = Field(..., description="South-West longitude")
    zoom_value: int = Field(2, description="Zoom level for the map")
    price_min: Optional[int] = Field(0, description="Minimum price")
    price_max: Optional[int] = Field(0, description="Maximum price (0 for no max)")
    place_type: Optional[str] = Field("", description="Room type: 'Private room', 'Entire home/apt', or empty")
    amenities: Optional[List[int]] = Field(None, description="List of amenity IDs")
    currency: str = Field("USD", description="Currency code")
    language: str = Field("en", description="Language code")
    proxy_url: str = Field("", description="Proxy URL if needed")
    limit: int = Field(10, description="Max number of results to return")

class AirbnbInfo(BaseModel):
    room_id: int
    name: str
    title: Optional[str]
    price: Optional[float]
    per_night: Optional[float]
    url: Optional[str]
    rating: Optional[float]
    review_count: Optional[int]
    images: Optional[List[str]]
    badges: Optional[List[str]]
    latitude: Optional[float]
    longitude: Optional[float]

class AirbnbSearchResponse(BaseModel):
    airbnbs: List[AirbnbInfo]
    lowest_price: Optional[float]
    current_price: Optional[float]

class AirbnbDetailsRequest(BaseModel):
    room_id: int = Field(..., description="Airbnb room/listing ID")
    currency: str = Field("USD", description="Currency code")
    language: str = Field("en", description="Language code")
    proxy_url: str = Field("", description="Proxy URL if needed")

class AirbnbDetailsResponse(BaseModel):
    name: str
    url: str
    rating: Optional[float]
    amenities: Optional[list[str]]
    images: Optional[list[str]]
    review_count: Optional[int]
    description: Optional[str]
    location: Optional[dict]
    host_name: Optional[str]
    is_superhost: Optional[bool]
    

# Initialize the categorizer once (loads models)
categorizer = BankTransactionCategorizer()

# Initialize the executor for blocking operations
executor = ThreadPoolExecutor(max_workers=4)  # Lower for Raspberry Pi

###############################
# API Endpoints               #
###############################

@app.get("/health")
def health_check():
    """Health check endpoint. Returns API status and current timestamp."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
def root():
    """Root endpoint. Returns API status and documentation links."""
    return {"message": "Travel API is running", "docs": "/docs", "health": "/health"}

@app.post("/categorize", response_model=CategorizedResponse)
def categorize_transactions(request: TransactionsRequest):
    """Categorize a list of bank transactions by description."""
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    log_file_path = os.path.join(os.path.dirname(__file__), "data", "transaction_requests.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    descriptions = [t.Description for t in request.transactions]
    
    log_df = pd.DataFrame({
        "timestamp": [timestamp] * len(descriptions),
        "Description": descriptions
    })

    file_exists = os.path.isfile(log_file_path)
    log_df.to_csv(log_file_path, mode='a', header=not file_exists, index=False)

    # Prepare DataFrame
    df = pd.DataFrame([t.dict() for t in request.transactions])
    # Predict
    results_df = categorizer.predict(df)
    # Build response
    results = [CategorizedTransaction(
        index=idx,
        Description=row["Description"],
        Category=row["Category"],
        Sub_Category=row["Sub_Category"],
        Category_Confidence=float(row["Category_Confidence"]),
        Sub_Category_Confidence=float(row["Sub_Category_Confidence"])
    ) for idx, (_, row) in enumerate(results_df.iterrows())]
    return CategorizedResponse(results=results) 

@app.post("/hotels/search", response_model=HotelSearchResponse)
def search_hotels(req: HotelSearchRequest):
    """Search for hotels based on user criteria."""
    try:
        hotel_data = [HotelData(
            checkin_date=req.checkin_date,
            checkout_date=req.checkout_date,
            location=req.location,
            room_type=req.room_type,
            amenities=req.amenities
        )]
        guests = Guests(adults=req.adults, children=req.children, infants=req.infants)
        result = get_hotels(
            hotel_data=hotel_data,
            guests=guests,
            room_type=req.room_type,
            amenities=req.amenities,
            fetch_mode=req.fetch_mode,
            limit=req.limit,
            sort_by="price"
        )
        hotels = [HotelInfo(
            name=h.name,
            price=getattr(h, 'price', None),
            rating=getattr(h, 'rating', None),
            url=getattr(h, 'url', None),
            amenities=getattr(h, 'amenities', None)
        ) for h in result.hotels]
        return HotelSearchResponse(
            hotels=hotels,
            lowest_price=getattr(result, 'lowest_price', None),
            current_price=getattr(result, 'current_price', None)
        )
    except Exception as e:
        logging.error(f"Hotel search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/flights/search", response_model=FlightSearchResponse)
def search_flights(req: FlightSearchRequest):
    """Search for flights based on user criteria, resolving city/country names to IATA codes."""
    try:
        # --- IATA resolution for from_airport and to_airport ---
        from_iata = resolve_to_iata(req.from_airport)
        to_iata = resolve_to_iata(req.to_airport)
        if not from_iata or not to_iata:
            raise HTTPException(status_code=400, detail="Could not resolve from_airport or to_airport to an airport IATA code.")
        passengers = Passengers(
            adults=req.adults,
            children=req.children,
            infants_in_seat=req.infants_in_seat,
            infants_on_lap=req.infants_on_lap
        )
        if req.trip == "round-trip":
            if not req.return_date:
                raise HTTPException(status_code=400, detail="return_date is required for round-trip flights.")
            # Generate Google Flights URL for the whole trip
            filter = create_filter(
                flight_data=[
                    FlightData(
                        date=req.date,
                        from_airport=req.from_airport,
                        to_airport=req.to_airport
                    ),
                    FlightData(
                        date=req.return_date,
                        from_airport=req.to_airport,
                        to_airport=req.from_airport
                    )
                ],
                trip="round-trip",
                seat=req.seat,
                passengers=passengers,
            )
            b64 = filter.as_b64().decode('utf-8')
            flight_url = f"https://www.google.com/travel/flights?tfs={b64}"

            # Outbound leg
            outbound_data = [FlightData(
                date=req.date,
                from_airport=req.from_airport,
                to_airport=req.to_airport
            )]
            outbound_result = get_flights(
                flight_data=outbound_data,
                trip="one-way",
                seat=req.seat,
                passengers=passengers,
                fetch_mode=req.fetch_mode
            )
            outbound_flights = [FlightInfo(
                name=f.name,
                departure=f.departure,
                arrival=f.arrival,
                arrival_time_ahead=getattr(f, 'arrival_time_ahead', None),
                duration=getattr(f, 'duration', None),
                stops=_parse_stops(getattr(f, 'stops', None)),
                delay=getattr(f, 'delay', None),
                price=_parse_price(getattr(f, 'price', None)),
                is_best=getattr(f, 'is_best', None),
                url=flight_url
            ) for f in getattr(outbound_result, 'flights', [])]
            outbound_flights = [f for f in outbound_flights if f.name and f.departure and f.arrival and (f.price is not None and f.price > 0)]

            # Return leg
            return_data = [FlightData(
                date=req.return_date,
                from_airport=req.to_airport,
                to_airport=req.from_airport
            )]
            return_result = get_flights(
                flight_data=return_data,
                trip="one-way",
                seat=req.seat,
                passengers=passengers,
                fetch_mode=req.fetch_mode
            )
            return_flights = [FlightInfo(
                name=f.name,
                departure=f.departure,
                arrival=f.arrival,
                arrival_time_ahead=getattr(f, 'arrival_time_ahead', None),
                duration=getattr(f, 'duration', None),
                stops=_parse_stops(getattr(f, 'stops', None)),
                delay=getattr(f, 'delay', None),
                price=_parse_price(getattr(f, 'price', None)),
                is_best=getattr(f, 'is_best', None),
                url=flight_url
            ) for f in getattr(return_result, 'flights', [])]
            return_flights = [f for f in return_flights if f.name and f.departure and f.arrival and (f.price is not None and f.price > 0)]

            # Use the lower of the two current prices, or outbound's if only one is present
            current_price = getattr(outbound_result, 'current_price', None)
            if getattr(return_result, 'current_price', None) is not None:
                try:
                    current_price = str(min(float(current_price), float(return_result.current_price)))
                except Exception:
                    pass

            return FlightSearchResponse(
                outbound_flights=outbound_flights,
                return_flights=return_flights,
                current_price=current_price
            )
        else:
            flight_data = [FlightData(
                date=req.date,
                from_airport=req.from_airport,
                to_airport=req.to_airport
            )]
            # Generate filter and URL
            filter = create_filter(
                flight_data=flight_data,
                trip=req.trip,
                seat=req.seat,
                passengers=passengers,
            )
            b64 = filter.as_b64().decode('utf-8')
            flight_url = f"https://www.google.com/travel/flights?tfs={b64}"

            result = get_flights(
                flight_data=flight_data,
                trip=req.trip,
                seat=req.seat,
                passengers=passengers,
                fetch_mode=req.fetch_mode
            )
            flights = [FlightInfo(
                name=f.name,
                departure=f.departure,
                arrival=f.arrival,
                arrival_time_ahead=getattr(f, 'arrival_time_ahead', None),
                duration=getattr(f, 'duration', None),
                stops=_parse_stops(getattr(f, 'stops', None)),
                delay=getattr(f, 'delay', None),
                price=_parse_price(getattr(f, 'price', None)),
                is_best=getattr(f, 'is_best', None),
                url=flight_url
            ) for f in result.flights]
            flights = [f for f in flights if f.name and f.departure and f.arrival and (f.price is not None and f.price > 0)]
            return FlightSearchResponse(
                flights=flights,
                current_price=getattr(result, 'current_price', None)
            )
    except Exception as e:
        logging.error(f"Flight search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@lru_cache(maxsize=1)
def get_airports():
    """Fetch and cache the list of airports from a remote CSV file."""
    resp = requests.get(AIRPORTS_CSV_URL)
    resp.raise_for_status()
    f = io.StringIO(resp.text)
    reader = csv.DictReader(f)
    airports = [row for row in reader]
    return airports

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth (in km)."""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def geocode_city(city_name):
    """Geocode a city name to latitude and longitude using Overpass API."""
    url = "https://overpass-api.de/api/interpreter"
    # Try city, then town, then village
    place_types = ["city", "town", "village"]
    headers = {"User-Agent": "trip-planner"}
    for place in place_types:
        query = f'''
        [out:json][timeout:25];
        node["name"="{city_name}"]["place"="{place}"];
        out center 1;
        '''
        resp = requests.post(url, data={"data": query}, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        if elements:
            lat = elements[0]["lat"]
            lon = elements[0]["lon"]
            return float(lat), float(lon)
    return None

def resolve_to_iata(location):
    """Resolve a city, country, or IATA code to a valid IATA airport code."""
    # If already IATA code, return as is
    if isinstance(location, str) and len(location) == 3 and location.isalpha():
        return location.upper()
    # Check hardcoded city/country mapping
    if isinstance(location, str):
        city_key = location.strip().lower()
        if city_key in CITY_TO_IATA:
            return CITY_TO_IATA[city_key]
    # Otherwise, treat as city name
    coords = geocode_city(location)
    if not coords:
        return None
    lat, lng = coords
    airports = get_airports()
    min_dist = float("inf")
    nearest = None
    preferred = []
    # Prefer airports with 'International' in the name and within 50km
    for airport in airports:
        iata = airport.get("code")
        try:
            airport_lat = float(airport["latitude"])
            airport_lng = float(airport["longitude"])
        except Exception:
            continue
        if not iata or airport_lat == 0 or airport_lng == 0:
            continue
        dist = haversine(lat, lng, airport_lat, airport_lng)
        name = airport.get("name", "").lower()
        city = airport.get("city", "").lower()
        # Prefer international airports
        if "international" in name and dist < 50:
            preferred.append((airport, dist))
        # Next, prefer regional airports
        elif "regional" in name and dist < 50:
            preferred.append((airport, dist + 10))  # Slightly deprioritize
        # Next, prefer city match
        elif city and city_key in city and dist < 100:
            preferred.append((airport, dist + 20))
        if dist < min_dist:
            min_dist = dist
            nearest = airport
    if preferred:
        preferred.sort(key=lambda x: x[1])
        nearest = preferred[0][0]
    return nearest["code"] if nearest else None

@app.post("/trip/plan", response_model=TripPlanResponse)
async def plan_trip(req: TripPlanRequest):
    """Plan a trip by finding best flights and hotels, given origin, destination, and preferences."""
    try:
        # --- IATA resolution for origin and destination ---
        origin_iata = resolve_to_iata(req.origin)
        destination_iata = resolve_to_iata(req.destination)
        if not origin_iata or not destination_iata:
            raise HTTPException(status_code=400, detail="Could not resolve origin or destination to an airport IATA code.")
        # ... rest of the function uses origin_iata and destination_iata ...
        async def _plan_trip_inner():
            logging.info(f"Trip planning started: {req}")
            start_time = time.time()

            passengers = Passengers(
                adults=req.adults,
                children=req.children,
                infants_in_seat=0,
                infants_on_lap=0
            )
            outbound_flight_data = [FlightData(
                date=req.depart_date,
                from_airport=origin_iata,
                to_airport=destination_iata
            )]
            # Generate outbound flight URL
            outbound_filter = create_filter(
                flight_data=outbound_flight_data,
                trip="one-way",
                seat="economy",
                passengers=passengers,
            )
            outbound_b64 = outbound_filter.as_b64().decode('utf-8')
            outbound_flight_url = f"https://www.google.com/travel/flights?tfs={outbound_b64}"

            hotel_data = [HotelData(
                checkin_date=req.depart_date,
                checkout_date=req.return_date if getattr(req, 'return_date', None) else req.depart_date,
                location=req.destination,
                room_type=req.room_type,
                amenities=req.amenities
            )]
            guests = Guests(adults=req.adults, children=req.children, infants=req.infants)
            return_flight_data = None
            return_flight_url = None
            warning_msgs = []
            # Only run outbound and round-trip return flight for optimization
            if getattr(req, 'return_date', None):
                return_flight_data = [FlightData(
                    date=req.return_date,
                    from_airport=destination_iata,
                    to_airport=origin_iata
                )]
                # Generate return flight URL
                return_filter = create_filter(
                    flight_data=return_flight_data,
                    trip="round-trip",
                    seat="economy",
                    passengers=passengers,
                )
                return_b64 = return_filter.as_b64().decode('utf-8')
                return_flight_url = f"https://www.google.com/travel/flights?tfs={return_b64}"
                trip_types = ["round-trip"]
            else:
                trip_types = []

            logging.info("Starting outbound flight and hotel search tasks")
            tasks = [
                run_blocking_with_log("outbound_flight", partial(call_with_timeout, get_flights, 60, flight_data=outbound_flight_data, trip="one-way", seat="economy", passengers=passengers, fetch_mode="local")),
                run_blocking_with_log("hotel", partial(call_with_timeout, get_hotels, 60, hotel_data=hotel_data, guests=guests, room_type=req.room_type, amenities=req.amenities, fetch_mode="fallback", limit=10, sort_by="price"))
            ]
            if return_flight_data:
                for trip_type in trip_types:
                    label = f"return_flight_{trip_type}"
                    logging.info(f"Starting return flight search for trip_type={trip_type}")
                    tasks.append(run_blocking_with_log(label, partial(call_with_timeout, get_flights, 60, flight_data=return_flight_data, trip=trip_type, seat="economy", passengers=passengers, fetch_mode="local")))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            logging.info(f"Tasks completed in {time.time() - start_time:.2f}s")

            outbound_result = results[0]
            hotel_result = results[1]
            return_result = results[2] if return_flight_data else None

            if isinstance(outbound_result, Exception):
                logging.error(f"Outbound flight search error: {outbound_result}")
                warning_msgs.append(f"Outbound flight search failed: {outbound_result}")
            if isinstance(hotel_result, Exception):
                logging.error(f"Hotel search error: {hotel_result}")
                warning_msgs.append(f"Hotel search failed: {hotel_result}")
            best_return_flight = None
            if return_flight_data:
                if isinstance(return_result, Exception):
                    logging.error(f"Return flight search error for round-trip: {return_result}")
                    warning_msgs.append(f"Return flight search failed: {return_result}")

            # Outbound flight
            outbound_flights = [
                FlightInfo(
                    name=f.name,
                    departure=f.departure,
                    arrival=f.arrival,
                    arrival_time_ahead=getattr(f, 'arrival_time_ahead', None),
                    duration=getattr(f, 'duration', None),
                    stops=_parse_stops(getattr(f, 'stops', None)),
                    delay=getattr(f, 'delay', None),
                    price=_parse_price(getattr(f, 'price', None)),
                    is_best=getattr(f, 'is_best', None),
                    url=outbound_flight_url
                ) for f in getattr(outbound_result, 'flights', [])
            ] if not isinstance(outbound_result, Exception) else []
            best_outbound_flight = min((f for f in outbound_flights if f.price is not None), key=lambda x: x.price, default=None)

            # Return flight (only round-trip)
            if return_flight_data and not isinstance(return_result, Exception):
                return_flights = [
                    FlightInfo(
                        name=f.name,
                        departure=f.departure,
                        arrival=f.arrival,
                        arrival_time_ahead=getattr(f, 'arrival_time_ahead', None),
                        duration=getattr(f, 'duration', None),
                        stops=_parse_stops(getattr(f, 'stops', None)),
                        delay=getattr(f, 'delay', None),
                        price=_parse_price(getattr(f, 'price', None)),
                        is_best=getattr(f, 'is_best', None),
                        url=return_flight_url
                    ) for f in getattr(return_result, 'flights', [])
                ]
                best_return_flight = min((f for f in return_flights if f.price is not None), key=lambda x: x.price, default=None)

            # Hotels
            hotels = [
                HotelInfo(
                    name=h.name,
                    price=getattr(h, 'price', None),
                    rating=getattr(h, 'rating', None),
                    url=getattr(h, 'url', None),
                    amenities=getattr(h, 'amenities', None)
                ) for h in getattr(hotel_result, 'hotels', [])
            ] if not isinstance(hotel_result, Exception) else []
            filtered_hotels = hotels
            if req.hotel_preferences:
                if req.hotel_preferences.star_rating:
                    filtered_hotels = [h for h in filtered_hotels if h.rating and h.rating >= req.hotel_preferences.star_rating]
                if req.hotel_preferences.max_price_per_night:
                    filtered_hotels = [h for h in filtered_hotels if h.price and h.price <= req.hotel_preferences.max_price_per_night]
                if req.hotel_preferences.amenities:
                    filtered_hotels = [h for h in filtered_hotels if h.amenities and all(a in h.amenities for a in req.hotel_preferences.amenities)]
            best_hotel = min((h for h in filtered_hotels if h.price is not None), key=lambda x: x.price, default=None)

            # Calculate total cost
            total_flight_cost = 0
            if best_outbound_flight and best_outbound_flight.price:
                total_flight_cost += best_outbound_flight.price * (req.adults + req.children)
            if best_return_flight and best_return_flight.price:
                total_flight_cost += best_return_flight.price * (req.adults + req.children)
            nights = (pd.to_datetime(req.return_date) - pd.to_datetime(req.depart_date)).days if getattr(req, 'return_date', None) else 1
            total_hotel_cost = (best_hotel.price * nights) if best_hotel and best_hotel.price else 0
            total_estimated_cost = total_flight_cost + total_hotel_cost
            per_person_per_day = total_estimated_cost / ((req.adults + req.children) * nights) if nights > 0 and (req.adults + req.children) > 0 else None

            breakdown = {
                "flight": total_flight_cost,
                "hotel": total_hotel_cost,
                "nights": nights,
                "adults": req.adults,
                "children": req.children
            }
            suggestions = None
            if req.max_total_budget and total_estimated_cost > req.max_total_budget:
                suggestions = "Consider adjusting your dates, reducing hotel star rating, or increasing your budget."

            logging.info(f"Trip planning completed successfully in {time.time() - start_time:.2f}s")
            return TripPlanResponse(
                best_outbound_flight=best_outbound_flight,
                best_return_flight=best_return_flight,
                best_hotel=best_hotel,
                total_estimated_cost=total_estimated_cost,
                per_person_per_day=per_person_per_day,
                breakdown=breakdown,
                suggestions=suggestions,
                warning="; ".join(warning_msgs) if warning_msgs else None
            )
        try:
            return await asyncio.wait_for(_plan_trip_inner(), timeout=90)
        except asyncio.TimeoutError:
            logging.error("Trip planning timed out after 90 seconds")
            raise HTTPException(status_code=504, detail="Trip planning took too long. Please try again later.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Trip plan error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) 

@app.post("/airbnbs/search", response_model=AirbnbSearchResponse)
def search_airbnbs(req: AirbnbSearchRequest):
    """Search for Airbnbs in a given area and date range."""
    try:
        results = pyairbnb.search_all(
            check_in=req.check_in,
            check_out=req.check_out,
            ne_lat=req.ne_lat,
            ne_long=req.ne_long,
            sw_lat=req.sw_lat,
            sw_long=req.sw_long,
            zoom_value=req.zoom_value,
            price_min=req.price_min,
            price_max=req.price_max,
            place_type=req.place_type,
            amenities=req.amenities or [],
            currency=req.currency,
            language=req.language,
            proxy_url=req.proxy_url
        )
        airbnbs = []
        lowest_price = None
        for item in results:
            logging.info(f"Raw Airbnb item: {json.dumps(item, indent=2, ensure_ascii=False)}")
            price = None
            per_night = None
            nights = 1
            try:
                price = float(item["price"]["unit"]["amount"])
                qualifier = item["price"]["unit"].get("qualifier", "")
                import re
                match = re.search(r"for (\\d+) nights?", qualifier)
                if match:
                    nights = int(match.group(1))
                if nights > 0:
                    per_night = price / nights
            except Exception:
                price = None
                per_night = None
            url = f"https://www.airbnb.com/rooms/{item['room_id']}" if "room_id" in item else ""
            amenities = item.get("amenities", [])
            rating = None
            review_count = None
            if "rating" in item:
                rating = item["rating"].get("value")
                review_count = item["rating"].get("reviewCount")
                if review_count is not None:
                    try:
                        review_count = int(review_count)
                    except Exception:
                        review_count = None
            images = [img["url"] for img in item.get("images", []) if "url" in img]
            location = item.get("coordinates", {})
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if longitude is None:
                longitude = location.get("longitud")
            badges = item.get("badges", [])
            airbnbs.append(AirbnbInfo(
                room_id=item.get("room_id"),
                name=item.get("name", ""),
                title=item.get("title", None),
                price=price,
                per_night=per_night,
                url=url,
                rating=rating,
                review_count=review_count,
                images=images,
                badges=badges,
                latitude=latitude,
                longitude=longitude
            ))
            if price is not None and (lowest_price is None or price < lowest_price):
                lowest_price = price
        return AirbnbSearchResponse(
            airbnbs=airbnbs[:req.limit],
            lowest_price=lowest_price,
            current_price=lowest_price
        )
    except Exception as e:
        logging.error(f"Airbnb search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/airbnbs/details", response_model=AirbnbDetailsResponse)
def airbnb_details(req: AirbnbDetailsRequest):
    """Fetch detailed information for a specific Airbnb listing."""
    try:
        details = pyairbnb.get_details(
            room_id=req.room_id,
            currency=req.currency,
            language=req.language,
            proxy_url=req.proxy_url
        )
        logging.info("Raw Airbnb details:\n%s", json.dumps(details, indent=2, ensure_ascii=False))
        name = details.get("name") or details.get("title") or ""
        url = f"https://www.airbnb.com/rooms/{req.room_id}"
        rating = None
        raw_rating = details.get("rating", None)
        if isinstance(raw_rating, dict):
            rating = raw_rating.get("value", None)
        elif isinstance(raw_rating, (float, int)):
            rating = float(raw_rating)
        review_count = None
        if isinstance(raw_rating, dict):
            review_count = raw_rating.get("reviewCount", None)
        if not review_count:
            host_details = details.get('host_details', {}).get('data', {}).get('node', {})
            review_count = host_details.get('reviewsReceivedFromGuests', {}).get('count')
        if not review_count:
            review_count = details.get('rating', {}).get('review_count')
        if review_count is not None:
            try:
                review_count = int(review_count)
            except Exception:
                review_count = None
        amenities = []
        raw_amenities = details.get("amenities", [])
        if isinstance(raw_amenities, dict):
            raw_amenities = list(raw_amenities.values())
        if isinstance(raw_amenities, list):
            for a in raw_amenities:
                if isinstance(a, dict):
                    if "title" in a:
                        amenities.append(a["title"])
                    elif "name" in a:
                        amenities.append(a["name"])
                    elif "description" in a:
                        amenities.append(a["description"])
                    else:
                        amenities.append(str(a))
                elif isinstance(a, str):
                    amenities.append(a)
        amenities = list(dict.fromkeys(amenities))
        images = []
        raw_images = details.get("images", [])
        if isinstance(raw_images, list):
            images = [img["url"] for img in raw_images if isinstance(img, dict) and "url" in img]
        description = details.get("description", None)
        location = details.get("coordinates", None)
        host_node = details.get('host', {})
        host_name = host_node.get('name')
        is_superhost = details.get('is_super_host')
        return AirbnbDetailsResponse(
            name=name,
            url=url,
            rating=rating,
            amenities=amenities,
            images=images,
            review_count=review_count,
            description=description,
            location=location,
            host_name=host_name,
            is_superhost=is_superhost
        )
    except Exception as e:
        logging.error(f"Airbnb details error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- Itinerary Category Groups and Time Slots ---
CATEGORY_GROUPS = {
    "Food": ["Restaurant", "Burger Joint", "Coffee Shop", "Bakery"],
    "Attractions": ["Museum", "Art Gallery", "Theme Park", "Aquarium", "Zoo"],
    "Nature": ["Park", "Scenic Lookout", "Beach", "Garden", "Trail"],
    "Nightlife": ["Bar", "Lounge", "Nightclub"],
    "Shopping": ["Mall", "Boutique", "Gift Shop", "Market"],
    "Entertainment": ["Arcade", "Bowling Alley", "Movie Theater"]
}

TIME_SLOTS = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 2)  # special case
}

def infer_category(category_name: str) -> str:
    for group, values in CATEGORY_GROUPS.items():
        if any(v.lower() in category_name.lower() for v in values):
            return group
    return "Other"

def determine_time_slot(hours: dict) -> list:
    if not hours or not hours.get("regular") or not isinstance(hours["regular"], list):
        return []
    now = datetime.now()
    current_hour = now.hour
    slot_tags = []
    for slot, (start, end) in TIME_SLOTS.items():
        if slot == "night":
            if current_hour >= 21 or current_hour < 2:
                slot_tags.append("night")
        elif current_hour >= start and current_hour < end:
            slot_tags.append(slot)
    return slot_tags

@app.get("/itinerary")
def get_itinerary(
    lat: float,
    lng: float,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    radius: float = 10000,
    limit: int = 30,
    query: str = "",
    open_now: str = "false"
):
    """Get grouped itinerary suggestions (places, events) for a location and date range."""
    if not lat or not lng:
        raise HTTPException(status_code=400, detail="Missing 'lat' or 'lng'")
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing 'start_date' or 'end_date'")
    try:
        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except Exception:
            raise HTTPException(status_code=400, detail="start_date and end_date must be in YYYY-MM-DD format")
        # --- Foursquare Places ---
        params = {
            "ll": f"{lat},{lng}",
            "radius": str(radius),
            "limit": str(limit),
            "sort": "relevance"
        }
        if query:
            params["query"] = query
        if open_now == "true":
            params["open_now"] = "true"
        api_key = os.environ.get("FOURSQUARE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="FOURSQUARE_API_KEY not set in environment")
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
            "X-Places-Api-Version": "2025-06-17"
        }
        response = requests.get(
            "https://places-api.foursquare.com/places/search",
            params=params,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        grouped = {}
        for place in results:
            categoryName = place.get("categories", [{}])[0].get("name", "Unknown")
            group = infer_category(categoryName)
            timeSlots = determine_time_slot(place.get("hours"))
            icon = None
            cat = place.get("categories", [{}])[0]
            if cat.get("icon"):
                icon = f"{cat['icon']['prefix']}64{cat['icon']['suffix']}"
            placeObj = {
                "id": place.get("fsq_place_id"),
                "name": place.get("name"),
                "address": place.get("location", {}).get("formatted_address", ""),
                "category": categoryName,
                "icon": icon,
                "distance_m": place.get("distance"),
                "rating": place.get("rating"),
                "price_level": place.get("price"),
                "open_now": place.get("hours", {}).get("open_now"),
                "time_slots": timeSlots,
                "lat": place.get("latitude"),
                "lng": place.get("longitude"),
                "website": place.get("website")
            }
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(placeObj)
        # --- PredictHQ Events ---
        events = []
        try:
            # PredictHQ expects 'within' as '{radius}{unit}@{lat},{lng}', e.g. '10km@-36.84,174.76'
            phq_within = f"{int(radius/1000)}km@{lat},{lng}"
            phq_start = {
                'gte': start_date,
                'lte': end_date
            }
            for event in phq.events.search(within=phq_within, start=phq_start):
                events.append({
                    "id": event.id,
                    "title": event.title,
                    "category": event.category,
                    "rank": event.rank,
                    "start": event.start.strftime('%Y-%m-%dT%H:%M:%S'),
                    "end": event.end.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(event, 'end') and event.end else None,
                    "location": event.location,
                    "place_hierarchies": getattr(event, 'place_hierarchies', None),
                    "description": getattr(event, 'description', None),
                    "labels": getattr(event, 'labels', None),
                    "timezone": getattr(event, 'timezone', None),
                    "phq_attendance": getattr(event, 'phq_attendance', None),
                    "phq_rank": getattr(event, 'phq_rank', None),
                })
        except Exception as e:
            logging.error(f"PredictHQ events error: {e}")
        if events:
            grouped["Events"] = events
        return {"itinerary": grouped}
    except requests.RequestException as err:
        logging.error(f"Foursquare API error: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch itinerary: {err}")
    except Exception as err:
        logging.error(f"Itinerary endpoint error: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch itinerary: {err}")

