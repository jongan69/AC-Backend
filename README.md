# AC-Backend

[![Refresh Webhook](https://github.com/jongan69/AC-Backend/actions/workflows/refresh-webhook.yml/badge.svg)](https://github.com/jongan69/AC-Backend/actions/workflows/refresh-webhook.yml)

## Architecture Overview

```mermaid
flowchart TD
    A[Client] -->|/categorize| B[Transaction Categorization]
    A -->|/hotels/search| C[Hotel Search]
    A -->|/flights/search| D[Flight Search]
    A -->|/trip/plan| E[Trip Planning]
    A -->|/airbnbs/search| F[Airbnb Search]
    A -->|/airbnbs/details| G[Airbnb Details]
    A -->|/itinerary| H[Itinerary Suggestions]
    B -->|uses| B1[BankTransactionCategorizer]
    C -->|uses| C1[get_hotels]
    D -->|uses| D1[get_flights]
    E -->|uses| D1
    E -->|uses| C1
    F -->|uses| F1[pyairbnb.search_all]
    G -->|uses| G1[pyairbnb.get_details]
    H -->|uses| H1[Foursquare API]
    H -->|uses| H2[PredictHQ Events]
```

## Endpoint-Feature Mapping

```mermaid
flowchart TD
    subgraph API Endpoints
        A1[/categorize\nPOST/] 
        A2[/hotels/search\nPOST/]
        A3[/flights/search\nPOST/]
        A4[/trip/plan\nPOST/]
        A5[/airbnbs/search\nPOST/]
        A6[/airbnbs/details\nPOST/]
        A7[/itinerary\nGET/]
        A8[/health\nGET/]
        A9[/\nGET/]
    end
    subgraph Features
        F1[Transaction Categorization]
        F2[Hotel Search]
        F3[Flight Search]
        F4[Trip Planning]
        F5[Airbnb Search]
        F6[Airbnb Details]
        F7[Itinerary Suggestions]
        F8[Health Check]
        F9[Root Welcome]
    end
    A1 --> F1
    A2 --> F2
    A3 --> F3
    A4 --> F4
    A5 --> F5
    A6 --> F6
    A7 --> F7
    A8 --> F8
    A9 --> F9
```

## API Sequence Example

```mermaid
sequenceDiagram
    participant Client
    participant API as AC-Backend API
    participant Hotels as Hotel Service
    participant Flights as Flight Service
    participant Airbnb as Airbnb Service
    participant Foursquare as Foursquare API
    participant PredictHQ as PredictHQ

    Client->>API: POST /categorize
    API->>API: Categorize transactions
    API-->>Client: Categorized results

    Client->>API: POST /hotels/search
    API->>Hotels: get_hotels
    Hotels-->>API: Hotel results
    API-->>Client: Hotel search results

    Client->>API: POST /flights/search
    API->>Flights: get_flights
    Flights-->>API: Flight results
    API-->>Client: Flight search results

    Client->>API: POST /trip/plan
    API->>Flights: get_flights (outbound/return)
    API->>Hotels: get_hotels
    Flights-->>API: Flight options
    Hotels-->>API: Hotel options
    API-->>Client: Best trip plan

    Client->>API: POST /airbnbs/search
    API->>Airbnb: pyairbnb.search_all
    Airbnb-->>API: Airbnb results
    API-->>Client: Airbnb search results

    Client->>API: POST /airbnbs/details
    API->>Airbnb: pyairbnb.get_details
    Airbnb-->>API: Airbnb details
    API-->>Client: Airbnb details

    Client->>API: GET /itinerary
    API->>Foursquare: Places API
    API->>PredictHQ: Events API
    Foursquare-->>API: Places
    PredictHQ-->>API: Events
    API-->>Client: Grouped itinerary
```

---

## API Endpoints

### Root & Health

#### `GET /`
- **Description:** Root endpoint. Returns API status and documentation links.
- **Response:**
  ```json
  { "message": "Travel API is running", "docs": "/docs", "health": "/health" }
  ```

#### `GET /health`
- **Description:** Health check endpoint. Returns API status and current timestamp.
- **Response:**
  ```json
  { "status": "healthy", "timestamp": "2025-01-01T12:00:00" }
  ```

---

### Transaction Categorization

#### `POST /categorize`
- **Description:** Categorizes a list of bank transactions by description.
- **Request Body:**
  ```json
  {
    "transactions": [
      { "Description": "string" }
    ]
  }
  ```
- **Response:**
  ```json
  {
    "results": [
      {
        "index": 0,
        "Description": "string",
        "Category": "string",
        "Sub_Category": "string",
        "Category_Confidence": 0.95,
        "Sub_Category_Confidence": 0.92
      }
    ]
  }
  ```

---

### Hotel Search

#### `POST /hotels/search`
- **Description:** Searches for hotels based on user criteria.
- **Request Body:**
  ```json
  {
    "checkin_date": "YYYY-MM-DD",
    "checkout_date": "YYYY-MM-DD",
    "location": "string",
    "adults": 1,
    "children": 0,
    "infants": 0,
    "room_type": "standard",
    "amenities": ["wifi", "breakfast"],
    "fetch_mode": "fallback",
    "limit": 3,
    "debug": false
  }
  ```
- **Response:**
  ```json
  {
    "hotels": [
      {
        "name": "string",
        "price": 120.0,
        "rating": 4.5,
        "url": "string",
        "amenities": ["wifi", "breakfast"]
      }
    ],
    "lowest_price": 100.0,
    "current_price": 120.0
  }
  ```

---

### Flight Search

#### `POST /flights/search`
- **Description:** Searches for flights (one-way or round-trip).
- **Request Body:**
  ```json
  {
    "date": "YYYY-MM-DD",
    "from_airport": "IATA",
    "to_airport": "IATA",
    "trip": "one-way",
    "seat": "economy",
    "adults": 1,
    "children": 0,
    "infants_in_seat": 0,
    "infants_on_lap": 0,
    "fetch_mode": "fallback",
    "return_date": "YYYY-MM-DD"
  }
  ```
- **Response (one-way):**
  ```json
  {
    "flights": [
      {
        "name": "string",
        "departure": "string",
        "arrival": "string",
        "arrival_time_ahead": "+1",
        "duration": "4h 30m",
        "stops": 0,
        "delay": "string",
        "price": 350.0,
        "is_best": true,
        "url": "string"
      }
    ],
    "current_price": "350.0"
  }
  ```
- **Response (round-trip):**
  ```json
  {
    "outbound_flights": [ ... ],
    "return_flights": [ ... ],
    "current_price": "700.0"
  }
  ```

---

### Trip Planning

#### `POST /trip/plan`
- **Description:** Plans a trip, including flights and hotels, and returns the best options and cost breakdown.
- **Request Body:**
  ```json
  {
    "origin": "IATA",
    "destination": "IATA",
    "depart_date": "YYYY-MM-DD",
    "return_date": "YYYY-MM-DD",
    "adults": 1,
    "children": 0,
    "infants": 0,
    "hotel_preferences": {
      "star_rating": 3,
      "max_price_per_night": 150.0,
      "amenities": ["Free Wi-Fi", "Breakfast ($)"]
    },
    "room_type": "standard",
    "amenities": ["wifi", "breakfast"],
    "max_total_budget": 3000.0
  }
  ```
- **Response:**
  ```json
  {
    "best_outbound_flight": { ... },
    "best_return_flight": { ... },
    "best_hotel": { ... },
    "total_estimated_cost": 2500.0,
    "per_person_per_day": 200.0,
    "breakdown": {
      "flight": 1500.0,
      "hotel": 1000.0,
      "nights": 5,
      "adults": 2,
      "children": 0
    },
    "suggestions": "Consider adjusting your dates...",
    "warning": "Partial results due to timeout."
  }
  ```

---

### Airbnb Search

#### `POST /airbnbs/search`
- **Description:** Searches for Airbnbs within a specified area and criteria.
- **Request Body:**
  ```json
  {
    "check_in": "YYYY-MM-DD",
    "check_out": "YYYY-MM-DD",
    "ne_lat": 35.0,
    "ne_long": 139.0,
    "sw_lat": 34.0,
    "sw_long": 138.0,
    "zoom_value": 2,
    "price_min": 0,
    "price_max": 0,
    "place_type": "Entire home/apt",
    "amenities": [1, 2],
    "currency": "USD",
    "language": "en",
    "proxy_url": "",
    "limit": 10
  }
  ```
- **Response:**
  ```json
  {
    "airbnbs": [
      {
        "room_id": 123456,
        "name": "string",
        "title": "string",
        "price": 200.0,
        "per_night": 100.0,
        "url": "string",
        "rating": 4.8,
        "review_count": 50,
        "images": ["url1", "url2"],
        "badges": ["Superhost"],
        "latitude": 35.0,
        "longitude": 139.0
      }
    ],
    "lowest_price": 100.0,
    "current_price": 100.0
  }
  ```

---

### Airbnb Details

#### `POST /airbnbs/details`
- **Description:** Fetches detailed information for a specific Airbnb listing.
- **Request Body:**
  ```json
  {
    "room_id": 123456,
    "currency": "USD",
    "language": "en",
    "proxy_url": ""
  }
  ```
- **Response:**
  ```json
  {
    "name": "string",
    "url": "string",
    "rating": 4.8,
    "amenities": ["Wi-Fi", "Kitchen"],
    "images": ["url1", "url2"],
    "review_count": 50,
    "description": "string",
    "location": { ... },
    "host_name": "string",
    "is_superhost": true
  }
  ```

---

### Itinerary Suggestions

#### `GET /itinerary`
- **Description:** Returns grouped itinerary suggestions (places, events) for a location and date range.
- **Query Parameters:**
  - `lat`, `lng`, `start_date`, `end_date` (required)
  - `radius`, `limit`, `query`, `open_now` (optional)
- **Response:**
  ```json
  {
    "itinerary": {
      "Food": [ ... ],
      "Attractions": [ ... ],
      "Events": [ ... ]
    }
  }
  ```

---

## NLTK Data Setup

This project uses NLTK resources (stopwords, wordnet, omw-1.4). Before running the API, ensure these are downloaded:

```bash
python scripts/download_nltk_data.py
```

If you encounter errors about missing NLTK data, see the troubleshooting section in the code or run the above script again.

---

## Running the Server

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download NLTK data:
   ```bash
   python scripts/download_nltk_data.py
   ```
3. Start the server:
   ```bash
   uvicorn api:app --reload
   ```
4. Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

---

## Project Structure
- `api.py`: Main FastAPI app and endpoints
- `utils/`: Utility modules (data prep, model, etc.)
- `scripts/`: Data processing and training scripts
- `images/structure.png`: Project structure diagram

---

## License
MIT
