"""
utils/airports.py

Contains the CITY_TO_IATA dictionary mapping major cities and countries to their main IATA airport codes.
"""

import csv
import requests
from functools import lru_cache
import difflib
from math import radians, sin, cos, sqrt, atan2

OPENFLIGHTS_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

CITY_TO_IATA = {
    # --- United States (multi-airport example) ---
    "new york": ["JFK", "EWR", "LGA"],
    "nyc": ["JFK", "EWR", "LGA"],
    "los angeles": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "chicago": ["ORD", "MDW"],
    "washington": ["IAD", "DCA", "BWI"],
    "san francisco": ["SFO", "OAK", "SJC"],
    "houston": ["IAH", "HOU"],
    "dallas": ["DFW", "DAL"],
    # --- United Kingdom ---
    "london": ["LHR", "LGW", "STN", "LTN", "LCY", "SEN"],
    "london heathrow": "LHR",
    "london gatwick": "LGW",
    # --- Japan ---
    "tokyo": ["HND", "NRT"],
    # --- Russia ---
    "moscow": ["SVO", "DME", "VKO"],
    # --- Brazil ---
    "rio de janeiro": ["GIG", "SDU"],
    "sao paulo": ["GRU", "CGH", "VCP"],
    # --- South Africa ---
    "johannesburg": ["JNB"],
    "cape town": ["CPT"],
    # --- Australia ---
    "sydney": ["SYD"],
    "melbourne": ["MEL", "AVV"],
    # --- Canada ---
    "toronto": ["YYZ", "YTZ"],
    # --- Ambiguous/Disambiguated ---
    "san jose, costa rica": "SJO",
    "san jose, california": "SJC",
    # --- Caribbean & Central America ---
    "nassau": "NAS",
    "kingston": "KIN",
    "port of spain": "POS",
    # --- Africa (expansion) ---
    "algiers": "ALG",
    "luanda": "LAD",
    "kinshasa": "FIH",
    "khartoum": "KRT",
    "kigali": "KGL",
    "kampala": "EBB",
    "harare": "HRE",
    "maputo": "MPM",
    # --- Middle East (expansion) ---
    "riyadh": "RUH",
    "jeddah": "JED",
    "amman": "AMM",
    "beirut": "BEY",
    "muscat": "MCT",
    # --- Asia (expansion) ---
    "jakarta": ["CGK", "HLP"],
    "bangkok": ["BKK", "DMK"],
    "seoul": ["ICN", "GMP"],
    "shanghai": ["PVG", "SHA"],
    # --- Country-level fallbacks (expansion) ---
    "saudi arabia": "RUH",
    "indonesia": "CGK",
    "south korea": "ICN",
    "thailand": "BKK",
    "malaysia": "KUL",
    "philippines": "MNL",
    "vietnam": "SGN",
    # --- Existing entries (keep all previous mappings below) ---
    # --- North America ---
    # United States
    "atlanta": "ATL",
    "baltimore": "BWI",
    "boston": "BOS",
    "charlotte": "CLT",
    "denver": "DEN",
    "detroit": "DTW",
    "fort lauderdale": "FLL",
    "honolulu": "HNL",
    "las vegas": "LAS",
    "minneapolis": "MSP",
    "newark": "EWR",
    "orlando": "MCO",
    "philadelphia": "PHL",
    "phoenix": "PHX",
    "portland": "PDX",
    "san diego": "SAN",
    "tampa": "TPA",
    # Canada
    "calgary": "YYC",
    "edmonton": "YEG",
    "halifax": "YHZ",
    "montreal": "YUL",
    "ottawa": "YOW",
    "vancouver": "YVR",
    "winnipeg": "YWG",
    # Mexico
    "cancun": "CUN",
    "guadalajara": "GDL",
    "mexico city": "MEX",
    # --- South America ---
    # Argentina
    "buenos aires": "EZE",
    # Brazil
    "brasilia": "BSB",
    "recife": "REC",
    "salvador": "SSA",
    # Chile
    "antofagasta": "ANF",
    "santiago": "SCL",
    # Colombia
    "bogota": "BOG",
    "medellin": "MDE",
    # Peru
    "lima": "LIM",
    # Uruguay
    "montevideo": "MVD",
    # Venezuela
    "caracas": "CCS",
    # Ecuador
    "quito": "UIO",
    # --- Europe ---
    # Austria
    "vienna": "VIE",
    # Belgium
    "brussels": "BRU",
    # Czech Republic
    "prague": "PRG",
    # Denmark
    "copenhagen": "CPH",
    # Finland
    "helsinki": "HEL",
    # France
    "lyon": "LYS",
    "nice": "NCE",
    "orly": "ORY",
    "paris": "CDG",
    # Germany
    "berlin": "BER",
    "frankfurt": "FRA",
    "hamburg": "HAM",
    "munich": "MUC",
    # Greece
    "athens": "ATH",
    # Hungary
    "budapest": "BUD",
    # Ireland
    "dublin": "DUB",
    # Italy
    "milan": "MXP",
    "naples": "NAP",
    "rome": "FCO",
    "venice": "VCE",
    # Netherlands
    "amsterdam": "AMS",
    # Norway
    "oslo": "OSL",
    # Poland
    "krakow": "KRK",
    "warsaw": "WAW",
    # Portugal
    "lisbon": "LIS",
    "porto": "OPO",
    # Russia
    "domodedovo": "DME",
    "vnukovo": "VKO",
    # Spain
    "barcelona": "BCN",
    "madrid": "MAD",
    "malaga": "AGP",
    "palma de mallorca": "PMI",
    # Sweden
    "stockholm": "ARN",
    # Switzerland
    "basel": "BSL",
    "geneva": "GVA",
    "zurich": "ZRH",
    # Turkey
    "istanbul": "IST",
    # United Kingdom
    "edinburgh": "EDI",
    "gatwick": "LGW",
    "manchester": "MAN",
    "stansted": "STN",
    # --- Asia ---
    # China
    "beijing": "PEK",
    "chengdu": "CTU",
    "guangzhou": "CAN",
    "kunming": "KMG",
    "shanghai": "PVG",
    "shenzhen": "SZX",
    "xian": "XIY",
    # Hong Kong
    "hong kong": "HKG",
    # India
    "bangalore": "BLR",
    "chennai": "MAA",
    "delhi": "DEL",
    "hyderabad": "HYD",
    "kolkata": "CCU",
    "mumbai": "BOM",
    # Indonesia
    "bali": "DPS",
    "denpasar": "DPS",
    "jakarta": "CGK",
    "surabaya": "SUB",
    # Japan
    "fukuoka": "FUK",
    "narita": "NRT",
    "osaka": "KIX",
    "sapporo": "CTS",
    "tokyo": "HND",
    # Malaysia
    "penang": "PEN",
    # Pakistan
    "islamabad": "ISB",
    "karachi": "KHI",
    "lahore": "LHE",
    # Philippines
    "cebu": "CEB",
    "davao": "DVO",
    "manila": "MNL",
    # Qatar
    "doha": "DOH",
    # Singapore
    "singapore": "SIN",
    # South Korea
    "gimpo": "GMP",
    "seoul": "ICN",
    # Taiwan
    "taipei": "TPE",
    # Thailand
    "phuket": "HKT",
    # UAE
    "abu dhabi": "AUH",
    "dubai": "DXB",
    "sharjah": "SHJ",
    # Vietnam
    "hanoi": "HAN",
    "ho chi minh city": "SGN",
    # --- Oceania ---
    # Australia
    "adelaide": "ADL",
    "brisbane": "BNE",
    "cairns": "CNS",
    "gold coast": "OOL",
    "perth": "PER",
    # New Zealand
    "auckland": "AKL",
    "christchurch": "CHC",
    "queenstown": "ZQN",
    "wellington": "WLG",
    # --- Africa ---
    # Egypt
    "cairo": "CAI",
    "sharm el sheikh": "SSH",
    # Ethiopia
    "addis ababa": "ADD",
    # Ghana
    "accra": "ACC",
    # Kenya
    "mombasa": "MBA",
    "nairobi": "NBO",
    # Morocco
    "casablanca": "CMN",
    "marrakesh": "RAK",
    # Nigeria
    "abuja": "ABV",
    "lagos": "LOS",
    "port harcourt": "PHC",
    # Senegal
    "dakar": "DSS",
    # South Africa
    "johannesburg": "JNB",
    # --- Middle East ---
    # Iran
    "mashhad": "MHD",
    "tehran": "IKA",
    # Iraq
    "baghdad": "BGW",
    # Israel
    "eilat": "ETM",
    "tel aviv": "TLV",
    # Jordan
    "amman": "AMM",
    # Kuwait
    "kuwait city": "KWI",
    # Lebanon
    "beirut": "BEY",
    # Oman
    "muscat": "MCT",
    # Qatar
    "dammam": "DMM",
    "jeddah": "JED",
    "riyadh": "RUH",
    # --- Country-level fallbacks ---
    "united states": "JFK",
    "canada": "YYZ",
    "brazil": "GRU",
    "argentina": "EZE",
    "germany": "FRA",
    "france": "CDG",
    "italy": "FCO",
    "spain": "MAD",
    "united kingdom": "LHR",
    "japan": "HND",
    "china": "PEK",
    "india": "DEL",
    "australia": "SYD",
    "new zealand": "AKL",
    "south africa": "JNB",
    "egypt": "CAI",
    "turkey": "IST",
    "russia": "SVO",
    "uae": "DXB",
    "qatar": "DOH",
}

def build_global_city_to_iata():
    """
    Download and parse OpenFlights airports.dat to build a global city+country to IATA mapping.
    Returns: dict[(city, country)] -> [IATA1, IATA2, ...]
    """
    city_to_iata = {}
    try:
        resp = requests.get(OPENFLIGHTS_AIRPORTS_URL, timeout=20)
        resp.raise_for_status()
        reader = csv.reader(resp.text.splitlines())
        for row in reader:
            # Columns: ID, Name, City, Country, IATA, ICAO, Lat, Lon, Alt, TZ, DST, Tz, Type, Source
            if len(row) < 5:
                continue
            city = row[2].strip().lower()
            country = row[3].strip().lower()
            iata = row[4].strip().upper()
            if not iata or iata == "\\N" or len(iata) != 3:
                continue
            key = (city, country)
            city_to_iata.setdefault(key, []).append(iata)
    except Exception as e:
        # If download fails, fallback to empty mapping
        print(f"Warning: Could not load OpenFlights airports.dat: {e}")
    return city_to_iata

@lru_cache(maxsize=1)
def get_global_city_to_iata():
    return build_global_city_to_iata()

def fuzzy_city_match(city, all_cities, cutoff=0.85):
    """Return the best fuzzy match for city from all_cities, or None if not close enough."""
    matches = difflib.get_close_matches(city, all_cities, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def lookup_iata(location, country_hint=None, fuzzy=True, prefer_international=True):
    """
    Lookup IATA code(s) for a city, city+country, or country.
    1. Try static CITY_TO_IATA mapping (for aliases, multi-airport cities, special cases).
    2. Then try OpenFlights global mapping for full coverage, with fuzzy matching and prioritizing international airports.
    Returns a list of IATA codes or a single code, or None if not found.
    """
    print(f"[lookup_iata] location={location!r}, country_hint={country_hint!r}, fuzzy={fuzzy}, prefer_international={prefer_international}")
    print(f"[lookup_iata] CITY_TO_IATA: {CITY_TO_IATA}")
    if not location:
        print("[lookup_iata] No location provided.")
        return None
    key = location.strip().lower()
    # 1. Try static mapping (with country disambiguation if provided)
    if country_hint:
        key_with_country = f"{key}, {country_hint.strip().lower()}"
        if key_with_country in CITY_TO_IATA:
            print(f"[lookup_iata] Found in static mapping with country: {CITY_TO_IATA[key_with_country]}")
            return CITY_TO_IATA[key_with_country]
    if key in CITY_TO_IATA:
        print(f"[lookup_iata] Found in static mapping: {CITY_TO_IATA[key]}")
        return CITY_TO_IATA[key]
    if country_hint and country_hint.strip().lower() in CITY_TO_IATA:
        print(f"[lookup_iata] Found country fallback in static mapping: {CITY_TO_IATA[country_hint.strip().lower()]}")
        return CITY_TO_IATA[country_hint.strip().lower()]
    # 2. Try dynamic OpenFlights mapping
    global_map = get_global_city_to_iata()
    city = key
    country = country_hint.strip().lower() if country_hint else None
    all_cities = set(c for (c, _) in global_map.keys())
    if fuzzy and city not in all_cities:
        fuzzy_city = fuzzy_city_match(city, all_cities)
        print(f"[lookup_iata] Fuzzy match for '{city}': {fuzzy_city}")
        if fuzzy_city:
            city = fuzzy_city
    # Try city+country
    if country:
        gkey = (city, country)
        if gkey in global_map:
            codes = global_map[gkey]
            if prefer_international:
                codes = prioritize_international(codes, city, country)
            print(f"[lookup_iata] Found in OpenFlights city+country: {codes}")
            return codes
    # fallback: try all cities matching the name (may be ambiguous)
    matches = [codes for (c, _), codes in global_map.items() if c == city]
    if matches:
        codes = [iata for sublist in matches for iata in sublist]
        if prefer_international:
            codes = prioritize_international(codes, city, country)
        print(f"[lookup_iata] Found in OpenFlights city-only: {codes}")
        return codes if codes else None
    print(f"[lookup_iata] No match found for location={location!r}, country_hint={country_hint!r}")
    return None

def prioritize_international(iata_codes, city, country):
    """Return a list of IATA codes, prioritizing those with 'International' in the airport name."""
    airports = get_airports_by_iata(iata_codes)
    intl = [a['code'] for a in airports if 'international' in a['name'].lower()]
    non_intl = [a['code'] for a in airports if 'international' not in a['name'].lower()]
    return intl + non_intl

def get_airports_by_iata(iata_codes):
    """Return a list of airport dicts for the given IATA codes from OpenFlights."""
    airports = []
    try:
        resp = requests.get(OPENFLIGHTS_AIRPORTS_URL, timeout=20)
        resp.raise_for_status()
        reader = csv.reader(resp.text.splitlines())
        for row in reader:
            if len(row) < 5:
                continue
            iata = row[4].strip().upper()
            if iata in iata_codes:
                airports.append({
                    'code': iata,
                    'name': row[1],
                    'city': row[2],
                    'country': row[3],
                    'latitude': row[6],
                    'longitude': row[7],
                })
    except Exception as e:
        print(f"Warning: Could not load OpenFlights airports.dat for airport details: {e}")
    return airports

# --- Geocoding and Nearest Airport Fallback ---

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def geocode_city(city_name):
    url = "https://overpass-api.de/api/interpreter"
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

def get_airports():
    """Fetch and cache the list of airports from OpenFlights."""
    resp = requests.get(OPENFLIGHTS_AIRPORTS_URL)
    resp.raise_for_status()
    f = resp.text.splitlines()
    reader = csv.DictReader(f, fieldnames=["id","name","city","country","iata","icao","lat","lon","alt","tz","dst","tzdb","type","source"])
    airports = [row for row in reader]
    return airports

def resolve_to_iata(location, country_hint=None):
    print(f"[resolve_to_iata] Resolving: location={location!r}, country_hint={country_hint!r}")
    # If already a 3-letter IATA code, validate it exists in OpenFlights
    if isinstance(location, str) and len(location) == 3 and location.isalpha():
        iata = location.upper()
        global_map = get_global_city_to_iata()
        all_iatas = {iata for codes in global_map.values() for iata in codes}
        if iata in all_iatas:
            print(f"[resolve_to_iata] Input is valid IATA code: {iata}")
            return iata
        else:
            print(f"[resolve_to_iata] Input looks like IATA but not found: {iata}")
    codes = lookup_iata(location, country_hint=country_hint, fuzzy=True, prefer_international=True)
    print(f"[resolve_to_iata] lookup_iata result: {codes}")
    if codes:
        if isinstance(codes, list):
            print(f"[resolve_to_iata] Multiple IATA codes found for {location!r}: {codes}. Returning first: {codes[0]}")
            return codes[0]
        print(f"[resolve_to_iata] Returning code: {codes}")
        return codes
    print(f"[resolve_to_iata] No IATA code found for {location!r}")
    return None

    coords = geocode_city(location)
    print(f"[resolve_to_iata] geocode_city result: {coords}")
    if not coords:
        print(f"[resolve_to_iata] Could not geocode location: {location!r}")
        return None
    lat, lng = coords
    airports = get_airports()
    min_dist = float("inf")
    nearest = None
    preferred = []
    for airport in airports:
        iata = airport.get("iata")
        try:
            airport_lat = float(airport["lat"])
            airport_lng = float(airport["lon"])
        except Exception:
            continue
        if not iata or iata == "\\N" or len(iata) != 3:
            continue
        dist = haversine(lat, lng, airport_lat, airport_lng)
        name = airport.get("name", "").lower()
        city = airport.get("city", "").lower()
        if "international" in name and dist < 50:
            preferred.append((airport, dist))
        elif "regional" in name and dist < 50:
            preferred.append((airport, dist + 10))
        elif city and location.strip().lower() in city and dist < 100:
            preferred.append((airport, dist + 20))
        if dist < min_dist:
            min_dist = dist
            nearest = airport
    if preferred:
        preferred.sort(key=lambda x: x[1])
        nearest = preferred[0][0]
    print(f"[resolve_to_iata] Nearest airport: {nearest['iata'] if nearest else None}")
    return nearest["iata"] if nearest else None 