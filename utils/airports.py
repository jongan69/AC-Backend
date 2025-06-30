"""
utils/airports.py

Contains the CITY_TO_IATA dictionary mapping major cities and countries to their main IATA airport codes.
"""

CITY_TO_IATA = {
    # --- North America ---
    # United States
    "atlanta": "ATL",
    "baltimore": "BWI",
    "boston": "BOS",
    "charlotte": "CLT",
    "chicago": "ORD",
    "dallas": "DFW",
    "denver": "DEN",
    "detroit": "DTW",
    "fort lauderdale": "FLL",
    "honolulu": "HNL",
    "houston": "IAH",
    "las vegas": "LAS",
    "los angeles": "LAX",
    "miami": "MIA",
    "minneapolis": "MSP",
    "new york": "JFK",
    "newark": "EWR",
    "orlando": "MCO",
    "philadelphia": "PHL",
    "phoenix": "PHX",
    "portland": "PDX",
    "san diego": "SAN",
    "san francisco": "SFO",
    "san jose": "SJC",
    "seattle": "SEA",
    "tampa": "TPA",
    "washington": "IAD",
    # Canada
    "calgary": "YYC",
    "edmonton": "YEG",
    "halifax": "YHZ",
    "montreal": "YUL",
    "ottawa": "YOW",
    "toronto": "YYZ",
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
    "rio de janeiro": "GIG",
    "salvador": "SSA",
    "sao paulo": "GRU",
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
    "moscow": "SVO",
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
    "london": "LHR",
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
    "kuala lumpur": "KUL",
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
    "bangkok": "BKK",
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
    "melbourne": "MEL",
    "perth": "PER",
    "sydney": "SYD",
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
    "cape town": "CPT",
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
    # Turkey
    "istanbul": "IST",
    # UAE
    "abu dhabi": "AUH",
    "dubai": "DXB",
    "sharjah": "SHJ",
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