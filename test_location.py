#!/usr/bin/env python3
"""Test script for location resolution endpoint"""

import requests
import json

def test_location_resolution():
    """Test the location resolution endpoint"""
    
    # Test data
    test_cities = ["Tokyo", "Paris", "London", "New York", "Los Angeles"]
    
    print("Testing Location Resolution Endpoint")
    print("=" * 50)
    
    for city in test_cities:
        print(f"\nTesting city: {city}")
        
        try:
            response = requests.post(
                "http://localhost:8000/resolve/location",
                json={"city": city},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS: Found {len(data['airports'])} airports")
                print(f"   Coordinates: {data['coordinates']}")
                print(f"   Country: {data['country']}")
                print(f"   Airports: {[airport['iata'] for airport in data['airports'][:3]]}")
            else:
                print(f"❌ ERROR: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Could not connect to server. Is it running?")
            break
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_location_resolution() 