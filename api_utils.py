import streamlit as st
import json
import requests
import time
import os 

API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCFGKXUPAfTKA97PURI4agLQI7pC4YpMLU")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def call_gemini_api_with_retry(payload, max_retries=3):
    """Handles API call with exponential backoff for resilience."""
    
    if not API_KEY or "YOUR_API_KEY_HERE" in API_KEY:
        return {"error": "Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable or Streamlit secret."}

    url = f"{GEMINI_API_URL}?key={API_KEY}"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return {"error": f"API HTTP Error: {e} - {response.text}"}
        except Exception as e:
            return {"error": f"API Connection Error: {e}"}
    return {"error": "API failed after multiple retries due to rate limiting or other issues."}

def call_gemini_api_for_suggestions(predicted_1y, debt_per, equity_per, rating):
    """
    Calls the Gemini API using Google Search grounding to suggest real-world funds
    based on predicted performance and risk profile.
    """

    system_prompt = (
        "You are a helpful and professional financial analyst. "
        "Your goal is to search the web for three real-world mutual funds (Name, AMC, Category) "
        "that have a recent 1-year return that meets or exceeds the target return provided by the user. "
        "The suggested funds should match the approximate risk profile (Debt/Equity mix) as closely as possible. "
        "Provide your suggestions in a clear, formatted list, followed by a brief summary of why these funds were chosen."
    )

    user_query = (
        f"Find three highly-rated mutual funds (Rating >= {rating}) with recent 1-year returns "
        f"greater than or equal to {predicted_1y:,.2f}%. "
        f"The fund's portfolio should ideally have an equity exposure of around {equity_per:,.0f}% and a debt exposure of around {debt_per:,.0f}%. "
        f"List the Fund Name, Asset Management Company (AMC), and its current 1-year return."
    )

    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "tools": [{ "google_search": {} }],
        "generationConfig": { # FIXED: Renamed 'config' to 'generationConfig'
            "temperature": 0.5 
        },
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
    }

    result = call_gemini_api_with_retry(payload)

    if 'error' in result:
        return result

    candidate = result.get('candidates', [None])[0]
    
    if not candidate or not candidate.get('content', {}).get('parts', [{}])[0].get('text'):
        return {"error": "API returned an empty or malformed response."}

    text = candidate['content']['parts'][0]['text']

    # Extract grounding sources
    sources = []
    grounding_metadata = candidate.get('groundingMetadata')
    if grounding_metadata and grounding_metadata.get('groundingAttributions'):
        sources = [
            {'uri': attr['web']['uri'], 'title': attr['web']['title']}
            for attr in grounding_metadata['groundingAttributions']
            if attr.get('web') and attr['web'].get('uri') and attr['web'].get('title')
        ]
        
    return {"text": text, "sources": sources}