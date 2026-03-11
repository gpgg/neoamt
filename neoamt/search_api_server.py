import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import requests
import numpy as np
from tqdm import tqdm
from urllib.parse import urlencode
import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import langid

class QueryRequest(BaseModel):
    query: str
    timeout: int
    zone: str
    api_key: str
    location: str


app = FastAPI()

@app.post("/bing_search")
async def bing_search_endpoint(request: QueryRequest):
    print("Received request:", request)
    query = request.query
    timeout = request.timeout
    zone = request.zone
    api_key = request.api_key
    location = request.location
    
    
    lang_code, lang_confidence = langid.classify(query)
    if lang_code == 'zh':
        mkt, setLang = "zh-CN", "zh"
    else:
        mkt, setLang = "en-US", "en"
    
    # Prepare URL with query parameters
    encoded_query = urlencode({
        "q": query, 
        "mkt": mkt, 
        "setLang": setLang
    })
    target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={location}"

    # Prepare headers and payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "zone": zone,
        "url": target_url,
        "format": "raw"
    }
    print("Sending request to Bing Search API...")
    print("Payload:", payload)

    # Send request
    # resp = requests.post(
    #     "https://api.brightdata.com/request",
    #     headers=headers,
    #     json=payload,
    #     timeout=timeout
    # )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.brightdata.com/request",
            headers=headers,
            json=payload
        )
    # print(resp.text)
    return {"result": resp.text}

if __name__ == "__main__":
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8001
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=16)