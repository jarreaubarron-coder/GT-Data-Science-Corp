import pandas as pd
import requests

#Define the SerpApi Endpoint (Google Search Engine)
url = "https://serpapi.com/search?engine=google_jobs"

#Package up the parameters (The query details and API key)
#Replace the api_key value with your own personal key from their dashboard
query_parameters = {
   "engine": "google_jobs",
    "q": "Entry Level",
    "location": "United States",
    "hl": "en",
    "gl": "us",
    "google_domain": "google.com",
    "api_key": "5540ae778b06f6b56c457fb3aaafecef068ad5a1b3c5cd42f4b27d0d7acb5d00"}

#Ping the API with the parameters attached
response = requests.get(url, params=query_parameters)

#Check connection and parse the nested payload
if response.status_code == 200:
    raw_data = response.json()

    #SerpApi returns a nested dictionary
    #For a general search, it's called 'organic_results'
    results_list = raw_data.get("jobs_results", [])

    #Convert the list of results into a clean DataFrame
    df = pd.DataFrame(results_list)

    print("Google search data pulled from SerpApi.")
    print(df.head())

    # Export as a csv for further cleaning and EDA. Change the file name if necessary
    df.to_csv("serpapi_raw_data.csv", index=False)

else:
    print(f"API call failed. Status Code: {response.status_code}")
    print(response.text)