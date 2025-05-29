import requests

# Query for recommendations
response = requests.post("http://localhost:8000/query", 
                        json={"question": "Best PC setup for video editing under $2000"})

print(response)
