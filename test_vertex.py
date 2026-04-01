import os
from google import genai
from google.oauth2 import service_account

# 1. Manually load the credentials from the JSON file
# Use the exact path you provided earlier
key_path = r"C:\Users\vs41798\Downloads\RAG\configs\api_key.json"

# Set the scope so Google knows you want to use the Cloud Platform
scopes = ["https://www.googleapis.com/auth/cloud-platform"]
creds = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)

# 2. Initialize the client with explicit credentials
client = genai.Client(
    vertexai=True,
    project="project-918b3a6f-fe6a-4f28-bdb",
    location="us-central1",
    credentials=creds  # This bypasses the search for "Default Credentials"
)

# 3. Test the connection
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Hello Vertex AI!"
)
print(response.text)
# for model in client.models.list():
#     print(f"Model ID: {model.name}")