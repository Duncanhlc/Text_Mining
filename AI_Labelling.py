import pandas as pd
import json
import xml.etree.ElementTree as ET
import google.genai as genai
from google.genai import types

# Load the .xml
tree = ET.parse("../Prompt.xml")
root = tree.getroot()
prompt = ET.tostring(root, encoding='unicode', method='text').strip()

# Load the .csv
df = pd.read_csv("Topic.csv")
topic = []

for topic_id, group in df.groupby('Topic'):
    group = group.sort_values(by='Rank')
    word = group['Word'].tolist()
    word_str = ", ".join(word)

    topic.append({
        "Topic_id": int(topic_id),
        "Word": word_str
    })

print(f"Total topics loaded: {len(topic)}")

# Call gemini
client = genai.Client(api_key="AIzaSyAqrBdaL3-GBKBgtpS2-yd-hdRV5YrO-3I")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"{prompt}\n\nDATA:\n{json.dumps(topic)}",
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
    )
)

# Save the result
try:
    result = response.parsed if response.parsed else json.loads(response.text)

    with open("Topic_Labelled.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Success! saved Topic_Labelled.json")
except Exception as e:
    print(f"Error parsing response: {e}")
    print("Raw Response:", response.text)