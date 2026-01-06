# https://drive.google.com/file/d/18Xm6Nr8QIUo7lgijkP998PQV355dVorH/view?usp=drive_link
# https://drive.google.com/uc?export=download&id=18Xm6Nr8QIUo7lgijkP998PQV355dVorH
import pathway as pw
from pathway.xpacks.llm import parsers
import requests  # For downloading
import os

drive_link = "https://drive.google.com/uc?export=download&id=18Xm6Nr8QIUo7lgijkP998PQV355dVorH"
local_path = os.path.expanduser("~/Books/The Count of Monte Cristo.txt")  # Local save path (adjust as needed)
os.makedirs(os.path.dirname(local_path), exist_ok=True)

response = requests.get(drive_link)
with open(local_path, 'wb') as f:
    f.write(response.content)

# Step 2: Ingestion - Read and parse the downloaded file
parser = parsers.Utf8Parser()

file_table = pw.io.fs.read(
    local_path,
    format="binary",
    mode="static"
)

parsed_table = parser(file_table)

# Step 3: Transformation (simple)
parsed_table = parsed_table.select(
    content=pw.this.text,
    metadata=pw.this.metadata,
)

# Step 4: Output to CSV
pw.io.csv.write(parsed_table, './graph-import/')

# Run
pw.run(monitoring_level=pw.MonitoringLevel.DEBUG)