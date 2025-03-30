import lancedb
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lrg1_db"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("docling")


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

query = input("Enter your search query: ")

result = table.search(query=query, query_type="vector").limit(5)
print(result.to_pandas())
