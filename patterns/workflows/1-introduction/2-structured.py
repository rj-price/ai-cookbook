import os
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

current_datetime = datetime.now()
current_date = current_datetime.strftime("%A, %d/%m/%Y")

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------


class CalendarEvent(BaseModel):
    name: str
    day: str
    date: str
    participants: list[str]


# --------------------------------------------------------------
# Step 2: Call the model
# --------------------------------------------------------------

system_prompt = f"""
Extract the event information. 
Return the name, day, date (DD/MM/YYYY), and attendees.
Today's date is {current_date}.
"""

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Zoe and I are going to the beach a week on Saturday.",
        },
    ],
    response_format=CalendarEvent,
)

# --------------------------------------------------------------
# Step 3: Parse the response
# --------------------------------------------------------------

event = completion.choices[0].message.parsed
print(f"Name: {event.name}")
print(f"Day: {event.day}")
print(f"Date: {event.date}")
print(f"Attendees: {event.participants}")