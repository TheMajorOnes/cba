from ollama import chat, Client
from pydantic import BaseModel
from pprint import pprint

class Identity(BaseModel):
    first_name: str
    last_name: str
    age: int
    passport_number: str

class Person(BaseModel):
    real_first_name: str
    real_last_name: str
    real_age: int
    list_of_identities: list[Identity]

client = Client(host='https://ollama.themajorones.dev')

response = client.chat(
    messages=[
        {
            'role': 'user',
            'content': 'generate a person with 3 fake identities.',
        }
    ],
    model='llama3.2',
    format=Person.model_json_schema(),
)

people = Person.model_validate_json(response.message.content)
pprint(people.model_dump(), sort_dicts=False)