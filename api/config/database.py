import os
import uuid
from dotenv import load_dotenv

# MongoDB driver
import asyncio
import motor.motor_asyncio

load_dotenv()

uri = os.getenv('MONGODB_URL')

client = motor.motor_asyncio.AsyncIOMotorClient(uri)
database = client.ihomer_book_recognizer
collection = database.users


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print(f"Ping received by MongoDB, you are successfully connected with ${os.getenv('MONGODB_DB_URL')}")
except Exception as e:
    print(e)

async def create_user():
    document = {'email': 's.jaspers1997@gmail.com', 'first_name': 'Stefan', 'last_name': 'Jaspers'}
    result = await collection.insert_one(document)
    print('result %s' % repr(result.inserted_id))

loop = client.get_io_loop()
loop.run_until_complete(create_user())