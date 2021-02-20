from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

class AsyncDb():
    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase

    def get(self):
        return self.db

    def connect_to_database(self, client_uri: str, client_port: int, db_name: str):
        self.client = AsyncIOMotorClient(client_uri, client_port)
        try:
        # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
        except:
            print("Server not available")
        self.db = self.client[db_name]

    def disconnect_from_database(self):
        self.client.close()