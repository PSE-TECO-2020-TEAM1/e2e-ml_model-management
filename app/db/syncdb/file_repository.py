from bson.objectid import ObjectId
from gridfs import GridFS
from pymongo.database import Database

class FileRepository():

    def __init__(self, db: Database):
        self.fs = GridFS(db)
    
    def get_file(self, id: ObjectId) -> bytes:
        return self.fs.get(id).read()

    def put_file(self, file: bytes) -> ObjectId:
        return self.fs.put(file)

    def delete_file(self, id: ObjectId):
        self.fs.delete(id)

    def replace_file(self, old_id: ObjectId, new_file: bytes) -> ObjectId:
        self.delete_file(old_id)
        return self.put_file(new_file)