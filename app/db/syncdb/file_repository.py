from bson.objectid import ObjectId
from gridfs import GridFS
from pymongo.database import Database
from app.db.error.non_existent_error import NonExistentError

class FileRepository():

    def __init__(self, db: Database):
        self.fs = GridFS(db)
    
    def get_file(self, id: ObjectId) -> bytes:
        if not self.fs.exists(id):
            raise NonExistentError("The request file with the id " + str(id) + " does not exist")
        return self.fs.get(id).read()

    def put_file(self, file: bytes) -> ObjectId:
        return self.fs.put(file)

    def delete_file(self, id: ObjectId):
        self.fs.delete(id)

    def replace_file(self, old_id: ObjectId, new_file: bytes) -> ObjectId:
        self.delete_file(old_id)
        return self.put_file(new_file)