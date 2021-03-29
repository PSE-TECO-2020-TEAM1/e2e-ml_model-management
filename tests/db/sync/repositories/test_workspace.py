from app.models.domain.training_data_set import TrainingDataSet
from bson.objectid import ObjectId
import mongomock
from app.db.sync.repositories.workspace import WorkspaceRepository

def test_set_training_data_set():
    workspace_id = ObjectId()
    db = mongomock.MongoClient()
    repo = WorkspaceRepository(db)
    repo.set_training_data_set(workspace_id=workspace_id, new_data_set=)
    db["workspaces"]

