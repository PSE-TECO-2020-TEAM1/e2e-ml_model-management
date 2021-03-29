from abc import ABC, abstractstaticmethod
from app.models.domain.mongo_model import OID

class ExternalDataSource(ABC):

    @abstractstaticmethod
    def last_modified(workspace_id: OID) -> int:
        pass

    @abstractstaticmethod
    def fetch_labels(workspace_id: OID):
        pass

    @abstractstaticmethod
    def fetch_samples(workspace_id: OID):
        pass