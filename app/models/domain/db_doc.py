from dataclasses import asdict, dataclass
from typing import Dict, Optional
from bson.objectid import ObjectId

@dataclass
class DbDoc():

    _id: Optional[ObjectId]

    def dict_for_db_insertion(self) -> Dict:
        self_dict = asdict(self)
        # If no ID is given during the object creation, the intent is an ID assignment by MongoDB so remove the None field
        if not self._id:
            del self_dict["_id"]
        return self_dict
