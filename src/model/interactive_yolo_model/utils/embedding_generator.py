import time
import torch
from typing import List, Dict, Tuple, Any
from database_service.database_service import DatabaseServices
from interactive_yolo_interfaces.msg import DatabaseCategoryInfo
from tensor_msg_conversion import float32TensorToTorchTensor

class EmbeddingGenerator:
    def __init__(self, database_service: DatabaseServices, ):
        self._database_service = database_service
        self._embedding_dict:Dict[List[torch.Tensor]] = dict()
        self._last_update_time = 0.0

    def update(self) -> bool:
        """
        Update the embeddings from the database.
        """

        update_time = time.time()
        embedding_updated = False


        print("get all categories")
        categories_infos_result= self._database_service.GetAllDatabaseCategories()
        if categories_infos_result is None:
            print("No access to database")
            return False
        
        categories_infos:List[DatabaseCategoryInfo] = categories_infos_result.infos


        print("process categories")
        for category_info in categories_infos:

            if category_info.embeddings_set_time == 0.0:
                continue

            update_category = False

            if category_info.name not in self._embedding_dict.keys():
                update_category = True

            elif category_info.embeddings_set_time > self._last_update_time:
                update_category = True

            if update_category:
                pe_list = []
                for pe_msg in category_info.embeddings:
                    pe = float32TensorToTorchTensor(pe_msg)
                    pe_list.append(pe)
                self._embedding_dict[category_info.name] = pe_list
                embedding_updated = True
        
        self._last_update_time = update_time
        return embedding_updated

    def get_embedding(self, categories_name: List[str] = None, fallback_model:Any = None)->Tuple[torch.Tensor, List[str], Dict[str, str]]:
        """
        Get the embeddings by categories names.

        returns embeddings, alias_name, alias_to_categories_name
        """

        if categories_name is None:
            if len(self._embedding_dict.keys()) == 0:
               categories_name = ["personne",]
            else:
                categories_name = list(self._embedding_dict.keys())

        if len(categories_name) == 0:
            if len(self._embedding_dict.keys()) == 0:
               categories_name = ["personne",]
            else:
                categories_name = list(self._embedding_dict.keys())

        pe_list = []
        alias_name_list = []
        category_alias_to_name = dict()

        for category_name in categories_name:
            cluster_available = False

            if category_name in self._embedding_dict.keys():
                if len(self._embedding_dict[category_name]) > 0:
                    cluster_available = True

            if not cluster_available:
                if fallback_model is not None:

                    alias_name = "__CLUSTER0__" + category_name
                    vpe = fallback_model.get_text_pe([category_name,]).cpu()

                    alias_name_list.append(alias_name)
                    pe_list.append(vpe)

                    category_alias_to_name[alias_name] = category_name

                else:
                    Exception(f"Category {category_name} not found and no fallback model available.")
            else:
                itt = 0
                for pe in self._embedding_dict[category_name]:

                    alias_name = "__CLUSTER" + str(itt) + "__" + category_name
                    itt += 1

                    alias_name_list.append(alias_name)
                    pe_list.append(pe)

                    category_alias_to_name[alias_name] = category_name

        return torch.cat(pe_list, dim=1), alias_name_list, category_alias_to_name


