from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import TransformComponent, BaseNode
from typing import Sequence
from pathlib import Path

class DocTagging(TransformComponent, BaseModel):
    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        for node in nodes:
            source = node.metadata.get("source", "")
            fileType = node.metadata.get("fileType", "")
            

            node.metadata["folder"] = Path(source).parent.name if source else ""
            node.excluded_llm_metadata_keys = ["source", "fileType",  "folder"]
            node.excluded_embed_metadata_keys = ["source", "fileType", "folder"]

        return nodes