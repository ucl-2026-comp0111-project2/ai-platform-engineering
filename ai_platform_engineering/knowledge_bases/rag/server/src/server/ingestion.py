import asyncio
import hashlib
import os
import time
import json
import traceback
from typing import List, Optional, Tuple
from common import utils

from common.multimodal_embeddings import UnsupportedImageFormatError

from langchain_core.documents import Document
from common.models.rag import DocumentChunkMetadata, DocumentMetadata, StructuredEntity
from langchain_milvus import Milvus
from common.graph_db.base import GraphDB
from common.job_manager import JobManager
from common.constants import INGESTOR_ID_KEY, DATASOURCE_ID_KEY, LAST_UPDATED_KEY, FRESH_UNTIL_KEY, PARENT_ENTITY_PK_KEY, PARENT_ENTITY_TYPE_KEY, SUB_ENTITY_INDEX_KEY, SUB_ENTITY_LABEL, ENTITY_TYPE_KEY
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Max images embedded per document, to cap API calls on image-heavy pages
MAX_IMAGES_PER_DOCUMENT = int(os.getenv("MAX_IMAGES_PER_DOCUMENT", "20"))
# Max concurrent image embedding requests (network-bound, safe to parallelize)
IMAGE_EMBED_CONCURRENCY = int(os.getenv("IMAGE_EMBED_CONCURRENCY", "5"))
# Max attempts per image before giving up (covers transient network/API errors)
IMAGE_EMBED_MAX_ATTEMPTS = int(os.getenv("IMAGE_EMBED_MAX_ATTEMPTS", "3"))
IMAGE_EMBED_RETRY_DELAY_SECONDS = float(os.getenv("IMAGE_EMBED_RETRY_DELAY_SECONDS", "1.0"))

class DocumentProcessor:
  # Milvus varchar field limit (65535 bytes, using 60000 to be safe with UTF-8 encoding)
  MILVUS_MAX_VARCHAR_LENGTH = 60000

  def __init__(self, vstore: Milvus, job_manager: JobManager, graph_rag_enabled: bool, data_graph_db: Optional[GraphDB] = None, max_property_length: int = 250, batch_size: int = 1000, image_vstore: Optional[Milvus] = None):
    """
    Args:
        vstore: Milvus instance
        job_manager: JobManager instance
        graph_rag_enabled: Whether graph RAG is enabled
        data_graph_db: GraphDB instance
        max_property_length: Maximum length for property values for structured entities
        batch_size: Batch size for ingestion into vector database
        image_vstore: Optional Milvus instance for storing image embeddings
            (pre-embedding ingestion). If not provided, images found in
            document metadata are skipped rather than embedded.
    """
    self.vstore = vstore
    self.image_vstore = image_vstore
    self.data_graph_db = data_graph_db
    self.graph_rag_enabled = graph_rag_enabled
    self.job_manager = job_manager
    self.max_property_length = max_property_length
    self.batch_size = batch_size
    self.logger = utils.get_logger("DocumentProcessor")

  @staticmethod
  def sanitize_entity_properties(entity: StructuredEntity, max_length: int = 250) -> None:
    """
    Sanitize entity properties by removing or filtering values that exceed max_length.
    - For string properties: remove the entire property if length > max_length
    - For list properties: remove individual elements that have length > max_length

    Args:
        entity: StructuredEntity to sanitize (modified in-place)
        max_length: Maximum allowed length for property values (default: 250)
    """
    properties_to_remove = []
    properties_to_update = {}

    for key, value in entity.all_properties.items():
      # Skip internal properties (starting with _)
      if key.startswith("_"):
        continue

      if isinstance(value, str):
        # Remove string properties that are too long
        if len(value) > max_length:
          properties_to_remove.append(key)

      elif isinstance(value, list):
        # Filter list elements that are too long
        filtered_list = []
        for item in value:
          # Only check string items in the list
          if isinstance(item, str):
            if len(item) <= max_length:
              filtered_list.append(item)
          else:
            # Keep non-string items as-is
            filtered_list.append(item)

        # Update the property with filtered list
        if len(filtered_list) != len(value):
          properties_to_update[key] = filtered_list

    # Remove properties that are too long
    for key in properties_to_remove:
      del entity.all_properties[key]

    # Update properties with filtered lists
    for key, value in properties_to_update.items():
      entity.all_properties[key] = value

  @staticmethod
  def format_entity_for_embedding(entity: StructuredEntity) -> str:
    """
    Format entity properties for embedding with emphasis on entity type and primary keys.
    Properties section is formatted as JSON for clean, readable output.
    """
    entity_properties = entity.get_external_properties()

    # Create formatted entity type using utility function
    formatted_type = utils.format_entity_type_for_display(entity.entity_type)

    # Extract primary key values
    primary_key_values = {}
    for pk_prop in entity.primary_key_properties:
      if pk_prop in entity_properties:
        primary_key_values[pk_prop] = entity_properties[pk_prop]

    # Build the formatted text with emphasis
    formatted_parts = []

    # Emphasize entity type at the top
    formatted_parts.append(f"=== ENTITY TYPE: {formatted_type} (Label: {entity.entity_type}) ===")
    formatted_parts.append("")

    # Emphasize primary key properties
    if primary_key_values:
      formatted_parts.append("=== PRIMARY KEY PROPERTIES ===")
      for pk_prop, pk_value in primary_key_values.items():
        formatted_parts.append(f"  {pk_prop}: {pk_value}")
      formatted_parts.append("")

    # Show additional key properties with their values if present
    if entity.additional_key_properties:
      formatted_parts.append("=== ADDITIONAL KEY PROPERTIES ===")
      # Collect unique values only
      unique_values = set()
      for key_set in entity.additional_key_properties:
        for key in key_set:
          if key in entity_properties:
            value = entity_properties[key]
            unique_values.add(str(value))
      # Print only unique values
      for value in sorted(unique_values):
        formatted_parts.append(f"  {value}")
      formatted_parts.append("")

    # Add all properties as JSON
    formatted_parts.append("=== ALL PROPERTIES ===")
    properties_json = utils.json_encode(entity_properties, indent=2)
    formatted_parts.append(properties_json)

    return "\n".join(formatted_parts)

  def _create_chunks_from_content(
    self,
    content: str,
    document_id: str,
    document_metadata: DocumentMetadata,
    max_chunk_size: int = 60000,
  ) -> Tuple[List[Document], List[str]]:
    """
    Create document chunks from text content with proper metadata.
    Splits content into chunks if it exceeds max_chunk_size.

    Args:
        content: The text content to chunk
        document_id: ID for the document
        document_metadata: Base metadata to inherit
        max_chunk_size: Maximum size per chunk

    Returns:
        Tuple of (list of Document chunks, list of chunk IDs)
    """
    chunks = []
    chunk_ids = []

    # Check if content needs chunking
    if len(content) > max_chunk_size:
      self.logger.debug(f"Content for {document_id} exceeds max size ({len(content)} > {max_chunk_size}), splitting into chunks")

      # Use text splitter to chunk the content
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=min(200, max_chunk_size // 10), length_function=len, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""])

      temp_doc = Document(page_content=content)
      split_docs = text_splitter.split_documents([temp_doc])

      self.logger.info(f"Split {document_id} into {len(split_docs)} chunks")

      # Create chunks with metadata
      for i, chunk_doc in enumerate(split_docs):
        chunk_id = f"{document_id}_chunk_{i}"

        chunk_metadata = DocumentChunkMetadata(
          id=chunk_id,
          document_id=document_id,
          datasource_id=document_metadata.datasource_id,
          ingestor_id=document_metadata.ingestor_id,
          title=document_metadata.title,
          description=document_metadata.description,
          is_structured_entity=document_metadata.is_structured_entity,
          document_type=document_metadata.document_type,
          document_ingested_at=document_metadata.document_ingested_at,
          fresh_until=document_metadata.fresh_until,
          metadata=document_metadata.metadata,
          chunk_index=i,
          total_chunks=len(split_docs),
        )

        chunk_doc.metadata = chunk_metadata.model_dump()
        chunks.append(chunk_doc)
        chunk_ids.append(chunk_id)
    else:
      # Content is small enough, create single chunk
      chunk_id = f"{document_id}_chunk_0"

      chunk_metadata = DocumentChunkMetadata(
        id=chunk_id,
        document_id=document_id,
        datasource_id=document_metadata.datasource_id,
        ingestor_id=document_metadata.ingestor_id,
        title=document_metadata.title,
        description=document_metadata.description,
        is_structured_entity=document_metadata.is_structured_entity,
        document_type=document_metadata.document_type,
        document_ingested_at=document_metadata.document_ingested_at,
        fresh_until=document_metadata.fresh_until,
        metadata=document_metadata.metadata,
        chunk_index=0,
        total_chunks=1,
      )

      chunk_doc = Document(page_content=content, metadata=chunk_metadata.model_dump())

      chunks.append(chunk_doc)
      chunk_ids.append(chunk_id)

    return chunks, chunk_ids

  def _process_structured_entity_document(
    self,
    doc: Document,
    document_metadata: DocumentMetadata,
    ingestor_id: str,
    datasource_id: str,
    fresh_until: int,
  ) -> Optional[Tuple[List[StructuredEntity], List[Document], List[str], List[str]]]:
    """
    Process a structured entity document by splitting nested entities for the graph database,
    but keeping the full entity text as a single document for vector search.

    Args:
        doc: The document containing the structured entity
        document_metadata: Metadata for the document
        ingestor_id: ID of the ingestor
        datasource_id: ID of the datasource
        fresh_until: Timestamp until which data is fresh

    Returns:
        Tuple of (entities list, chunks list, chunk_ids list, error_messages list), or None if entity parsing failed
    """
    # Initialize result containers
    all_entities = []
    all_chunks = []
    all_chunk_ids = []
    validation_errors = []

    # Parse the structured entity from document
    entity = self._parse_structured_entity(doc)
    if entity is None:
      # Validation failed, skip this entity
      return None

    entity_type = entity.entity_type

    self.logger.debug(f"Parsed structured entity of type {entity_type}")

    # Flatten properties BUT preserve arrays of dicts (they'll be split later)
    # This ensures primary_key_properties with dot notation work correctly
    entity.all_properties = utils.flatten_dict(entity.all_properties, wildcard_index=False, preserve_list_of_dicts=True)

    # Create the full entity text from the entity
    # This is what will be used for vector search embedding
    full_entity_text = self.format_entity_for_embedding(entity)

    # Split the entity into multiple entities for the graph database
    # Properties are already flattened except arrays of dicts
    entities = self.split_nested_structured_entity(entity)
    self.logger.debug(f"Split entity into {len(entities)} entities for graph DB (including parent)")

    # Log primary key after splitting
    if entities:
      parent_entity = entities[0]  # First entity is always the parent
      self.logger.debug(f"Parent entity primary key: {parent_entity.generate_primary_key()}")

    # Process each split entity and add to graph database collection
    for split_entity in entities:
      # Validate entity after splitting and flattening (properties now have dot notation)
      all_props = split_entity.all_properties

      # Validate primary_key_properties - critical, must exist
      missing_primary_keys = [key for key in split_entity.primary_key_properties if key not in all_props]
      if missing_primary_keys:
        error_msg = f"StructuredEntity type '{split_entity.entity_type}': missing primary key properties: {missing_primary_keys}. Available properties: {list(all_props.keys())}"
        self.logger.warning(f"Skipping entity - {error_msg}")
        validation_errors.append(error_msg)
        continue

      # Validate additional_key_properties - optional, just warn and remove invalid ones
      if split_entity.additional_key_properties:
        valid_additional_keys = []
        for id_keys in split_entity.additional_key_properties:
          missing_additional_keys = [key for key in id_keys if key not in all_props]
          if not missing_additional_keys:
            # All keys exist, keep this additional key set
            valid_additional_keys.append(id_keys)
          else:
            warning_msg = f"StructuredEntity type '{split_entity.entity_type}' with primary key '{split_entity.generate_primary_key()}': additional_key_properties {id_keys} has missing properties: {missing_additional_keys}. These additional keys will be ignored."
            self.logger.warning(warning_msg)
            validation_errors.append(warning_msg)

        # Update with only valid additional keys
        split_entity.additional_key_properties = valid_additional_keys

      # Sanitize entity properties (remove values that exceed max_property_length)
      self.sanitize_entity_properties(split_entity, max_length=self.max_property_length)

      # Add document metadata fields to entity properties
      split_entity.all_properties.update({INGESTOR_ID_KEY: ingestor_id, DATASOURCE_ID_KEY: datasource_id, LAST_UPDATED_KEY: document_metadata.document_ingested_at, FRESH_UNTIL_KEY: fresh_until})

      # Add to entities list
      all_entities.append(split_entity)

    # Create ONE document with the full entity text for vector search
    # Use the original (parent) entity information for the document ID
    # Set structured entity metadata in the document metadata
    document_id = self.structured_document_id(entity.entity_type, entity.generate_primary_key())
    entity_primary_key = entity.generate_primary_key()
    if document_metadata.metadata is None:
      document_metadata.metadata = {}
    document_metadata.metadata.update({"structured_entity_type": entity.entity_type, "structured_entity_pk": entity_primary_key})

    # Prepare metadata for chunking - set structured entity metadata explicitly for root entity
    entity_doc_metadata = DocumentMetadata(
      document_id=document_id,
      datasource_id=document_metadata.datasource_id,
      ingestor_id=document_metadata.ingestor_id,
      title=document_metadata.title,
      description=document_metadata.description,
      is_structured_entity=True,
      document_type=self.structured_document_type(entity.entity_type),
      document_ingested_at=document_metadata.document_ingested_at,
      fresh_until=document_metadata.fresh_until,
      metadata=document_metadata.metadata,
    )

    # Use common chunking method for the full entity text
    chunks, chunk_ids = self._create_chunks_from_content(
      content=full_entity_text,
      document_id=document_id,
      document_metadata=entity_doc_metadata,
    )

    all_chunks.extend(chunks)
    all_chunk_ids.extend(chunk_ids)

    return all_entities, all_chunks, all_chunk_ids, validation_errors

  def split_nested_structured_entity(self, entity: StructuredEntity) -> List[StructuredEntity]:
    """
    Split a nested structured entity into a list of entities.

    This function recursively processes an entity's properties:
    - Homogeneous lists of primitives (str, bool, int, float) are kept as-is
    - Lists of dictionaries are split into separate entities
    - Sub-dictionaries are flattened into the parent with dot notation
    - Lists of dictionaries in flattened sub-dictionaries are split as usual

    For each property that contains a list of dictionaries:
    - Creates new entities with type: parent_entity_type + "_" + property_name (singularized/capitalized)
    - Adds SUB_ENTITY_INDEX_KEY: index in the list
    - Adds PARENT_ENTITY_PK_KEY: primary key of parent entity
    - Uses ENTITY_TYPE_KEY (already present in all entities) in primary key to avoid clashes
    - Sets primary_key_properties to [PARENT_ENTITY_PK_KEY, ENTITY_TYPE_KEY, SUB_ENTITY_INDEX_KEY]
    - Recursively processes nested structures

    Returns:
        List[StructuredEntity]: List containing the modified parent entity and all extracted nested entities
    """
    result_entities = []

    # Get all properties (including internal ones starting with _)
    all_properties = entity.all_properties.copy()
    parent_primary_key = entity.generate_primary_key()
    parent_entity_type = entity.entity_type

    # Separate internal properties (starting with _) from external ones
    internal_properties = {k: v for k, v in all_properties.items() if k.startswith("_")}
    external_properties = {k: v for k, v in all_properties.items() if not k.startswith("_")}

    # Process external properties: flatten dicts, split list of dicts, keep homogeneous lists
    processed_properties = self._process_entity_properties(external_properties, parent_entity_type, parent_primary_key, result_entities)

    # Merge back internal properties (they should always be preserved)
    processed_properties.update(internal_properties)

    # Create the updated parent entity, preserving additional_types and additional_key_properties
    updated_parent = StructuredEntity(
      entity_type=parent_entity_type,
      all_properties=processed_properties,
      primary_key_properties=entity.primary_key_properties,
      additional_types=entity.additional_types,
      additional_key_properties=entity.additional_key_properties,
    )

    # Parent entity comes first in the result
    return [updated_parent] + result_entities

  def _process_entity_properties(self, properties: dict, parent_entity_type: str, parent_primary_key: str, result_entities: List[StructuredEntity], prefix: str = "") -> dict:
    """
    Process entity properties by splitting lists of dicts.
    Properties should already be flattened (except arrays of dicts) before calling this.

    Args:
        properties: Properties to process (already flattened, except arrays of dicts)
        parent_entity_type: Type of parent entity
        parent_primary_key: Primary key of parent entity
        result_entities: List to collect split entities
        prefix: Prefix for flattened keys (for recursive calls within split entities)

    Returns:
        Processed properties dictionary
    """
    processed = {}

    for key, value in properties.items():
      # Properties are already flattened, just use the key
      full_key = f"{prefix}.{key}" if prefix else key

      if isinstance(value, list) and value and isinstance(value[0], dict):
        # Split list of dictionaries into separate entities
        self._split_list_of_dicts(full_key, value, parent_entity_type, parent_primary_key, result_entities)
        # Don't add to processed properties (it's been split out)

      elif isinstance(value, dict):
        # This should only happen in recursive calls within split entities
        # Flatten these sub-dictionaries
        flattened = self._process_entity_properties(value, parent_entity_type, parent_primary_key, result_entities, full_key)
        processed.update(flattened)

      elif self._is_homogeneous_primitive_list(value) if isinstance(value, list) else True:
        # Keep primitives and homogeneous lists of primitives as-is
        processed[full_key] = value
      else:
        # Mixed or complex list - convert to string for safety
        processed[full_key] = str(value) if isinstance(value, list) else value

    return processed

  def _is_homogeneous_primitive_list(self, lst: list) -> bool:
    """
    Check if a list contains only primitive types (str, bool, int, float, None).

    Args:
        lst: List to check

    Returns:
        True if all elements are primitives, False otherwise
    """
    if not lst:
      return True

    primitive_types = (str, bool, int, float, type(None))
    return all(isinstance(item, primitive_types) for item in lst)

  def _split_list_of_dicts(self, prop_key: str, prop_value: list, parent_entity_type: str, parent_primary_key: str, result_entities: List[StructuredEntity]) -> None:
    """
    Split a list of dictionaries into separate entities.

    Args:
        prop_key: Property key (may include dots from flattening)
        prop_value: List of dictionaries
        parent_entity_type: Type of parent entity
        parent_primary_key: Primary key of parent entity
        result_entities: List to collect split entities
    """
    # Generate the new entity type name
    # Use more context to avoid collisions (e.g., nodeAffinity vs podAntiAffinity)
    key_parts = prop_key.split(".")

    # Use last 2 parts if available for better disambiguation
    if len(key_parts) >= 2:
      # e.g., "nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution"
      # becomes "Nodeaffinity_Preferredduringschedulingignoredduringexecution"
      type_parts = [self._generate_type_suffix(key_parts[-2]), self._generate_type_suffix(key_parts[-1])]
      type_suffix = "_".join(type_parts)
    else:
      # Only one part, use it
      type_suffix = self._generate_type_suffix(key_parts[-1])

    new_entity_type = f"{parent_entity_type}_{type_suffix}"

    # Create a new entity for each item in the list
    for index, item_dict in enumerate(prop_value):
      # Create properties for the new sub-entity
      sub_entity_properties = item_dict.copy()
      sub_entity_properties[PARENT_ENTITY_PK_KEY] = parent_primary_key
      sub_entity_properties[PARENT_ENTITY_TYPE_KEY] = parent_entity_type
      # Convert index to string to ensure homogeneous arrays in Neo4j
      sub_entity_properties[SUB_ENTITY_INDEX_KEY] = str(index)
      # Add entity type to properties so it can be used in primary key
      sub_entity_properties[ENTITY_TYPE_KEY] = new_entity_type

      # Create the new entity with SUB_ENTITY_LABEL to mark it as a sub-entity
      # Include ENTITY_TYPE_KEY in primary key to avoid clashes between different sub-entity types
      sub_entity = StructuredEntity(
        entity_type=new_entity_type,
        all_properties=sub_entity_properties,
        primary_key_properties=[PARENT_ENTITY_PK_KEY, ENTITY_TYPE_KEY, SUB_ENTITY_INDEX_KEY],
        additional_types={SUB_ENTITY_LABEL},
      )

      self.logger.debug(f"Created sub-entity '{new_entity_type}' with additional_types: {sub_entity.additional_types}")

      # Recursively process the sub-entity for nested structures
      sub_entities = self.split_nested_structured_entity(sub_entity)

      # Verify labels are preserved after recursive processing
      for se in sub_entities:
        self.logger.debug(f"After recursive processing: entity_type='{se.entity_type}', additional_types={se.additional_types}")

      result_entities.extend(sub_entities)

  def _generate_type_suffix(self, property_key: str) -> str:
    """
    Generate a type suffix from a property key. Based on common patterns and heuristics.

    This performs basic singularization and capitalization:
    - "containers" -> "Container"
    - "items" -> "Item"
    - "status" -> "Status"
    - "ops" -> "Ops"

    Args:
        property_key: The property key to convert

    Returns:
        str: The formatted type suffix
    """
    # Words that should not be singularized - heuristic based on common patterns
    NO_SINGULARIZE_SUFFIXES = {"prometheus", "aas", "ops", "status", "series", "species", "apparatus", "progress", "chassis", "redis", "jenkins"}

    # Remove underscores and convert to title case
    words = property_key.replace("_", " ").split()

    # Simple singularization heuristic
    result_words = []
    for word in words:
      word_lower = word.lower()

      # If word is too short, don't singularize
      if len(word) <= 4:
        singular = word
      # Check if word should not be singularized
      elif any(word_lower.endswith(suffix) for suffix in NO_SINGULARIZE_SUFFIXES):
        singular = word
      # Handle common plurals d
      elif word_lower.endswith("ies"):
        # "categories" -> "category"
        singular = word[:-3] + "y"
      elif word_lower.endswith("ses"):
        # "addresses" -> "address"
        singular = word[:-2]
      elif word_lower.endswith("s") and not word_lower.endswith("ss"):
        # "containers" -> "container"
        singular = word[:-1]
      else:
        singular = word

      result_words.append(singular.capitalize())

    return "".join(result_words)

  @staticmethod
  def structured_document_type(entity_type: str) -> str:
    return f"structured:{entity_type}"

  @staticmethod
  def structured_document_id(entity_type: str, entity_pk: str) -> str:
    return f"structured:{entity_type}:{entity_pk}"

  @staticmethod
  def parse_structured_entity_from_document_id(document_id: str) -> Tuple[str, str]:
    """
    Parse entity_type and entity_pk from a structured document ID of the form "structured:entity_type:entity_pk"
    """
    parts = document_id.split(":")
    if len(parts) < 3 or parts[0] != "structured":
      raise ValueError(f"Invalid structured document ID format: {document_id}")
    entity_type = parts[1]
    entity_pk = ":".join(parts[2:])  # In case entity_pk contains ':'
    return entity_type, entity_pk

  def _parse_document_metadata(self, doc: Document) -> DocumentMetadata:
    """
    Parse document metadata from Document.metadata dict into DocumentMetadata model.
    """
    try:
      return DocumentMetadata.model_validate(doc.metadata)
    except Exception as e:
      self.logger.error(f"Failed to parse document metadata: {e}")
      raise ValueError(f"Invalid document metadata: {e}")

  def _parse_structured_entity(self, doc: Document) -> Optional[StructuredEntity]:
    """
    Parse document page_content into a StructuredEntity if it's a structured entity document.
    Returns None if parsing fails (entity will be skipped).
    Validation happens after splitting and flattening on server side.
    Optimized for high-throughput ingestion.
    """
    try:
      # Use Pydantic's model_validate_json to parse JSON string directly into StructuredEntity
      entity = StructuredEntity.model_validate_json(doc.page_content)
      return entity
    except Exception as e:
      self.logger.error(f"Failed to parse structured entity from document: {e}")
      raise ValueError(f"Invalid structured entity document: {e}")

  def _chunk_document(self, doc: Document, document_metadata: DocumentMetadata, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Chunk a document if it exceeds chunk_size, otherwise return as single chunk.
    If chunk_size is 0, skip chunking entirely.
    Chunk size is capped at MILVUS_MAX_VARCHAR_LENGTH to respect Milvus field limits.
    """
    content = doc.page_content
    if not content:
      self.logger.warning("Empty content, returning empty chunks")
      return []

    chunks = []

    # Cap chunk size at Milvus limit
    effective_chunk_size = min(chunk_size, self.MILVUS_MAX_VARCHAR_LENGTH) if chunk_size > 0 else self.MILVUS_MAX_VARCHAR_LENGTH

    # Check if document needs chunking
    if len(content) > effective_chunk_size and chunk_size > 0:
      self.logger.debug(f"Document exceeds chunk size ({len(content)} > {effective_chunk_size}), splitting into chunks")

      # Create document-specific text splitter
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=effective_chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""])

      doc_chunks = text_splitter.split_documents([doc])
      self.logger.debug(f"Split document into {len(doc_chunks)} chunks for: {document_metadata.document_id}")

      for i, chunk_doc in enumerate(doc_chunks):
        # Compute chunk id and add the metadata from document, as well as chunk-specific info
        chunk_id = f"{document_metadata.document_id}_chunk_{i}"
        chunk_metadata = DocumentChunkMetadata(
          id=chunk_id,
          document_id=document_metadata.document_id,
          datasource_id=document_metadata.datasource_id,
          ingestor_id=document_metadata.ingestor_id,
          title=document_metadata.title,
          description=document_metadata.description,
          is_structured_entity=document_metadata.is_structured_entity,
          document_type=document_metadata.document_type,
          document_ingested_at=document_metadata.document_ingested_at,
          fresh_until=document_metadata.fresh_until,
          metadata=document_metadata.metadata,
          chunk_index=i,
          total_chunks=len(doc_chunks),
        )

        chunk_doc.metadata = chunk_metadata.model_dump()
        chunks.append(chunk_doc)
    else:
      if chunk_size == 0:
        self.logger.debug(f"Chunk size is 0, skipping chunking for: {document_metadata.document_id}")
      else:
        self.logger.debug(f"Document is smaller than chunk size, processing as single chunk: {document_metadata.document_id}")

      # Compute chunk id and add the metadata from document, as well as chunk-specific info
      chunk_id = f"{document_metadata.document_id}_chunk_0"
      chunk_metadata = DocumentChunkMetadata(
        id=chunk_id,
        document_id=document_metadata.document_id,
        datasource_id=document_metadata.datasource_id,
        ingestor_id=document_metadata.ingestor_id,
        title=document_metadata.title,
        description=document_metadata.description,
        is_structured_entity=document_metadata.is_structured_entity,
        document_type=document_metadata.document_type,
        document_ingested_at=document_metadata.document_ingested_at,
        fresh_until=document_metadata.fresh_until,
        metadata=document_metadata.metadata,
        chunk_index=0,
        total_chunks=1,
      )

      doc.metadata = chunk_metadata.model_dump()
      chunks.append(doc)

    return chunks

  async def ingest_documents(self, ingestor_id: str, datasource_id: str, job_id: str, documents: List[Document], fresh_until: int, chunk_size: int, chunk_overlap: int):
    """
    Process documents, splitting into chunks if necessary, and handle both regular documents and structured entities.

    Args:
        ingestor_id: ID of the ingestor ingesting the documents
        datasource_id: ID of the datasource ingesting the documents
        job_id: ID of the ingestion job
        documents: List of documents to ingest
        fresh_until: Timestamp until which this data is considered fresh (epoch seconds)
        chunk_size: Maximum size for document chunks
        chunk_overlap: Overlap between chunks
    """
    if not documents:
      self.logger.warning("No documents provided for ingestion")
      return

    self.logger.info(f"Starting ingestion of {len(documents)} documents")

    # Step 1: Process each document and collect chunks and structured entities
    all_chunks = []
    all_chunk_ids = []
    all_entities = []  # Simple list of entities

    for doc in documents:
      try:
        # Step 1a: Parse document metadata
        document_metadata = self._parse_document_metadata(doc)

        # Override metadata fields - this is to ensure correct association (even if doc.metadata has different values)
        document_metadata.ingestor_id = ingestor_id
        document_metadata.datasource_id = datasource_id
        # Use per-document fresh_until if provided, otherwise fall back to batch-level value
        document_metadata.fresh_until = document_metadata.fresh_until or fresh_until
        document_metadata.document_ingested_at = int(time.time())

        # Step 1b: Check if it's a structured entity and parse if needed
        if self.graph_rag_enabled and document_metadata.is_structured_entity:
          try:
            result = self._process_structured_entity_document(
              doc=doc,
              document_metadata=document_metadata,
              ingestor_id=ingestor_id,
              datasource_id=datasource_id,
              fresh_until=fresh_until,
            )

            # Skip if entity parsing failed
            if result is None:
              self.logger.warning("Skipping structured entity document due to parsing failure")
              continue

            entities, chunks, chunk_ids, validation_errors = result

            # Add validation errors to job if any occurred
            if validation_errors:
              for error in validation_errors:
                await self.job_manager.add_error_msg(job_id, error)

            # Collect entities, chunks, and chunk IDs
            all_entities.extend(entities)
            all_chunks.extend(chunks)
            all_chunk_ids.extend(chunk_ids)

          except Exception as e:
            error_msg = f"Failed to parse structured entity: {e}"
            self.logger.error(f"{error_msg}, skipping")
            self.logger.error(traceback.format_exc())
            await self.job_manager.add_error_msg(job_id, error_msg)
            continue

        else:
          # adding this debug log to clarify processing of regular documents when graph RAG is disabled but document is a structured entity
          if not self.graph_rag_enabled and document_metadata.is_structured_entity:
            self.logger.debug(f"Document marked as structured entity but graph RAG is disabled, treating as regular document: {document_metadata.document_id}")

          # Step 2: Chunk regular documents
          chunks = self._chunk_document(doc, document_metadata, chunk_size, chunk_overlap)

          for chunk in chunks:
            chunk_metadata = DocumentChunkMetadata.model_validate(chunk.metadata)
            all_chunks.append(chunk)
            all_chunk_ids.append(chunk_metadata.id)

      except Exception as e:
        error_msg = f"Failed to parse and process document: {e}"
        self.logger.error(f"{error_msg}, skipping")
        self.logger.error(traceback.format_exc())
        await self.job_manager.add_error_msg(job_id, error_msg)
        continue

    # Step 3: Add all document chunks to vector database
    deduped_chunks = []  # Initialize for final message
    if all_chunks:
      self.logger.info(f"Adding {len(all_chunks)} document chunks to vector database")

      # Deduplicate chunks by ID (keep first occurrence)
      seen_ids = set()
      deduped_chunk_ids = []

      for chunk, chunk_id in zip(all_chunks, all_chunk_ids):
        if chunk_id not in seen_ids:
          seen_ids.add(chunk_id)
          deduped_chunks.append(chunk)
          deduped_chunk_ids.append(chunk_id)
        else:
          self.logger.debug(f"Skipping duplicate chunk ID: {chunk_id}")

      if len(deduped_chunks) < len(all_chunks):
        self.logger.warning(f"Removed {len(all_chunks) - len(deduped_chunks)} duplicate chunks. Original: {len(all_chunks)}, After dedup: {len(deduped_chunks)}")

      try:
        # Update job message
        await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Adding {len(deduped_chunks)} document chunks to vector database")
        await self.vstore.aupsert(documents=deduped_chunks, ids=deduped_chunk_ids, batch_size=self.batch_size)
        self.logger.info(f"Successfully added {len(deduped_chunks)} chunks to vector database")
        # Track chunk count
        await self.job_manager.increment_chunk_count(job_id, len(deduped_chunks))
        # Update job with success message
        await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Added {len(deduped_chunks)} document chunks to vector database")
      except Exception as e:
        error_msg = f"Failed to add chunks to vector database: {e}"
        self.logger.error(error_msg)
        self.logger.error(traceback.format_exc())
        await self.job_manager.add_error_msg(job_id, error_msg)
        raise

    # Step 3b: Extract images from documents and store their embeddings
    # (pre-embedding ingestion). Skipped entirely if image_vstore is not configured.
    if self.image_vstore is not None:
      await self._ingest_images(documents=documents, job_id=job_id)

    # Step 4: Add structured entities to graph database in one batch
    if all_entities and self.data_graph_db:
      total_entities = len(all_entities)
      self.logger.info(f"Adding {total_entities} structured entities to graph database in one batch")

      # Update job message
      await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Adding {total_entities} structured entities in one batch")

      try:
        # Add all entities to graph database in ONE  call
        await self.data_graph_db.update_entity_batch(entities=all_entities, batch_size=1000)
        self.logger.info(f"Successfully added {total_entities} entities to graph database in ONE batch")

        # Final success message for structured entities
        await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Successfully added {total_entities} structured entities to graph database in one batch")
      except Exception as e:
        error_msg = f"Failed to add entities to graph database: {e}"
        self.logger.error(error_msg)
        self.logger.error(traceback.format_exc())
        await self.job_manager.add_error_msg(job_id, error_msg)
        # Continue with the rest of the processing

    # Final completion message
    total_entities = len(all_entities)
    completion_msg = f"[Server] Ingestion complete: {len(deduped_chunks) if all_chunks else 0} document chunks, {total_entities} structured entities"
    self.logger.info(completion_msg)
    await self.job_manager.upsert_job(job_id=job_id, message=completion_msg)

  async def _ingest_images(self, documents: List[Document], job_id: str) -> None:
    """Extract image URLs from document metadata, embed and store new ones."""
    image_docs, image_ids = self._collect_image_candidates(documents)
    if not image_docs:
      return

    existing_ids = await self._get_existing_image_ids(image_ids)
    to_process = [(doc, doc_id) for doc, doc_id in zip(image_docs, image_ids) if doc_id not in existing_ids]
    skipped = len(image_docs) - len(to_process)

    if not to_process:
      self.logger.info(f"All {len(image_docs)} images already stored, nothing to embed")
      return

    self.logger.info(f"Embedding {len(to_process)} new images ({skipped} already stored, skipped)")
    await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Embedding {len(to_process)} images")

    successful = await self._embed_and_store_images(to_process)

    self.logger.info(f"Successfully embedded {successful}/{len(to_process)} new images ({skipped} already stored)")
    await self.job_manager.upsert_job(job_id=job_id, message=f"[Server] Embedded {successful}/{len(to_process)} images ({skipped} already stored)")

  def _collect_image_candidates(self, documents: List[Document]) -> Tuple[List[Document], List[str]]:
    """Extract image URLs from document metadata into (image_doc, deterministic_id) pairs."""
    image_docs: List[Document] = []
    image_ids: List[str] = []
    for doc in documents:
      nested_metadata = doc.metadata.get("metadata") or {}
      images = nested_metadata.get("images") or []
      if isinstance(images, str):
        images = json.loads(images)
      source_url = nested_metadata.get("source", "")

      if len(images) > MAX_IMAGES_PER_DOCUMENT:
        self.logger.info(f"Capping images for {source_url or 'unknown'}: {len(images)} found, embedding first {MAX_IMAGES_PER_DOCUMENT}")
        images = images[:MAX_IMAGES_PER_DOCUMENT]

      for image in images:
        image_url = image.get("url")
        if not image_url:
          continue
        image_docs.append(Document(page_content=image_url, metadata={"alt_text": image.get("alt_text", ""), "source_document": source_url}))
        image_ids.append(self._image_id_for_url(image_url))
    return image_docs, image_ids

  @staticmethod
  def _image_id_for_url(image_url: str) -> str:
    """Deterministic id so re-embedding the same URL updates rather than duplicates."""
    url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
    return f"img_{url_hash}"

  async def _get_existing_image_ids(self, image_ids: List[str]) -> set:
    """Query which of the given ids are already stored, to skip re-embedding unchanged images."""
    if not image_ids:
      return set()
    try:
      existing = await self.image_vstore.aclient.query(
        collection_name=self.image_vstore.collection_name,
        filter=f"pk in [{', '.join(repr(i) for i in image_ids)}]",
        output_fields=["pk"],
      )
      return {row["pk"] for row in existing}
    except Exception as e:
      self.logger.warning(f"Could not check existing images, embedding all: {e}")
      return set()

  async def _embed_and_store_images(self, to_process: List[Tuple[Document, str]]) -> int:
    """Embed images concurrently (thread pool), then batch-insert pre-computed vectors."""
    embedder = getattr(self.image_vstore.embeddings, "embedder", None)
    embedding_provider = embedder.__class__.__name__ if embedder is not None else None
    semaphore = asyncio.Semaphore(IMAGE_EMBED_CONCURRENCY)
    loop = asyncio.get_event_loop()

    async def embed_bounded(url: str) -> Optional[List[float]]:
      async with semaphore:
        return await loop.run_in_executor(None, self._embed_with_retries, embedder, url)

    vectors = await asyncio.gather(*[embed_bounded(doc.page_content) for doc, _ in to_process])

    texts, embeddings, metadatas, ids = [], [], [], []
    for (doc, doc_id), vector in zip(to_process, vectors):
      if vector is None:
        continue
      texts.append(doc.page_content)
      embeddings.append(vector)
      metadata = dict(doc.metadata)
      if embedding_provider:
        metadata["embedding_provider"] = embedding_provider
      metadatas.append(metadata)
      ids.append(doc_id)

    if not texts:
      return 0

    try:
      await self.image_vstore.aadd_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
      return len(texts)
    except Exception as e:
      self.logger.warning(f"Failed to store {len(texts)} embedded images: {e}")
      return 0

  def _embed_with_retries(self, embedder, url: str) -> Optional[List[float]]:
    """Runs in a worker thread: plain synchronous HTTP calls, safe to parallelize."""
    for attempt in range(1, IMAGE_EMBED_MAX_ATTEMPTS + 1):
      try:
        return embedder.embed_image_url(url)
      except UnsupportedImageFormatError as e:
        self.logger.warning(f"Skipping unsupported image format: {url} ({e})")
        return None
      except Exception as e:
        if attempt >= IMAGE_EMBED_MAX_ATTEMPTS:
          self.logger.warning(f"Failed to embed image {url} after {attempt} attempts: {e}")
          return None
        time.sleep(IMAGE_EMBED_RETRY_DELAY_SECONDS)
    return None
