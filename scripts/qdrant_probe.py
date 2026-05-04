"""Qdrant probe — verify container is reachable, create collection, upsert+query."""
import sys
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION = "tracks_probe"
DIM = 768  # MERT-v1-95M hidden dim

client = QdrantClient(host="127.0.0.1", port=6333)
print(f"qdrant reachable: {[c.name for c in client.get_collections().collections]}")

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
)

rng = np.random.default_rng(42)
points = [
    qm.PointStruct(
        id=str(uuid.uuid4()),
        vector=rng.standard_normal(DIM).tolist(),
        payload={"title": f"track_{i}", "bpm": 120 + i, "key": "8A"},
    )
    for i in range(5)
]
client.upsert(collection_name=COLLECTION, points=points, wait=True)

info = client.get_collection(COLLECTION)
print(f"collection '{COLLECTION}': points={info.points_count} dim={info.config.params.vectors.size}")

query = rng.standard_normal(DIM).tolist()
hits = client.query_points(collection_name=COLLECTION, query=query, limit=3).points
print("nearest 3:")
for h in hits:
    print(f"  {h.payload['title']:10s} bpm={h.payload['bpm']} score={h.score:.4f}")

hits = client.query_points(
    collection_name=COLLECTION,
    query=query,
    limit=3,
    query_filter=qm.Filter(must=[qm.FieldCondition(key="bpm", range=qm.Range(gte=122))]),
).points
print(f"filtered (bpm>=122): {len(hits)} hits")
for h in hits:
    print(f"  {h.payload['title']:10s} bpm={h.payload['bpm']}")

client.delete_collection(COLLECTION)
print("OK")
