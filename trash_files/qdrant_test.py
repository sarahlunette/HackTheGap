from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Connect to local Qdrant server
client = QdrantClient(url="https://ac035607-022e-44cf-9607-31bbb73ccb29.us-east4-0.gcp.cloud.qdrant.io:6333",
                      api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rWqAOEL_W2VdlCfC0y_8Xsu08UQHWuZCE0A6DfznWIM"
                      )

# 1. Create a collection
client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE)
)

# 2. Insert points
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"text": "hello world"}
    ),
    PointStruct(
        id=2,
        vector=[0.4, 0.3, 0.2, 0.1],
        payload={"text": "bonjour"}
    ),
]

client.upsert(
    collection_name="my_collection",
    points=points
)

print("Data inserted!")
