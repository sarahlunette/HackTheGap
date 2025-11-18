import os
import sys
sys.path.append('..')
from app.main import query_knowledge_base

def test_rag_returns_string():
    out = query_knowledge_base("What is resilience?")
    assert isinstance(out, str)
