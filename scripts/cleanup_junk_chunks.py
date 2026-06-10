"""Delete junk chunks (< 20 chars) from ChromaDB."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.config.settings import get_settings
from src.retrieval.vectorstore import get_vectorstore

s = get_settings()
vs = get_vectorstore(s)
col = vs._collection

total = col.count()
print(f"Total docs before cleanup: {total}")

# Collect IDs of junk chunks
batch_size = 1000
offset = 0
junk_ids = []

while offset < total:
    batch = col.get(limit=batch_size, offset=offset, include=["documents"])
    ids = batch["ids"]
    docs = batch["documents"]
    if not docs:
        break
    for doc_id, doc_text in zip(ids, docs):
        if len(doc_text.strip()) < 20:
            junk_ids.append(doc_id)
    offset += len(docs)
    print(f"  Scanned {offset}/{total} docs, found {len(junk_ids)} junk so far...")

print(f"\nFound {len(junk_ids)} junk chunks to delete")

# Delete in batches
if junk_ids:
    delete_batch_size = 500
    for i in range(0, len(junk_ids), delete_batch_size):
        batch = junk_ids[i:i+delete_batch_size]
        col.delete(ids=batch)
        print(f"  Deleted batch {i//delete_batch_size + 1} ({len(batch)} chunks)")

    new_total = col.count()
    print(f"\nTotal docs after cleanup: {new_total}")
    print(f"Removed: {total - new_total} junk chunks")
else:
    print("No junk chunks found!")
