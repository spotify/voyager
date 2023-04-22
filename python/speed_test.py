import voyager
import time


a = time.time()
index = voyager.Index.load(
    "/Users/psobot/Code/spotit-aggregation/top-100k/index.hnsw",
    voyager.Space.Cosine,
    16,
    voyager.StorageDataType.Float8,
)
b = time.time()
print(f"Loaded index in {b-a:.6f} seconds.")


a = time.time()
index.ids
b = time.time()
print(f"Referencing .ids took {b-a:.6f} seconds: {index.ids}")

a = time.time()
len(index.ids)
b = time.time()
print(f"len(index.ids) took {b-a:.6f} seconds. ({len(index.ids):,} items)")

a = time.time()
list(index.ids)
b = time.time()
print(f"list(index.ids) took {b-a:.6f} seconds.")

iterator = iter(index.ids)
a = time.time()
for _ in iterator:
    pass
b = time.time()
print(f"iterating through index.ids took {b-a:.6f} seconds.")

ids_list = list(index.ids)
a = time.time()
for _ in ids_list:
    pass
b = time.time()
print(f"iterating through list(index.ids) took {b-a:.6f} seconds.")
