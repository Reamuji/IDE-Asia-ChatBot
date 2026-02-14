from ollama import embed

import time

def time_call(fn, runs=50):
    loadTime = 0
    start = time.perf_counter() * 1000
    fn()
    loadTime = time.perf_counter() * 1000 - start

    
    times = []
    for _ in range(runs):
        start = time.perf_counter() * 1000
        fn()
        times.append(time.perf_counter() * 1000 - start)
    return {
        "lad_ms":loadTime,
        "avg_ms": sum(times) / runs,
        "min_ms": min(times),
        "max_ms": max(times),
        "full data": times
    }


# response = embed(model='all-minilm:l6-v2', input='Hello, world!')
# print(response)

# response = embed(model='llama3.2', input='Hello, world!')
# print(response['embeddings'])

t1 = time_call(lambda: embed(model='qwen3-embedding:0.6b', input='Hello, world!'))
t2 = time_call(lambda: embed(model='embeddinggemma:latest', input='Hello, world!'))

print(t1)
print(t2)
