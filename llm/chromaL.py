import chromadb

chroma_client = chromadb.Client()


# in future uodate use @chroma-core/ollama instead
from ollama import embed
class llama3p2Embedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='llama3.2', input=input)['embeddings']
class minilmEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='all-minilm:l6-v2', input=input)['embeddings']


collection = chroma_client.create_collection(
    name="IDEAsiaFacts",
    embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
)

docs = [
  "IDE Asia / PT Solusi Inovasi Bangsa is an Indonesian company operating in the information technology and IT services industry.",
  "IDE Asia is commonly known as IDE Asia.",
  "IDE Asia was founded in 2019.",
  "IDE Asia is a private company (Perseroan Terbatas).",
  "IDE Asia focuses on IT consulting, software development, and managed IT services.",
  "IDE Asia provides services primarily to clients in Indonesia and the Asia-Pacific region.",
  "IDE Asia originally positioned itself to fulfill telecommunication service demands.",
  "IDE Asia offers IT outsourcing services by providing ready-to-work technical talent based on client needs.",
  "IDE Asia provides IT enhancement services to improve system efficiency, security, and automation.",
  "IDE Asia delivers IT project services covering planning, development, production, and post-production stages.",
  "IDE Asia offers custom software development services for business-specific solutions.",
  "IDE Asia provides AI training and AI implementation services for business automation and innovation.",
  "IDE Asia offers technical support and managed services for client systems and infrastructure.",
  "IDE Asia states that it provides 24/7 client support services.",
  "The headquarters of IDE Asia is located in Bandung, West Java, Indonesia.",
  "One listed office address is Jl. Komp. Luxor No.5 Kav.11, Bandung, West Java, Indonesia.",
  "IDE Asia may have multiple registered or operational office locations in Bandung.",
  "The official website of IDE Asia is https://ide.asia.",
  "The official contact email of IDE Asia is info@ide.asia.",
  "The official contact phone number of IDE Asia is +62 821-1567-8446.",
  "IDE Asia states its official opening hours as Monday to Saturday from 10:00 AM to 4:00 PM.",
  "IDE Asia is closed on Sundays.",
  "IDE Asia is certified under ISO 9001 for quality management systems.",
  "IDE Asia is certified under ISO 27001:2022 for information security management systems.",
  "IDE Asia applies agile methodologies in its software development processes.",
  "IDE Asia employs software developers, UI/UX designers, business analysts, and technical support engineers.",
  "IDE Asia has recruited for cybersecurity-related roles, including Palo Alto Networks Cortex XDR operators.",
  "IDE Asia has an estimated workforce size ranging from approximately 50 to 200 employees depending on source.",
  "IDE Asia emphasizes professionalism, reliability, innovation, diversity, and excellence as core values.",
  "IDE Asia’s vision is to become a leading IT company in the Asia-Pacific region with global service capability.",
  "IDE Asia’s mission includes improving employee expertise, maintaining integrity, delivering seamless services, and encouraging collaboration."
]

ids = [f"id{i}" for i in range(len(docs))]

collection.add(
    ids=ids,
    documents=docs
)

results = collection.query(
    query_texts=["Contact"], # Chroma will embed this for you
    n_results=3 # how many results to return
)

print(results)