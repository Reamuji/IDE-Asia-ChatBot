from ollama import embed,chat
import chromadb
import time

ChromaDB = chromadb.Client()


# gemma3 gk support Embedding

class llama3p2Embedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='llama3.2', input=input)['embeddings']
    
class testEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='granite4:3b', input=input)['embeddings']

class commandREmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='command-r7b:latest', input=input)['embeddings']




class minilm22mEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='all-minilm:l6-v2', input=input)['embeddings']
    
class minilm33mEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='all-minilm:l12-v2', input=input)['embeddings']
    
class snowflake22mEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='snowflake-arctic-embed:22m', input=input)['embeddings']
    
class snowflake33mEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='snowflake-arctic-embed:33m', input=input)['embeddings']
        
class snowflake137mEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='snowflake-arctic-embed:137m', input=input)['embeddings']
        
class gemmaEmbedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='embeddinggemma', input=input)['embeddings']
        
class qwen3Embedding(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return embed(model='qwen3-embedding:0.6b', input=input)['embeddings']
    
    
    

        



# docs = [
#   "IDE Asia / PT Solusi Inovasi Bangsa is an Indonesian company operating in the information technology and IT services industry.",
#   "IDE Asia is commonly known as IDE Asia.",
#   "IDE Asia was founded in 2019.",
#   "IDE Asia is a private company (Perseroan Terbatas).",
#   "IDE Asia focuses on IT consulting, software development, and managed IT services.",
#   "IDE Asia provides services primarily to clients in Indonesia and the Asia-Pacific region.",
#   "IDE Asia originally positioned itself to fulfill telecommunication service demands.",
#   "IDE Asia offers IT outsourcing services by providing ready-to-work technical talent based on client needs.",
#   "IDE Asia provides IT enhancement services to improve system efficiency, security, and automation.",
#   "IDE Asia delivers IT project services covering planning, development, production, and post-production stages.",
#   "IDE Asia offers custom software development services for business-specific solutions.",
#   "IDE Asia provides AI training and AI implementation services for business automation and innovation.",
#   "IDE Asia offers technical support and managed services for client systems and infrastructure.",
#   "IDE Asia states that it provides 24/7 client support services.",
#   "The headquarters of IDE Asia is located in Bandung, West Java, Indonesia.",
#   "One listed office address is Jl. Komp. Luxor No.5 Kav.11, Bandung, West Java, Indonesia.",
#   "IDE Asia may have multiple registered or operational office locations in Bandung.",
#   "The official website of IDE Asia is https://ide.asia.",
#   "The official contact email of IDE Asia is info@ide.asia.",
#   "The official contact phone number of IDE Asia is +62 821-1567-8446.",
#   "IDE Asia states its official opening hours as Monday to Saturday from 10:00 AM to 4:00 PM.",
#   "IDE Asia is closed on Sundays.",
#   "IDE Asia is certified under ISO 9001 for quality management systems.",
#   "IDE Asia is certified under ISO 27001:2022 for information security management systems.",
#   "IDE Asia applies agile methodologies in its software development processes.",
#   "IDE Asia employs software developers, UI/UX designers, business analysts, and technical support engineers.",
#   "IDE Asia has recruited for cybersecurity-related roles, including Palo Alto Networks Cortex XDR operators.",
#   "IDE Asia has an estimated workforce size ranging from approximately 50 to 200 employees depending on source.",
#   "IDE Asia emphasizes professionalism, reliability, innovation, diversity, and excellence as core values.",
#   "IDE Asia’s vision is to become a leading IT company in the Asia-Pacific region with global service capability.",
#   "IDE Asia’s mission includes improving employee expertise, maintaining integrity, delivering seamless services, and encouraging collaboration."
# ]
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
  "IDE Asia’s mission includes improving employee expertise, maintaining integrity, delivering seamless services, and encouraging collaboration.",
  "IDE Asia / PT Solusi Inovasi Bangsa adalah perusahaan Indonesia yang bergerak di industri teknologi informasi dan layanan TI.",
  "IDE Asia umumnya dikenal dengan nama IDE Asia.",
  "IDE Asia didirikan pada tahun 2019.",
  "IDE Asia merupakan perusahaan swasta (Perseroan Terbatas).",
  "IDE Asia berfokus pada konsultasi TI, pengembangan perangkat lunak, dan layanan TI terkelola.",
  "IDE Asia menyediakan layanan terutama bagi klien di Indonesia dan kawasan Asia-Pasifik.",
  "IDE Asia awalnya memposisikan diri untuk memenuhi kebutuhan layanan telekomunikasi.",
  "IDE Asia menawarkan layanan alih daya TI dengan menyediakan talenta teknis siap kerja sesuai kebutuhan klien.",
  "IDE Asia menyediakan layanan peningkatan TI untuk meningkatkan efisiensi sistem, keamanan, dan otomatisasi.",
  "IDE Asia menyediakan layanan proyek TI yang mencakup tahap perencanaan, pengembangan, produksi, dan pascaproduksi.",
  "IDE Asia menawarkan layanan pengembangan perangkat lunak kustom untuk solusi yang disesuaikan dengan kebutuhan bisnis.",
  "IDE Asia menyediakan layanan pelatihan AI dan implementasi AI untuk otomatisasi dan inovasi bisnis.",
  "IDE Asia menawarkan dukungan teknis dan layanan terkelola untuk sistem dan infrastruktur klien.",
  "IDE Asia menyatakan bahwa mereka menyediakan layanan dukungan klien 24/7.",
  "Kantor pusat IDE Asia berlokasi di Bandung, Jawa Barat, Indonesia.",
  "Salah satu alamat kantor yang terdaftar adalah Jl. Komp. Luxor No.5 Kav.11, Bandung, Jawa Barat, Indonesia.",
  "IDE Asia kemungkinan memiliki beberapa lokasi kantor terdaftar atau operasional di Bandung.",
  "Situs web resmi IDE Asia adalah https://ide.asia.",
  "Email kontak resmi IDE Asia adalah info@ide.asia.",
  "Nomor telepon kontak resmi IDE Asia adalah +62 821-1567-8446.",
  "IDE Asia menyatakan jam operasional resminya adalah Senin hingga Sabtu pukul 10.00 hingga 16.00.",
  "IDE Asia tutup pada hari Minggu.",
  "IDE Asia tersertifikasi ISO 9001 untuk sistem manajemen mutu.",
  "IDE Asia tersertifikasi ISO 27001:2022 untuk sistem manajemen keamanan informasi.",
  "IDE Asia menerapkan metodologi agile dalam proses pengembangan perangkat lunaknya.",
  "IDE Asia mempekerjakan pengembang perangkat lunak, desainer UI/UX, analis bisnis, dan insinyur dukungan teknis.",
  "IDE Asia telah merekrut untuk peran yang berkaitan dengan keamanan siber, termasuk operator Palo Alto Networks Cortex XDR.",
  "IDE Asia memiliki estimasi jumlah karyawan berkisar antara sekitar 50 hingga 200 orang, tergantung sumber.",
  "IDE Asia menekankan profesionalisme, keandalan, inovasi, keberagaman, dan keunggulan sebagai nilai inti.",
  "Visi IDE Asia adalah menjadi perusahaan TI terkemuka di kawasan Asia-Pasifik dengan kemampuan layanan global.",
  "Misi IDE Asia mencakup peningkatan keahlian karyawan, menjaga integritas, memberikan layanan tanpa hambatan, dan mendorong kolaborasi."
]
ids = [f"id{i}" for i in range(len(docs))]



collectionMinilm22m= ChromaDB.create_collection(
    name="IDEAsiaFacts1",
    # embedding_function=llama3p2Embedding()
    embedding_function=minilm22mEmbedding()
    # embedding_function=commandREmbedding()
    # embedding_function=snowflakeEmbedding()
    # embedding_function=testEmbedding()
)

collectionSnowflake33m = ChromaDB.create_collection(
    name="IDEAsiaFacts2",
    # embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
    # embedding_function=commandREmbedding()
    embedding_function=snowflake33mEmbedding()
    # embedding_function=testEmbedding()
)
collectionMinilm33m= ChromaDB.create_collection(
    name="IDEAsiaFacts3",
    # embedding_function=llama3p2Embedding()
    embedding_function=minilm33mEmbedding()
    # embedding_function=commandREmbedding()
    # embedding_function=snowflakeEmbedding()
    # embedding_function=testEmbedding()
)

collectionSnowflake137m = ChromaDB.create_collection(
    name="IDEAsiaFacts4",
    # embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
    # embedding_function=commandREmbedding()
    embedding_function=snowflake137mEmbedding()
    # embedding_function=testEmbedding()
)

collectionSnowflake22m = ChromaDB.create_collection(
    name="IDEAsiaFacts5",
    # embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
    # embedding_function=commandREmbedding()
    embedding_function=snowflake22mEmbedding()
    # embedding_function=testEmbedding()
)

collectionGemma= ChromaDB.create_collection(
    name="IDEAsiaFacts6",
    # embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
    # embedding_function=commandREmbedding()
    embedding_function=gemmaEmbedding()
    # embedding_function=testEmbedding()
)

collectionQwen3 = ChromaDB.create_collection(
    name="IDEAsiaFacts7",
    # embedding_function=llama3p2Embedding()
    # embedding_function=minilmEmbedding()
    # embedding_function=commandREmbedding()
    embedding_function=qwen3Embedding()
    # embedding_function=testEmbedding()
)

collectionMinilm22m.add(
    ids=ids,
    documents=docs
)
collectionMinilm33m.add(
    ids=ids,
    documents=docs
)
collectionSnowflake22m.add(
    ids=ids,
    documents=docs
)
collectionSnowflake33m.add(
    ids=ids,
    documents=docs
)
collectionSnowflake137m.add(
    ids=ids,
    documents=docs
)
collectionGemma.add(
    ids=ids,
    documents=docs
)
collectionQwen3.add(
    ids=ids,
    documents=docs
)


questions = [
  "Lokasi kantornya dimana ?",
  "Email kalian apa ?",
  "Perusahaan ini bergerak di bidang apa ?",
  "No telpnya berapa ?",
"pusatnya dimana ?"
]

for question in questions:
    print(f"pertanyaan : {question}")

    # print(f"Snowflake22m")
    # query = collectionSnowflake22m.query(
    #     query_texts=[question], # Chroma will embed this for you
    #     n_results=5 # how many results to return
    # )
    # konteks = "\n".join(query["documents"][0])
    # print(konteks)
    # time.sleep(3)
    # print(f"")

    print(f"Minilm22m")
    query = collectionMinilm22m.query(
        query_texts=[question], # Chroma will embed this for you
        n_results=5 # how many results to return
    )
    konteks = "\n".join(query["documents"][0])
    print(konteks)
    time.sleep(3)
    print(f"")

    print(f"qwen3")
    query = collectionQwen3.query(
        query_texts=[question], # Chroma will embed this for you
        n_results=5 # how many results to return
    )
    konteks = "\n".join(query["documents"][0])
    print(konteks)
    time.sleep(3)
    print(f"")

    print(f"Gemma Embedding")
    query = collectionGemma.query(
        query_texts=[question], # Chroma will embed this for you
        n_results=5 # how many results to return
    )
    konteks = "\n".join(query["documents"][0])
    print(konteks)
    time.sleep(3)
    print(f"")

    # print(f"Snowflake137m")
    # query = collectionSnowflake137m.query(
    #     query_texts=[question], # Chroma will embed this for you
    #     n_results=5 # how many results to return
    # )
    # konteks = "\n".join(query["documents"][0])
    # print(konteks)
    # time.sleep(3)
    # print(f"")

    print(f"")
    print(f"")

# this test has proven that minilm22m is the best at retrieveg information from question
# using other model and using bigger version result in worse les rellevant result