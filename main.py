from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import argparse
import pathlib
import torch
import sys

@dataclass
class Doc:
    id: int
    text: str

def load_lines(path: pathlib.Path) -> List[Doc]:
    """讀取文字檔，每一行視為一段；自動去除空行與前後空白。"""
    lines = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        t = raw.strip()
        if t:
            lines.append(Doc(id=len(lines), text=t))
    if not lines:
        raise ValueError("資料為空，請確認檔案內容。")
    return lines

def embed_texts(model: SentenceTransformer, docs: List[Doc]) -> torch.Tensor:
    texts = [d.text for d in docs]
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def search(
    model: SentenceTransformer,
    doc_embeddings: torch.Tensor,
    docs: List[Doc],
    query: str,
    top_k: int = 3
) -> List[Tuple[float, Doc]]:
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q, doc_embeddings)[0]   # shape: (N,)
    top_k = min(top_k, len(docs))
    top = torch.topk(scores, k=top_k)
    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        results.append((float(score), docs[idx]))
    return results

def main():
    parser = argparse.ArgumentParser(description="Mini Semantic Searcher")
    parser.add_argument("--data", type=str, default="data/sample.txt", help="語料檔路徑 (每行一段)")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer 模型名稱")
    parser.add_argument("--top_k", type=int, default=3, help="每次查詢回傳幾筆結果")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"找不到語料檔：{data_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("🚀 載入語料...")
    docs = load_lines(data_path)

    print("🧠 載入模型（第一次會自動下載）...")
    model = SentenceTransformer(args.model)

    print("📦 向量化語料...")
    doc_embeddings = embed_texts(model, docs)

    print("\n✅ 準備完成！輸入查詢開始搜尋，輸入 q 離開。")
    while True:
        try:
            query = input("\n🔎 查詢：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break
        if not query:
            continue
        if query.lower() == "q":
            print("再見！")
            break

        results = search(model, doc_embeddings, docs, query, top_k=args.top_k)
        print("\n📚 最相關結果：")
        for rank, (score, doc) in enumerate(results, start=1):
            print(f"{rank:>2}. [相似度 {score:.4f}] {doc.text}")

if __name__ == "__main__":
    main()
