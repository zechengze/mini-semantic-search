# Mini Semantic Searcher

一個簡單的 **語意搜尋 (Semantic Search)** Demo，使用 [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)。  

---

## 功能
- 使用 Sentence Transformers 將文件轉換為向量表示
- 利用餘弦相似度 (cosine similarity) 進行語意搜尋
- 提供互動式 CLI，可輸入問題查詢最相關的文件
- 架構簡單，方便擴充（如 PDF/Word/Excel 載入器，或 Web 介面）

---

## 專案結構
mini_semantic_search/
│── main.py # 主程式（語意搜尋核心）
│── requirements.txt # 相依套件
│── README.md # 專案說明文件
│
├── data/
│ └── sample.txt # 範例語料庫（每行一個文件）
├── .gitignore
└── .venv/ # 虛擬環境（不會提交到 Git）

---

## 安裝
```bash
git clone https://github.com/zechengze/mini-semantic-search.git
cd mini-semantic-search

# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安裝套件
pip install -r requirements.txt

使用方式
python main.py

範例查詢：
如何快速做資料儀表板
cosine similarity 是什麼
什麼是 RAG
輸入 q 退出程式。

---