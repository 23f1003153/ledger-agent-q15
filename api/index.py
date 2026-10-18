import os, re, json
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- 1. LOAD + CLEAN DATA (runs once at startup) ----------
RATES = {"USD": 1.0, "EUR": 1.03, "INR": 0.01229}   # usd per 1 unit

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(HERE, "..", "export.jsonl"),
           os.path.join(HERE, "export.jsonl"),
           "export.jsonl"]:
    if os.path.exists(_p):
        orders = pd.read_json(_p, lines=True)
        break

# tidy text columns
for c in ["status", "currency", "region", "product", "customer", "id"]:
    orders[c] = orders[c].astype(str).str.strip()
orders["status"] = orders["status"].str.lower()
orders["currency"] = orders["currency"].str.upper()

# dates: mixed "+05:30" and "Z" -> parse as real UTC instants
orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True, format="ISO8601")
orders["updated_at"] = pd.to_datetime(orders["updated_at"], utc=True, format="ISO8601")

# keep only the latest version of each order id
orders = (orders.sort_values("updated_at")
                .drop_duplicates("id", keep="last")
                .reset_index(drop=True))

# business date = India time
ist = orders["created_at"].dt.tz_convert("Asia/Kolkata")
orders["order_date"] = ist.dt.date
orders["year"] = ist.dt.year
orders["month"] = ist.dt.month
orders["quarter"] = ist.dt.quarter

# money in USD (do NOT round each row; round only the final answer)
orders["usd"] = orders["amount"] * orders["currency"].map(RATES)

paid = orders[orders["status"] == "paid"]
refunded = orders[orders["status"] == "refunded"]

# ---------- 2. LLM THAT WRITES PANDAS CODE ----------
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LLM_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4.1-mini"

def preview(df):
    return df.head(3).to_string()

SYSTEM = f"""You answer questions about Acme Appliances' order ledger by writing pandas code.

DataFrames already loaded (cleaned, one row per order id, latest version only):
- orders   : ALL orders, any status
- paid     : only status == 'paid'
- refunded : only status == 'refunded'
Columns: {list(orders.columns)}
Sample rows:
{preview(orders)}

Facts:
- `usd` is the order amount already converted to USD. Always use `usd` for money.
- `year`, `month`, `quarter`, `order_date` are already in India time (Asia/Kolkata), based on created_at.
- Regions seen: {sorted(orders['region'].unique())}
- Products seen: {sorted(orders['product'].unique())}
- Statuses seen: {sorted(orders['status'].unique())}
- Revenue = sum of `usd` over `paid` rows.
- Refund amount = sum of `usd` over `refunded` rows. Refund count = number of rows in `refunded`.
- Units sold = sum of `qty`. Customers are ids like 'C1094' (no names). Count customers with nunique().
- Use product and region names exactly as in the lists above (match case-insensitively to the user's wording).
- Do not round inside the calculation; the caller rounds.

Write Python that sets a variable named `result` to the final answer:
- money -> a number (float), counts -> int, a name -> exact string from the data, a list -> Python list.
Return ONLY code. No markdown, no explanation."""

def ask_llm(messages, timeout):
    r = httpx.post(LLM_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={"model": MODEL, "temperature": 0, "messages": messages},
        timeout=timeout)
    data = r.json()
    if "choices" not in data:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text[:500]}")
    code = data["choices"][0]["message"]["content"]
    return re.sub(r"^```(?:python)?\s*|```\s*$", "", code.strip(), flags=re.M).strip()

def run_code(code):
    scope = {"orders": orders, "paid": paid, "refunded": refunded, "pd": pd, "np": np}
    exec(code, scope)
    res = scope["result"]
    if isinstance(res, (np.integer,)):
        return int(res)
    if isinstance(res, (np.floating, float)):
        return round(float(res), 2)
    if isinstance(res, (pd.Series, pd.DataFrame)):
        return json.loads(res.to_json())
    return res

def answer(question):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    code = ask_llm(msgs, timeout=9)
    try:
        return run_code(code)
    except Exception as e:
        msgs += [{"role": "assistant", "content": code},
                 {"role": "user", "content": f"That code failed: {e}. Fix it. Return only code."}]
        return run_code(ask_llm(msgs, timeout=5))

# ---------- 3. WEB SERVER ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str

@app.post("/")
@app.post("/api/index")
def ask(q: Q):
    return {"answer": answer(q.question)}

@app.get("/")
@app.get("/api/index")
def health():
    return {"status": "ok"}
