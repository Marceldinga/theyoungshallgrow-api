from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from supabase import create_client
except Exception as e:
    raise RuntimeError("Missing dependency: supabase-py. Add it to requirements.txt") from e


# =============================================================================
# CONFIG
# =============================================================================
APP_NAME = "theyoungshallgrow-api (younchat)"
DEFAULT_SCHEMA = os.getenv("SUPABASE_SCHEMA", "public").strip() or "public"

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_ROUTER_COMPLETIONS_URL = "https://router.huggingface.co/v1/completions"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ✅ ONLY these 3 models (hard-locked)
HF_ALLOWED_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# ✅ Allowlist the relations younchat can read
RELATIONS: Dict[str, Dict[str, Any]] = {
    # Tables
    "members": {"type": "table", "truth": True},
    "contributions": {"type": "table"},
    "foundation_contributions": {"type": "table"},
    "loans": {"type": "table"},
    "loan_payments": {"type": "table"},
    "fines": {"type": "table"},
    "payouts": {"type": "table"},
    "sessions": {"type": "table"},
    "minutes": {"type": "table"},
    "attendance": {"type": "table"},
    "signatures": {"type": "table"},
    "audit_log": {"type": "table"},
    "app_state": {"type": "table"},
    "loan_requests": {"type": "table"},
    "loan_repayments_pending": {"type": "table"},
    "profiles": {"type": "table"},
    "ml_training_data": {"type": "table"},
    "member_contribution_totals": {"type": "table"},
    "interest_ledger": {"type": "table"},

    # Views (optional)
    "v_dashboard_kpis": {"type": "view"},
    "v_finance_kpis": {"type": "view"},
    "v_member_financial_totals": {"type": "view"},
    "v_loans_with_member": {"type": "view"},
    "v_loan_payments_with_member": {"type": "view"},
    "v_contributions_with_member": {"type": "view"},
    "v_foundation_contributions_with_member": {"type": "view"},
    "v_payouts_with_member": {"type": "view"},
    "v_next_beneficiary": {"type": "view"},
    "v_loans_dpd": {"type": "view"},
    "v_loans_next_interest": {"type": "view"},
    "v_loans_next_interest_with_member": {"type": "view"},
    "v_loan_power_status": {"type": "view"},
    "v_attendance_all_time_per_member": {"type": "view"},
    "v_attendance_by_member_session": {"type": "view"},
    "v_attendance_member_totals": {"type": "view"},
    "v_attendance_with_member": {"type": "view"},
}

# ✅ EXACT intro line required by you
def _intro_only() -> str:
    return "Hello 👋🏽 I’m younchat — your Njangi assistant."


# =============================================================================
# ENV / CLIENTS
# =============================================================================
def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()

SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = _env("SUPABASE_SERVICE_KEY")

HF_TOKEN = _env("HF_TOKEN")
HF_FORCE_MODE = _env("HF_FORCE_MODE", "auto").lower()

TAVILY_API_KEY = _env("TAVILY_API_KEY")
INTERNET_MODE = _env("INTERNET_MODE", "off").lower()

def _internet_enabled() -> bool:
    if INTERNET_MODE == "off":
        return False
    return bool(TAVILY_API_KEY)

def _supabase_clients():
    # Backend should use service key if present; fallback to anon for public-read only.
    sb_anon = None
    sb_service = None
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        sb_anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        sb_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return sb_anon, sb_service

SB_ANON, SB_SERVICE = _supabase_clients()


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title=APP_NAME)

# CORS: allow Flutter web / local dev. You can tighten later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS (API)
# =============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    schema: Optional[str] = None

    # Flutter should pass these back each time (stateless server).
    last_member_id: Optional[str] = None

    # Optional: client-side chat history (for HF wording only)
    history: Optional[List[Dict[str, str]]] = None  # [{"role":"user"/"assistant","content":"..."}]


class ChatResponse(BaseModel):
    reply: str
    used_source: str
    member_id_focus: Optional[str] = None
    dataframe: Optional[Dict[str, Any]] = None  # {"title": "...", "rows": [...], "columns": [...]}
    meta: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# HELPERS
# =============================================================================
def _clean(text: str) -> str:
    return (text or "").strip()

def _lc(text: str) -> str:
    return _clean(text).lower()

def _force_hello_prefix(text: str) -> str:
    t = _clean(text)
    if not t:
        return "Hello 👋🏽"
    if not t.lower().startswith("hello"):
        return "Hello 👋🏽 " + t
    return t

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def _to_float(x: Any) -> float:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0

def _fmt(x: Any) -> str:
    return f"{_to_float(x):,.2f}"

def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{x * 100:.1f}%"
    except Exception:
        return "—"

def _ratio(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _safe_sum(df: pd.DataFrame, col: Optional[str]) -> float:
    if df is None or df.empty or not col or col not in df.columns:
        return 0.0
    return float(_to_num_series(df[col]).sum())

def _db_proof_line(row_counts: Dict[str, int]) -> str:
    ts = _utc_now()
    if not row_counts:
        return f"DB Proof: (no row counts) • fetched_at={ts}"
    parts = [f"{k}={int(v)}" for k, v in row_counts.items()]
    return f"DB Proof: {', '.join(parts)} • fetched_at={ts}"

def _relation_guard(rel: str) -> None:
    if rel not in RELATIONS:
        raise HTTPException(status_code=400, detail=f"Relation not allowed: {rel}")

def _sb_select(
    schema: str,
    relation: str,
    cols: str = "*",
    limit: int = 2000,
    filters: Optional[List[Tuple[str, str, Any]]] = None,
    order: Optional[Tuple[str, bool]] = None,
) -> pd.DataFrame:
    _relation_guard(relation)
    sb = SB_SERVICE or SB_ANON
    if sb is None:
        return pd.DataFrame()

    def _apply(q):
        if filters:
            for col, op, val in filters:
                if val is None:
                    continue
                if op == "eq":
                    q = q.eq(col, val)
                elif op == "gte":
                    q = q.gte(col, val)
                elif op == "lte":
                    q = q.lte(col, val)
                elif op == "ilike":
                    q = q.ilike(col, val)
                elif op == "in":
                    q = q.in_(col, val)  # type: ignore
        if order:
            col, asc = order
            q = q.order(col, desc=not asc)
        return q

    # Prefer schema; fallback if older supabase-py behavior
    try:
        q = sb.schema(schema).table(relation).select(cols).limit(limit)
        q = _apply(q)
        res = q.execute()
        return pd.DataFrame(getattr(res, "data", None) or [])
    except Exception:
        try:
            q = sb.table(relation).select(cols).limit(limit)
            q = _apply(q)
            res = q.execute()
            return pd.DataFrame(getattr(res, "data", None) or [])
        except Exception:
            return pd.DataFrame()

def _rpc_finance_snapshot(schema: str) -> Dict[str, Any]:
    """
    Calls fn_finance_snapshot() if you created it in Supabase.
    Returns {} if missing/blocked.
    """
    sb = SB_SERVICE or SB_ANON
    if sb is None:
        return {}
    try:
        res = sb.schema(schema).rpc("fn_finance_snapshot", {}).execute()
    except Exception:
        try:
            res = sb.rpc("fn_finance_snapshot", {}).execute()
        except Exception:
            return {}

    data = getattr(res, "data", None)
    if not data:
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return {}

def _snapshot_to_metrics(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None

    # Nested format
    if isinstance(snapshot.get("totals"), dict) or isinstance(snapshot.get("counts"), dict) or isinstance(snapshot.get("ratios"), dict):
        totals = snapshot.get("totals") or {}
        counts = snapshot.get("counts") or {}
        ratios = snapshot.get("ratios") or {}
        return {
            "notes": [],
            "row_counts": {k: int(v) for k, v in (counts or {}).items() if v is not None},
            "total_contributions": totals.get("total_contributions"),
            "foundation_total": totals.get("foundation_total"),
            "total_fines": totals.get("total_fines"),
            "active_loan_exposure": totals.get("active_loan_exposure"),
            "unpaid_interest": totals.get("unpaid_interest"),
            "interest_total": totals.get("interest_ledger_total") or totals.get("interest_total"),
            "active_loan_count": counts.get("active_loans") or counts.get("active_loan_count") or 0,
            "overdue_loan_count": counts.get("overdue_loans") or counts.get("overdue_loan_count") or 0,
            "overdue_ratio": ratios.get("overdue_ratio"),
            "liquidity_pressure_ratio": ratios.get("liquidity_pressure_ratio"),
        }

    # Flat format
    rc = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    return {
        "notes": [],
        "row_counts": {k: int(v) for k, v in (rc or {}).items() if v is not None},
        "total_contributions": snapshot.get("total_contributions"),
        "foundation_total": snapshot.get("foundation_total"),
        "total_fines": snapshot.get("total_fines"),
        "active_loan_exposure": snapshot.get("active_loan_exposure"),
        "unpaid_interest": snapshot.get("unpaid_interest"),
        "interest_total": snapshot.get("interest_ledger_total") or snapshot.get("interest_total"),
        "active_loan_count": snapshot.get("active_loan_count") or 0,
        "overdue_loan_count": snapshot.get("overdue_loan_count") or 0,
        "overdue_ratio": snapshot.get("overdue_ratio"),
        "liquidity_pressure_ratio": snapshot.get("liquidity_pressure_ratio"),
    }


# =============================================================================
# INTENT DETECTION (DB commands vs HF wording)
# =============================================================================
def _wants_internet(text: str) -> bool:
    t = _lc(text)
    return t.startswith("web:") or t.startswith("internet:") or t.startswith("tavily:")

def _strip_web_prefix(q: str) -> str:
    return re.sub(r"^(web:|internet:|tavily:)\s*", "", (q or "").strip(), flags=re.IGNORECASE).strip()

def _wants_tables_list(text: str) -> bool:
    return _lc(text) in {"tables", "relations", "views", "list tables", "list views"}

def _wants_describe(text: str) -> bool:
    t = _lc(text)
    return t.startswith("describe ") or t.startswith("columns ") or t.startswith("cols ") or t.startswith("schema ")

def _wants_show_table(text: str) -> bool:
    t = _lc(text)
    return t.startswith("show ") or t.startswith("preview ") or t.startswith("open ")

def _wants_list_members(text: str) -> bool:
    t = _lc(text)
    phrases = [
        "list all members",
        "list members",
        "show all members",
        "show members",
        "members list",
        "all members",
        "member list",
        "who are the members",
        "member ids",
    ]
    return t in {"members", "member"} or any(p in t for p in phrases)

def _wants_kpis(text: str) -> bool:
    t = _lc(text)
    return any(k in t for k in ["kpi", "kpis", "finance kpi", "finance kpis", "dashboard kpi"])

def _wants_loans(text: str) -> bool:
    t = _lc(text)
    return any(k in t for k in ["loan", "loans", "borrow", "repay", "repayment", "overdue", "dpd", "interest due"])

def _wants_financial_review(text: str) -> bool:
    t = _lc(text)
    triggers = [
        "how are we doing",
        "are we stable",
        "is njangi healthy",
        "njangi health",
        "health score",
        "financial condition",
        "risk review",
        "any risk",
        "liquidity",
        "credit risk",
        "executive summary",
        "summary",
        "control tower",
        "financial intelligence",
    ]
    return any(x in t for x in triggers)

def _wants_verify_member(text: str) -> bool:
    t = _lc(text)
    return t.startswith("verify member ") or t.startswith("verify ")

def _extract_verify_member_id(text: str) -> Optional[str]:
    t = _lc(text)
    t = re.sub(r"^verify(\s+member)?\s+", "", t).strip()
    m = re.search(r"(\d+)", t)
    return m.group(1) if m else None

def _extract_relation_name(text: str) -> Optional[str]:
    t = _lc(text)
    t = re.sub(r"^(show|preview|open|describe|columns|cols|schema)\s+", "", t).strip()
    t = re.sub(r"^table\s+", "", t).strip()
    t = re.sub(r"[^\w]+$", "", t)
    if not t:
        return None
    token = t.split()[0]
    return token if token in RELATIONS else None

_MEMBER_ID_PATTERNS = [
    re.compile(r"\bmember[_\s-]?id\s*[:=#]?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bmember\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bid\s*[:=#]?\s*(\d+)\b", re.IGNORECASE),
]

def _extract_member_id(text: str) -> Optional[str]:
    t = _clean(text)
    if not t:
        return None
    if t.isdigit():
        return t
    for pat in _MEMBER_ID_PATTERNS:
        m = pat.search(t)
        if m:
            return str(m.group(1))
    return None

def _is_db_command(text: str) -> bool:
    t = _lc(text)
    if not t:
        return False
    if t in RELATIONS:
        return True
    if _wants_list_members(t) or _wants_loans(t) or _wants_kpis(t) or _wants_tables_list(t):
        return True
    if _wants_show_table(t) or _wants_describe(t) or _wants_verify_member(t):
        return True
    finance_words = [
        "contribution", "contributions", "payout", "payouts",
        "loan", "loans", "repayment", "interest", "unpaid",
        "overdue", "balance", "exposure", "liquidity",
        "foundation", "kpi", "kpis", "risk", "health score",
        "grade", "total",
    ]
    return any(w in t for w in finance_words)


# =============================================================================
# MEMBERS TRUTH (source of truth)
# =============================================================================
def _load_members_truth(schema: str, limit: int = 3000) -> pd.DataFrame:
    df = _sb_select(schema, "members", cols="*", limit=limit)
    if df.empty:
        return df

    id_col = _pick_col(df, ["id", "member_id"])
    name_col = _pick_col(df, ["name", "full_name"])
    display_col = _pick_col(df, ["display_name"])

    if not id_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["member_id"] = df[id_col].astype(str)

    disp_clean = (
        df[display_col].astype(str).replace(["None", "nan", "NaN", "NULL", "null"], "").fillna("").str.strip()
        if display_col and display_col in df.columns
        else pd.Series([""] * len(df))
    )
    nm_clean = (
        df[name_col].astype(str).replace(["None", "nan", "NaN", "NULL", "null"], "").fillna("").str.strip()
        if name_col and name_col in df.columns
        else pd.Series([""] * len(df))
    )

    out["member_name"] = disp_clean.where(disp_clean != "", nm_clean).fillna("").replace("", "(no name)")

    try:
        out["_id_num"] = pd.to_numeric(out["member_id"], errors="coerce")
        out = out.sort_values(["_id_num", "member_id"], ascending=True).drop(columns=["_id_num"])
    except Exception:
        pass

    return out

def _member_name_from_truth(members_truth: pd.DataFrame, member_id: str) -> str:
    if members_truth is None or members_truth.empty:
        return "(unknown)"
    hit = members_truth[members_truth["member_id"].astype(str) == str(member_id)]
    if hit.empty:
        return "(unknown)"
    return str(hit.iloc[0]["member_name"])

def _member_exists(members_truth: pd.DataFrame, member_id: str) -> bool:
    if members_truth is None or members_truth.empty:
        return False
    return not members_truth[members_truth["member_id"].astype(str) == str(member_id)].empty


# =============================================================================
# FINANCE (DB grounded)
# =============================================================================
def _active_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans
    status_col = _pick_col(loans, ["status"])
    if not status_col:
        return loans
    s = loans[status_col].astype(str).str.lower().fillna("")
    active_status = {"active", "open", "ongoing", "overdue", "late", "running", "disbursed"}
    return loans[s.isin(active_status)]

def _overdue_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans
    status_col = _pick_col(loans, ["status"])
    if status_col:
        s = loans[status_col].astype(str).str.lower().fillna("")
        return loans[s.isin({"overdue", "late"})]
    dpd_col = _pick_col(loans, ["dpd", "days_past_due", "overdue_days"])
    if dpd_col:
        dpd = _to_num_series(loans[dpd_col])
        return loans[dpd > 0]
    return loans.iloc[0:0]

def _loan_balance_col(loans: pd.DataFrame) -> Optional[str]:
    # ✅ your schema uses principal_current
    return _pick_col(loans, ["principal_current", "outstanding_principal", "principal_remaining", "principal", "amount"])

def _unpaid_interest_col(loans: pd.DataFrame) -> Optional[str]:
    return _pick_col(loans, ["unpaid_interest", "interest_unpaid", "interest_due", "interest_balance"])

def _member_risk_grade(active_bal: float, unpaid: float) -> str:
    if active_bal <= 0 and unpaid <= 0:
        return "A"
    if active_bal > 0 and unpaid <= 0:
        return "B"
    return "C"

def _compute_member_totals_from_tables(schema: str, member_id: str) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []

    contributions = _sb_select(schema, "contributions", cols="*", limit=200000, filters=[("member_id", "eq", member_id)])
    foundation = _sb_select(schema, "foundation_contributions", cols="*", limit=200000, filters=[("member_id", "eq", member_id)])
    fines = _sb_select(schema, "fines", cols="*", limit=200000, filters=[("member_id", "eq", member_id)])
    loans = _sb_select(schema, "loans", cols="*", limit=200000, filters=[("member_id", "eq", member_id)])
    interest_ledger = _sb_select(schema, "interest_ledger", cols="*", limit=200000, filters=[("member_id", "eq", member_id)])

    contrib_col = _pick_col(contributions, ["amount"])
    found_col = _pick_col(foundation, ["amount"])
    fines_col = _pick_col(fines, ["amount"])
    interest_col = _pick_col(interest_ledger, ["amount"])

    active = _active_loan_filter(loans)
    bal_col = _loan_balance_col(active)
    unpaid_col = _unpaid_interest_col(active)

    if contrib_col is None and not contributions.empty:
        notes.append("Missing contributions amount column (expected: amount).")
    if found_col is None and not foundation.empty:
        notes.append("Missing foundation_contributions amount column (expected: amount).")
    if fines_col is None and not fines.empty:
        notes.append("Missing fines amount column (expected: amount).")
    if bal_col is None and not active.empty:
        notes.append("Missing loans balance column (expected: principal_current or principal).")
    if unpaid_col is None and not active.empty:
        notes.append("Missing loans unpaid interest column (expected: unpaid_interest).")

    out = {
        "source": "tables",
        "contributions_total": _safe_sum(contributions, contrib_col),
        "foundation_total": _safe_sum(foundation, found_col),
        "fines_total": _safe_sum(fines, fines_col),
        "active_loan_balance": _safe_sum(active, bal_col) if bal_col else 0.0,
        "active_unpaid_interest": _safe_sum(active, unpaid_col) if unpaid_col else 0.0,
        "interest_total": _safe_sum(interest_ledger, interest_col),
        "_rows": {
            "members": 1,
            "contributions": int(len(contributions)),
            "foundation_contributions": int(len(foundation)),
            "fines": int(len(fines)),
            "loans": int(len(loans)),
            "interest_ledger": int(len(interest_ledger)),
        },
    }
    return out, notes

def _collect_global_finance(schema: str) -> Dict[str, Any]:
    # Snapshot-first
    snap = _rpc_finance_snapshot(schema)
    if snap:
        return {"ok": True, "notes": [], "snapshot": snap}

    # Fallback: view (if exists)
    ctx: Dict[str, Any] = {"ok": True, "notes": ["Snapshot unavailable → fallback compute."], "df": {}}
    if "v_finance_kpis" in RELATIONS:
        ctx["df"]["v_finance_kpis"] = _sb_select(schema, "v_finance_kpis", cols="*", limit=200)
    ctx["df"]["contributions"] = _sb_select(schema, "contributions", cols="*", limit=200000)
    ctx["df"]["foundation_contributions"] = _sb_select(schema, "foundation_contributions", cols="*", limit=200000)
    ctx["df"]["loans"] = _sb_select(schema, "loans", cols="*", limit=200000)
    ctx["df"]["interest_ledger"] = _sb_select(schema, "interest_ledger", cols="*", limit=200000)
    ctx["df"]["fines"] = _sb_select(schema, "fines", cols="*", limit=200000)
    return ctx

def _compute_global_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    snap = ctx.get("snapshot") or {}
    snap_metrics = _snapshot_to_metrics(snap) if isinstance(snap, dict) else None
    if snap_metrics is not None:
        return snap_metrics

    dfc = (ctx.get("df") or {}).get("contributions", pd.DataFrame())
    dff = (ctx.get("df") or {}).get("foundation_contributions", pd.DataFrame())
    dfl = (ctx.get("df") or {}).get("loans", pd.DataFrame())
    dfi = (ctx.get("df") or {}).get("interest_ledger", pd.DataFrame())
    dffines = (ctx.get("df") or {}).get("fines", pd.DataFrame())

    notes: List[str] = []

    contrib_col = _pick_col(dfc, ["amount"])
    total_contributions: Optional[float] = _safe_sum(dfc, contrib_col) if contrib_col else (0.0 if dfc.empty else None)

    foundation_col = _pick_col(dff, ["amount"])
    foundation_total: Optional[float] = _safe_sum(dff, foundation_col) if foundation_col else (0.0 if dff.empty else None)

    fines_col = _pick_col(dffines, ["amount"])
    total_fines: Optional[float] = _safe_sum(dffines, fines_col) if fines_col else (0.0 if dffines.empty else None)

    active_loans = _active_loan_filter(dfl)
    overdue_loans = _overdue_loan_filter(active_loans)

    bal_col = _loan_balance_col(active_loans)
    active_loan_exposure: Optional[float] = _safe_sum(active_loans, bal_col) if bal_col else (0.0 if active_loans.empty else None)

    unpaid_col = _unpaid_interest_col(active_loans)
    unpaid_interest: Optional[float] = _safe_sum(active_loans, unpaid_col) if unpaid_col else (0.0 if active_loans.empty else None)

    active_count = int(len(active_loans)) if active_loans is not None else 0
    overdue_count = int(len(overdue_loans)) if overdue_loans is not None else 0
    overdue_ratio: Optional[float] = (overdue_count / active_count) if active_count > 0 else (0.0 if overdue_count == 0 else None)

    interest_col = _pick_col(dfi, ["amount"])
    interest_total: Optional[float] = _safe_sum(dfi, interest_col) if interest_col else (0.0 if dfi.empty else None)

    liquidity_pressure = _ratio(active_loan_exposure, total_contributions) if active_loan_exposure is not None else None

    row_counts = {
        "contributions": int(len(dfc)),
        "foundation_contributions": int(len(dff)),
        "loans": int(len(dfl)),
        "interest_ledger": int(len(dfi)),
        "fines": int(len(dffines)),
    }

    return {
        "notes": notes,
        "row_counts": row_counts,
        "total_contributions": total_contributions,
        "foundation_total": foundation_total,
        "total_fines": total_fines,
        "active_loan_exposure": active_loan_exposure,
        "active_loan_count": active_count,
        "overdue_loan_count": overdue_count,
        "overdue_ratio": overdue_ratio,
        "unpaid_interest": unpaid_interest,
        "interest_total": interest_total,
        "liquidity_pressure_ratio": liquidity_pressure,
    }

def _risk_classification(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    signals: List[str] = []
    lpr = metrics.get("liquidity_pressure_ratio")
    overdue_ratio = metrics.get("overdue_ratio")
    unpaid_interest = metrics.get("unpaid_interest")

    if lpr is not None and lpr > 0.75:
        signals.append("Liquidity pressure > 75% (Active Loan Exposure ÷ Total Contributions).")
    if overdue_ratio is not None and overdue_ratio > 0.20:
        signals.append("Overdue ratio is elevated (over 20% of active loans).")
    if unpaid_interest is not None and unpaid_interest > 0:
        signals.append("Unpaid interest exists on active loans.")

    score = 0
    if lpr is not None:
        score += 2 if lpr > 0.75 else (1 if lpr > 0.50 else 0)
    if overdue_ratio is not None:
        score += 2 if overdue_ratio > 0.30 else (1 if overdue_ratio > 0.10 else 0)
    if unpaid_interest is not None:
        score += 1 if unpaid_interest > 0 else 0

    if score >= 5:
        return "High", signals
    if score >= 3:
        return "Elevated", signals
    if score >= 1:
        return "Moderate", signals
    return "Low", signals

def _build_control_tower_report(metrics: Dict[str, Any]) -> str:
    risk_label, signals = _risk_classification(metrics)
    lines: List[str] = []
    lines.append("Hello 👋🏽 Njangi Financial Intelligence Review (DB-grounded)\n")

    lines.append("1️⃣ Current Situation")
    lines.append(f"- Total contributions: **{_fmt(metrics.get('total_contributions'))}**" if metrics.get("total_contributions") is not None else "- Total contributions: **Not available**")
    lines.append(f"- Foundation reserves (total): **{_fmt(metrics.get('foundation_total'))}**" if metrics.get("foundation_total") is not None else "- Foundation reserves: **Not available**")
    lines.append(f"- Active loan exposure: **{_fmt(metrics.get('active_loan_exposure'))}**" if metrics.get("active_loan_exposure") is not None else "- Active loan exposure: **Not available**")
    lines.append(f"- Active loans (count): **{int(metrics.get('active_loan_count', 0) or 0)}**")
    lines.append(f"- Overdue loans (count): **{int(metrics.get('overdue_loan_count', 0) or 0)}**")
    lines.append(f"- Overdue ratio: **{_pct(metrics.get('overdue_ratio'))}**" if metrics.get("overdue_ratio") is not None else "- Overdue ratio: **Not available**")
    lines.append(f"- Unpaid interest (active): **{_fmt(metrics.get('unpaid_interest'))}**" if metrics.get("unpaid_interest") is not None else "- Unpaid interest: **Not available**")
    lines.append(f"- Liquidity Pressure Ratio (Exposure ÷ Contributions): **{_pct(metrics.get('liquidity_pressure_ratio'))}**" if metrics.get("liquidity_pressure_ratio") is not None else "- Liquidity Pressure Ratio: **Not available**")
    lines.append(f"- Interest ledger total: **{_fmt(metrics.get('interest_total'))}**" if metrics.get("interest_total") is not None else "- Interest ledger total: **Not available**")

    lines.append("\n2️⃣ Risk Assessment")
    lines.append(f"- Risk classification: **{risk_label}**")
    if signals:
        lines.append("- Early warning signals:")
        for s in signals:
            lines.append(f"  - {s}")
    else:
        lines.append("- Early warning signals: **None detected**")

    lines.append("\n🧾 DB Proof")
    lines.append(f"- {_db_proof_line(metrics.get('row_counts') or {})}")

    notes = metrics.get("notes") or []
    if notes:
        lines.append("\n🔒 Data Integrity Notes")
        for n in notes:
            lines.append(f"- {n}")

    return "\n".join(lines)


# =============================================================================
# TAVILY (optional)
# =============================================================================
def _tavily_search(query: str) -> Dict[str, Any]:
    if not _internet_enabled():
        return {"ok": False, "error": "Internet is OFF", "results": []}

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        r = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
        if r.status_code >= 400:
            return {"ok": False, "error": f"Tavily error {r.status_code}: {r.text[:300]}", "results": []}
        data = r.json() or {}
        results = data.get("results") or []
        clean = []
        for it in results:
            clean.append({"title": it.get("title"), "url": it.get("url"), "content": (it.get("content") or "")[:300]})
        return {"ok": True, "results": clean}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


# =============================================================================
# HF ROUTER (foundation model) — only for NON-DB prompts
# =============================================================================
def _post_with_retries(url: str, headers: dict, payload: dict, timeout: int = 60) -> Tuple[bool, str]:
    last_err = ""
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HF error {r.status_code}: {r.text[:600]}"
                time.sleep(1.0 + attempt * 1.5)
                continue
            if r.status_code >= 400:
                return False, f"HF error {r.status_code}: {r.text[:600]}"
            return True, r.text
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 + attempt * 1.5)
    return False, last_err or "HF transient error"

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    out: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(f"[SYSTEM]\n{content}\n")
        elif role == "assistant":
            out.append(f"[ASSISTANT]\n{content}\n")
        else:
            out.append(f"[USER]\n{content}\n")
    out.append("[ASSISTANT]\n")
    return "\n".join(out)

def _hf_router_chat(model: str, token: str, messages: List[Dict[str, str]], timeout: int = 60) -> Tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 650}
    ok, raw = _post_with_retries(HF_ROUTER_CHAT_URL, headers, payload, timeout=timeout)
    if not ok:
        return False, raw
    try:
        data = json.loads(raw)
        text = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        return True, str(text).strip()
    except Exception:
        return False, f"Bad HF chat response: {raw[:600]}"

def _hf_router_completions(model: str, token: str, prompt: str, timeout: int = 60) -> Tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "temperature": 0.2, "max_tokens": 650}
    ok, raw = _post_with_retries(HF_ROUTER_COMPLETIONS_URL, headers, payload, timeout=timeout)
    if not ok:
        return False, raw
    try:
        data = json.loads(raw)
        text = ((data.get("choices") or [{}])[0].get("text") or "")
        return True, str(text).strip()
    except Exception:
        return False, f"Bad HF completions response: {raw[:600]}"

def _younchat_hf_system_prompt() -> str:
    return (
        "You are younchat — the Autonomous Financial Intelligence Engine for the Njangi platform \"theyoungshallgrow\".\n"
        "ABSOLUTE DATA INTEGRITY:\n"
        "- NEVER invent numbers.\n"
        "- NEVER guess balances, totals, dates, counts, or member IDs.\n"
        "- NEVER output SQL.\n"
        "- NEVER output Python.\n"
        "If user asks for Njangi numbers, tell them to use DB commands (members, loans, finance kpis, tables, show/describe).\n"
        "Style: professional, analytical, bullet-structured. Start with Hello.\n"
    )

def _looks_like_code_output(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t:
        return False
    if "```" in t:
        return True
    code_markers = ["import ", "def ", "class ", "select ", "create table", "alter table", "drop table"]
    return any(m in t for m in code_markers)

def _hf_call(token: str, messages: List[Dict[str, str]]) -> Tuple[bool, str, str, str]:
    force = (HF_FORCE_MODE or "auto").strip().lower()
    prompt = _messages_to_prompt(messages)
    model_order = list(HF_ALLOWED_MODELS)

    def _looks_instruct(mname: str) -> bool:
        mlc = (mname or "").lower()
        return any(x in mlc for x in ["instruct", "mistral", "llama-3", "llama-3.1"])

    def _should_try_next(err_text: str) -> bool:
        e = (err_text or "").lower()
        return any(s in e for s in ["404", "not found", "429", "500", "502", "503", "504", "timeout", "server error", "not supported"])

    last_err = ""
    last_mode = "failed"
    last_model = model_order[0] if model_order else ""

    for chosen in model_order:
        last_model = chosen
        if force == "chat":
            order = ["chat"]
        elif force == "completions":
            order = ["completions"]
        else:
            order = ["completions", "chat"] if _looks_instruct(chosen) else ["chat", "completions"]

        for mode in order:
            last_mode = mode
            if mode == "completions":
                ok, txt = _hf_router_completions(chosen, token, prompt)
                if ok and txt:
                    return True, txt, "completions", chosen
                last_err = txt
            else:
                ok, txt = _hf_router_chat(chosen, token, messages)
                if ok and txt:
                    return True, txt, "chat", chosen
                last_err = txt

        if not _should_try_next(last_err):
            break

    return False, last_err or "Unknown HF error", last_mode, last_model


# =============================================================================
# CORE CHAT ROUTER (DB first)
# =============================================================================
def _df_payload(title: str, df: pd.DataFrame, limit: int = 200) -> Dict[str, Any]:
    if df is None:
        return {"title": title, "columns": [], "rows": []}
    if len(df) > limit:
        df = df.head(limit)
    return {
        "title": title,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }

def _member_report_tables_only(schema: str, member_id: str, members_truth: pd.DataFrame) -> str:
    if not _member_exists(members_truth, member_id):
        return (
            "Hello 👋🏽 I can’t confirm that member_id exists in `members` (source of truth). "
            "Type **members** to verify IDs, then retry."
        )

    name = _member_name_from_truth(members_truth, member_id)
    table_totals, table_notes = _compute_member_totals_from_tables(schema, member_id)

    active_bal = _to_float(table_totals.get("active_loan_balance"))
    unpaid = _to_float(table_totals.get("active_unpaid_interest"))
    grade = _member_risk_grade(active_bal, unpaid)

    lines: List[str] = []
    lines.append("Hello 👋🏽 Member Financial Intelligence (DB-grounded)\n")
    lines.append("1️⃣ Current Situation")
    lines.append(f"- Member: **{name}** (member_id={member_id})")
    lines.append(f"- Contributions total: **{_fmt(table_totals.get('contributions_total'))}**")
    lines.append(f"- Foundation total: **{_fmt(table_totals.get('foundation_total'))}**")
    lines.append(f"- Fines total: **{_fmt(table_totals.get('fines_total'))}**")
    lines.append(f"- Active loan balance: **{_fmt(table_totals.get('active_loan_balance'))}**")
    lines.append(f"- Active unpaid interest: **{_fmt(table_totals.get('active_unpaid_interest'))}**")
    lines.append(f"- Interest ledger total: **{_fmt(table_totals.get('interest_total'))}**")

    lines.append("\n2️⃣ Risk Assessment")
    lines.append(f"- Member Risk Grade: **{grade}** (A/B/C)")

    lines.append("\n🧾 DB Proof")
    lines.append(f"- {_db_proof_line(table_totals.get('_rows', {}))}")

    if table_notes:
        lines.append("\n🔒 Data Integrity Notes")
        for n in table_notes:
            lines.append(f"- {n}")

    return "\n".join(lines)

def _handle_db_commands(schema: str, q: str, last_member_id: Optional[str]) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Returns: reply, used_source, member_id_focus, dataframe_payload
    """
    members_truth = _load_members_truth(schema=schema, limit=3000)

    # web:
    if _wants_internet(q):
        if not _internet_enabled():
            return "Hello 👋🏽 Internet is OFF. Set TAVILY_API_KEY and INTERNET_MODE=on.", "tavily:off", last_member_id, None
        query = _strip_web_prefix(q)
        res = _tavily_search(query)
        if not res.get("ok"):
            return f"Hello 👋🏽 Internet error: {res.get('error')}", "tavily:error", last_member_id, None
        items = res.get("results") or []
        if not items:
            return "Hello 👋🏽 No web results found.", "tavily:none", last_member_id, None
        lines = ["Hello 👋🏽 Here are the top web results:\n"]
        for it in items[:5]:
            title = it.get("title") or "Source"
            url = it.get("url") or ""
            snippet = (it.get("content") or "").strip()
            lines.append(f"- {title} — {url}" if url else f"- {title}")
            if snippet:
                lines.append(f"  - {snippet[:180]}…")
        return "\n".join(lines), "tavily", last_member_id, None

    # tables/relations
    if _wants_tables_list(q):
        rows = [{"relation": k, "type": RELATIONS[k].get("type", "?")} for k in sorted(RELATIONS.keys())]
        df = pd.DataFrame(rows)
        return "Hello 👋🏽 Here are the tables/views younchat can read:", "relations", last_member_id, _df_payload("Readable relations (allowlist)", df)

    # describe
    if _wants_describe(q):
        rel = _extract_relation_name(q)
        if not rel:
            return "Hello 👋🏽 Say: **describe loans** (or any allowlisted table/view).", "describe:help", last_member_id, None
        df = _sb_select(schema, rel, cols="*", limit=1)
        cols = list(df.columns) if df is not None else []
        out = pd.DataFrame({"column_name": cols})
        return f"Hello 👋🏽 Columns for **{rel}** ({RELATIONS[rel]['type']}):", f"describe:{rel}", last_member_id, _df_payload(f"Columns: {rel}", out)

    # show/preview
    if _wants_show_table(q):
        rel = _extract_relation_name(q)
        if not rel:
            return "Hello 👋🏽 Say: **show contributions** (or any allowlisted table/view).", "show:help", last_member_id, None
        df = _sb_select(schema, rel, cols="*", limit=2000)
        return f"Hello 👋🏽 Preview of **{rel}** ({RELATIONS[rel]['type']}):", f"show:{rel}", last_member_id, _df_payload(f"Preview: {rel}", df)

    # members list
    if _wants_list_members(q):
        if members_truth is None or members_truth.empty:
            return "Hello 👋🏽 I couldn’t read **members** (source of truth). Check RLS/permissions.", "members:error", last_member_id, None
        # Keep it readable
        lines = ["Hello 👋🏽 Here are all members (from `members`):\n"]
        for r in members_truth.itertuples(index=False):
            lines.append(f"- **{r.member_id}** • {r.member_name}")
        return "\n".join(lines), "members", last_member_id, _df_payload("members (truth)", members_truth)

    # KPIs
    if _wants_kpis(q):
        if "v_finance_kpis" in RELATIONS:
            df = _sb_select(schema, "v_finance_kpis", cols="*", limit=200)
            if df.empty:
                return "Hello 👋🏽 No KPI rows returned.", "v_finance_kpis", last_member_id, None
            return "Hello 👋🏽 Finance KPIs (from `v_finance_kpis`):", "v_finance_kpis", last_member_id, _df_payload("Finance KPIs", df)
        return "Hello 👋🏽 v_finance_kpis not available.", "kpis:fallback", last_member_id, None

    # Loans
    if _wants_loans(q):
        mid = _extract_member_id(q) or last_member_id
        if "v_loans_with_member" in RELATIONS:
            filters = [("member_id", "eq", mid)] if mid else None
            df = _sb_select(schema, "v_loans_with_member", cols="*", limit=5000, filters=filters)
            src = "v_loans_with_member"
        else:
            filters = [("member_id", "eq", mid)] if mid else None
            df = _sb_select(schema, "loans", cols="*", limit=5000, filters=filters)
            src = "loans"
        title = "Loans" if not mid else f"Loans for {_member_name_from_truth(members_truth, mid)} (member_id={mid})"
        if df.empty:
            return f"Hello 👋🏽 {title}: no rows returned.", src, mid, None
        return f"Hello 👋🏽 {title} (from `{src}`):", src, mid, _df_payload(title, df)

    # Financial review (control tower)
    if _wants_financial_review(q):
        ctx = _collect_global_finance(schema)
        metrics = _compute_global_metrics(ctx)
        return _build_control_tower_report(metrics), "finance_intel", last_member_id, None

    # Verify member (simple: tables-only report; you can extend to compare view vs tables later)
    if _wants_verify_member(q):
        mid = _extract_verify_member_id(q) or last_member_id
        if not mid:
            return "Hello 👋🏽 Say: **verify member 10**", "verify:help", last_member_id, None
        # For Flutter: keep it tight
        return _member_report_tables_only(schema, str(mid), members_truth), "verify:tables", str(mid), None

    # Member report when user just types a number
    mid2 = _extract_member_id(q)
    if mid2:
        return _member_report_tables_only(schema, str(mid2), members_truth), "member:tables", str(mid2), None

    # direct relation name
    if _lc(q) in RELATIONS:
        rel = _lc(q)
        df = _sb_select(schema, rel, cols="*", limit=2000)
        return f"Hello 👋🏽 Preview of **{rel}**:", f"show:{rel}", last_member_id, _df_payload(f"Preview: {rel}", df)

    # If it smells like DB but didn't match a command
    return (
        "Hello 👋🏽 I can answer using your real Njangi database only.\n\n"
        "Try:\n"
        "- **members**\n"
        "- **loans**\n"
        "- **finance kpis**\n"
        "- **tables**\n"
        "- **show contributions**\n"
        "- **describe loans**\n"
        "- Ask: **How are we doing?** (Control Tower Review)\n"
        "- Or type a member_id like: **5**\n"
    ), "db:guide", last_member_id, None


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "time": datetime.now(timezone.utc).isoformat(),
        "supabase_url_set": bool(SUPABASE_URL),
        "supabase_anon_set": bool(SUPABASE_ANON_KEY),
        "supabase_service_set": bool(SUPABASE_SERVICE_KEY),
        "hf_token_set": bool(HF_TOKEN),
        "hf_models_locked": HF_ALLOWED_MODELS,
        "internet": "ON" if _internet_enabled() else "OFF",
        "schema_default": DEFAULT_SCHEMA,
    }

@app.get("/relations")
def relations():
    return [{"relation": k, "type": RELATIONS[k].get("type")} for k in sorted(RELATIONS.keys())]

@app.get("/describe/{relation}")
def describe(relation: str, schema: str = DEFAULT_SCHEMA):
    _relation_guard(relation)
    df = _sb_select(schema, relation, cols="*", limit=1)
    return {"relation": relation, "type": RELATIONS[relation]["type"], "columns": list(df.columns)}

@app.get("/preview/{relation}")
def preview(relation: str, schema: str = DEFAULT_SCHEMA, limit: int = 50):
    _relation_guard(relation)
    limit = max(1, min(int(limit), 2000))
    df = _sb_select(schema, relation, cols="*", limit=limit)
    return _df_payload(f"Preview: {relation}", df, limit=limit)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = _clean(req.message)
    if not q:
        raise HTTPException(status_code=400, detail="message required")

    schema = (req.schema or DEFAULT_SCHEMA).strip() or DEFAULT_SCHEMA
    last_member_id = _clean(req.last_member_id or "") or None

    # Update focus from message if it contains a member id
    detected = _extract_member_id(q)
    if detected:
        last_member_id = detected

    # 1) DB tools first
    if _is_db_command(q) or _wants_internet(q):
        reply, used_source, member_focus, df = _handle_db_commands(schema, q, last_member_id)
        reply = reply if reply == _intro_only() else _force_hello_prefix(reply)
        return ChatResponse(
            reply=reply,
            used_source=used_source,
            member_id_focus=member_focus,
            dataframe=df,
            meta={
                "schema": schema,
                "internet": "ON" if _internet_enabled() else "OFF",
                "hf_token_set": bool(HF_TOKEN),
            },
        )

    # 2) If not a DB command, use HF foundation model for wording (optional)
    if HF_TOKEN:
        system = _younchat_hf_system_prompt()

        # build a short context from client history
        history = req.history or []
        trimmed = history[-10:] if len(history) > 10 else history

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for m in trimmed:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # include current message
        messages.append({"role": "user", "content": q})

        ok, txt, mode, model_used = _hf_call(HF_TOKEN, messages)
        used_source = f"hf:{mode}:{model_used}" if ok else f"hf:failed:{model_used}"

        if not ok:
            reply = f"Hello 👋🏽 HF is not reachable: {txt}"
        else:
            if _looks_like_code_output(txt):
                reply = (
                    "Hello 👋🏽 I can’t output code (SQL/Python). "
                    "Use: **members**, **loans**, **finance kpis**, **tables**, **show <table>**, **describe <table>**, or type a **member_id**."
                )
                used_source = used_source + ":blocked_code"
            else:
                reply = _force_hello_prefix(txt)

        return ChatResponse(
            reply=reply,
            used_source=used_source,
            member_id_focus=last_member_id,
            dataframe=None,
            meta={
                "schema": schema,
                "internet": "ON" if _internet_enabled() else "OFF",
                "hf_token_set": True,
            },
        )

    # 3) No HF token → fallback
    return ChatResponse(
        reply="Hello 👋🏽",
        used_source="local:fallback",
        member_id_focus=last_member_id,
        dataframe=None,
        meta={"schema": schema, "internet": "ON" if _internet_enabled() else "OFF", "hf_token_set": False},
    )
