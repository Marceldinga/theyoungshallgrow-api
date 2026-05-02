from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    raise RuntimeError("Missing dependency: supabase-py. Add `supabase` to requirements.txt") from e


APP_NAME = "theyoungshallgrow-api (younchat MACCO Brain)"
DEFAULT_SCHEMA = (os.getenv("SUPABASE_SCHEMA", "public").strip() or "public")

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_ROUTER_COMPLETIONS_URL = "https://router.huggingface.co/v1/completions"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

HF_ALLOWED_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

RELATIONS: Dict[str, Dict[str, Any]] = {
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


def _intro_only() -> str:
    return "Hello 👋🏽 I’m younchat — your Njangi assistant."


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = _env("SUPABASE_SERVICE_KEY")
HF_TOKEN = _env("HF_TOKEN")
TAVILY_API_KEY = _env("TAVILY_API_KEY")
INTERNET_MODE = _env("INTERNET_MODE", "off").lower()

_SUPABASE_INIT_ERROR = ""


def _internet_enabled() -> bool:
    return INTERNET_MODE != "off" and bool(TAVILY_API_KEY)


def _clean_env_value(v: Optional[str]) -> str:
    if not v:
        return ""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1].strip()
    return v


def _supabase_clients():
    global _SUPABASE_INIT_ERROR

    url = _clean_env_value(SUPABASE_URL)
    anon = _clean_env_value(SUPABASE_ANON_KEY)
    service = _clean_env_value(SUPABASE_SERVICE_KEY)

    sb_anon = None
    sb_service = None

    if not url:
        _SUPABASE_INIT_ERROR = "SUPABASE_URL missing"
        return None, None

    if anon:
        try:
            sb_anon = create_client(url, anon)
        except Exception as e:
            _SUPABASE_INIT_ERROR = f"Anon key error: {e}"

    if service:
        try:
            sb_service = create_client(url, service)
        except Exception as e:
            msg = f"Service key error: {e}"
            _SUPABASE_INIT_ERROR = f"{_SUPABASE_INIT_ERROR} | {msg}" if _SUPABASE_INIT_ERROR else msg

    return sb_anon, sb_service


SB_ANON, SB_SERVICE = _supabase_clients()

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    schema: Optional[str] = None
    last_member_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    reply: str
    used_source: str
    member_id_focus: Optional[str] = None
    dataframe: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


def _clean(text: str) -> str:
    return (text or "").strip()


def _lc(text: str) -> str:
    return _clean(text).lower()


def _force_hello_prefix(text: str) -> str:
    t = _clean(text)
    if not t:
        return "Hello 👋🏽"
    return t if t.lower().startswith("hello") else "Hello 👋🏽 " + t


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
    parts = [f"{k}={int(v)}" for k, v in row_counts.items()]
    return f"DB Proof: {', '.join(parts) if parts else '(no row counts)'} • fetched_at={_utc_now()}"


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
                    q = q.in_(col, val)
        if order:
            col, asc = order
            q = q.order(col, desc=not asc)
        return q

    try:
        q = sb.schema(schema).table(relation).select(cols).limit(limit)
        res = _apply(q).execute()
        return pd.DataFrame(getattr(res, "data", None) or [])
    except Exception:
        try:
            q = sb.table(relation).select(cols).limit(limit)
            res = _apply(q).execute()
            return pd.DataFrame(getattr(res, "data", None) or [])
        except Exception:
            return pd.DataFrame()


def _rpc_finance_snapshot(schema: str) -> Dict[str, Any]:
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

    totals = snapshot.get("totals") if isinstance(snapshot.get("totals"), dict) else snapshot
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    ratios = snapshot.get("ratios") if isinstance(snapshot.get("ratios"), dict) else {}

    return {
        "notes": [],
        "row_counts": {k: int(v) for k, v in counts.items() if v is not None},
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
    phrases = ["list all members", "list members", "show all members", "show members", "members list", "all members", "member list", "who are the members", "member ids"]
    return t in {"members", "member"} or any(p in t for p in phrases)


def _wants_kpis(text: str) -> bool:
    return any(k in _lc(text) for k in ["kpi", "kpis", "finance kpi", "finance kpis", "dashboard kpi"])


def _wants_loans(text: str) -> bool:
    return any(k in _lc(text) for k in ["loan", "loans", "borrow", "repay", "repayment", "overdue", "dpd", "interest due"])


def _wants_financial_review(text: str) -> bool:
    t = _lc(text)
    triggers = ["how are we doing", "are we stable", "is njangi healthy", "njangi health", "health score", "financial condition", "risk review", "any risk", "liquidity", "credit risk", "executive summary", "summary", "control tower", "financial intelligence"]
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
    token = t.split()[0] if t else ""
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
    finance_words = ["contribution", "contributions", "payout", "payouts", "loan", "loans", "repayment", "interest", "unpaid", "overdue", "balance", "exposure", "liquidity", "foundation", "kpi", "kpis", "risk", "health score", "grade", "total"]
    return any(w in t for w in finance_words)


def _logic_profile(text: str) -> Dict[str, Any]:
    t = re.sub(r"[^\w\s]", " ", _lc(text))
    words = t.split()
    deductive = {"all", "every", "each", "no", "none", "if", "then", "must", "therefore"}
    inductive = {"most", "many", "probably", "likely", "usually", "sample", "trend", "average"}
    d = sum(1 for w in words if w in deductive)
    i = sum(1 for w in words if w in inductive)
    mode = "deductive" if d > i else "inductive" if i > d else "mixed"
    return {"mode": mode, "deductive_score": d, "inductive_score": i, "confidence": min(0.95, 0.5 + 0.08 * max(d, i))}


def _macco_brain(q: str, last_member_id: Optional[str]) -> Dict[str, Any]:
    mid = _extract_member_id(q) or last_member_id
    intent = "general"
    use_db = False
    use_web = False
    use_hf = True
    confidence = 0.50

    if _wants_internet(q):
        intent, use_web, use_hf, confidence = "internet_search", True, False, 0.98
    elif _is_db_command(q):
        intent, use_db, use_hf, confidence = "database_financial_intelligence", True, False, 0.94
    elif mid:
        intent, use_db, use_hf, confidence = "member_financial_report", True, False, 0.93

    return {
        "agent": "MACCO",
        "intent": intent,
        "confidence": confidence,
        "use_db": use_db,
        "use_web": use_web,
        "use_hf": use_hf,
        "member_id": mid,
        "logic": _logic_profile(q),
    }


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

    disp = df[display_col].astype(str).replace(["None", "nan", "NaN", "NULL", "null"], "").fillna("").str.strip() if display_col and display_col in df.columns else pd.Series([""] * len(df))
    nm = df[name_col].astype(str).replace(["None", "nan", "NaN", "NULL", "null"], "").fillna("").str.strip() if name_col and name_col in df.columns else pd.Series([""] * len(df))

    out["member_name"] = disp.where(disp != "", nm).fillna("").replace("", "(no name)")

    try:
        out["_id_num"] = pd.to_numeric(out["member_id"], errors="coerce")
        out = out.sort_values(["_id_num", "member_id"]).drop(columns=["_id_num"])
    except Exception:
        pass

    return out


def _member_name_from_truth(members_truth: pd.DataFrame, member_id: str) -> str:
    if members_truth is None or members_truth.empty:
        return "(unknown)"
    hit = members_truth[members_truth["member_id"].astype(str) == str(member_id)]
    return "(unknown)" if hit.empty else str(hit.iloc[0]["member_name"])


def _member_exists(members_truth: pd.DataFrame, member_id: str) -> bool:
    return members_truth is not None and not members_truth.empty and not members_truth[members_truth["member_id"].astype(str) == str(member_id)].empty


def _active_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans
    status_col = _pick_col(loans, ["status"])
    if not status_col:
        return loans
    active_status = {"active", "open", "ongoing", "overdue", "late", "running", "disbursed"}
    return loans[loans[status_col].astype(str).str.lower().fillna("").isin(active_status)]


def _overdue_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans
    status_col = _pick_col(loans, ["status"])
    if status_col:
        return loans[loans[status_col].astype(str).str.lower().fillna("").isin({"overdue", "late"})]
    dpd_col = _pick_col(loans, ["dpd", "days_past_due", "overdue_days"])
    if dpd_col:
        return loans[_to_num_series(loans[dpd_col]) > 0]
    return loans.iloc[0:0]


def _loan_balance_col(loans: pd.DataFrame) -> Optional[str]:
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
    contributions = _sb_select(schema, "contributions", limit=200000, filters=[("member_id", "eq", member_id)])
    foundation = _sb_select(schema, "foundation_contributions", limit=200000, filters=[("member_id", "eq", member_id)])
    fines = _sb_select(schema, "fines", limit=200000, filters=[("member_id", "eq", member_id)])
    loans = _sb_select(schema, "loans", limit=200000, filters=[("member_id", "eq", member_id)])
    interest_ledger = _sb_select(schema, "interest_ledger", limit=200000, filters=[("member_id", "eq", member_id)])

    active = _active_loan_filter(loans)
    return {
        "contributions_total": _safe_sum(contributions, _pick_col(contributions, ["amount"])),
        "foundation_total": _safe_sum(foundation, _pick_col(foundation, ["amount"])),
        "fines_total": _safe_sum(fines, _pick_col(fines, ["amount"])),
        "active_loan_balance": _safe_sum(active, _loan_balance_col(active)),
        "active_unpaid_interest": _safe_sum(active, _unpaid_interest_col(active)),
        "interest_total": _safe_sum(interest_ledger, _pick_col(interest_ledger, ["amount"])),
        "_rows": {
            "members": 1,
            "contributions": int(len(contributions)),
            "foundation_contributions": int(len(foundation)),
            "fines": int(len(fines)),
            "loans": int(len(loans)),
            "interest_ledger": int(len(interest_ledger)),
        },
    }, []


def _collect_global_finance(schema: str) -> Dict[str, Any]:
    snap = _rpc_finance_snapshot(schema)
    if snap:
        return {"ok": True, "notes": [], "snapshot": snap}
    return {
        "ok": True,
        "notes": ["Snapshot unavailable → fallback compute."],
        "df": {
            "contributions": _sb_select(schema, "contributions", limit=200000),
            "foundation_contributions": _sb_select(schema, "foundation_contributions", limit=200000),
            "loans": _sb_select(schema, "loans", limit=200000),
            "interest_ledger": _sb_select(schema, "interest_ledger", limit=200000),
            "fines": _sb_select(schema, "fines", limit=200000),
        },
    }


def _compute_global_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    snap = ctx.get("snapshot") or {}
    snap_metrics = _snapshot_to_metrics(snap) if isinstance(snap, dict) else None
    if snap_metrics is not None:
        return snap_metrics

    dfc = ctx["df"].get("contributions", pd.DataFrame())
    dff = ctx["df"].get("foundation_contributions", pd.DataFrame())
    dfl = ctx["df"].get("loans", pd.DataFrame())
    dfi = ctx["df"].get("interest_ledger", pd.DataFrame())
    dffines = ctx["df"].get("fines", pd.DataFrame())

    active = _active_loan_filter(dfl)
    overdue = _overdue_loan_filter(active)

    total_contributions = _safe_sum(dfc, _pick_col(dfc, ["amount"]))
    active_loan_exposure = _safe_sum(active, _loan_balance_col(active))
    active_count = int(len(active))
    overdue_count = int(len(overdue))

    return {
        "notes": ctx.get("notes", []),
        "row_counts": {
            "contributions": int(len(dfc)),
            "foundation_contributions": int(len(dff)),
            "loans": int(len(dfl)),
            "interest_ledger": int(len(dfi)),
            "fines": int(len(dffines)),
        },
        "total_contributions": total_contributions,
        "foundation_total": _safe_sum(dff, _pick_col(dff, ["amount"])),
        "total_fines": _safe_sum(dffines, _pick_col(dffines, ["amount"])),
        "active_loan_exposure": active_loan_exposure,
        "active_loan_count": active_count,
        "overdue_loan_count": overdue_count,
        "overdue_ratio": overdue_count / active_count if active_count else 0.0,
        "unpaid_interest": _safe_sum(active, _unpaid_interest_col(active)),
        "interest_total": _safe_sum(dfi, _pick_col(dfi, ["amount"])),
        "liquidity_pressure_ratio": _ratio(active_loan_exposure, total_contributions),
    }


def _risk_classification(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    signals = []
    lpr = metrics.get("liquidity_pressure_ratio")
    overdue_ratio = metrics.get("overdue_ratio")
    unpaid_interest = metrics.get("unpaid_interest")

    score = 0
    if lpr is not None:
        score += 2 if lpr > 0.75 else 1 if lpr > 0.50 else 0
        if lpr > 0.75:
            signals.append("Liquidity pressure > 75% (Active Loan Exposure ÷ Total Contributions).")
    if overdue_ratio is not None:
        score += 2 if overdue_ratio > 0.30 else 1 if overdue_ratio > 0.10 else 0
        if overdue_ratio > 0.20:
            signals.append("Overdue ratio is elevated.")
    if unpaid_interest is not None and unpaid_interest > 0:
        score += 1
        signals.append("Unpaid interest exists on active loans.")

    return ("High" if score >= 5 else "Elevated" if score >= 3 else "Moderate" if score >= 1 else "Low", signals)


def _build_control_tower_report(metrics: Dict[str, Any]) -> str:
    risk_label, signals = _risk_classification(metrics)
    lines = [
        "Hello 👋🏽 Njangi Financial Intelligence Review (DB-grounded)\n",
        "1️⃣ Current Situation",
        f"- Total contributions: **{_fmt(metrics.get('total_contributions'))}**",
        f"- Foundation reserves: **{_fmt(metrics.get('foundation_total'))}**",
        f"- Active loan exposure: **{_fmt(metrics.get('active_loan_exposure'))}**",
        f"- Active loans: **{int(metrics.get('active_loan_count', 0) or 0)}**",
        f"- Overdue loans: **{int(metrics.get('overdue_loan_count', 0) or 0)}**",
        f"- Overdue ratio: **{_pct(metrics.get('overdue_ratio'))}**",
        f"- Unpaid interest: **{_fmt(metrics.get('unpaid_interest'))}**",
        f"- Liquidity pressure ratio: **{_pct(metrics.get('liquidity_pressure_ratio'))}**",
        f"- Interest ledger total: **{_fmt(metrics.get('interest_total'))}**",
        "\n2️⃣ Risk Assessment",
        f"- Risk classification: **{risk_label}**",
    ]
    lines.append("- Early warning signals: **None detected**" if not signals else "- Early warning signals:")
    for s in signals:
        lines.append(f"  - {s}")
    lines.append("\n🧾 DB Proof")
    lines.append(f"- {_db_proof_line(metrics.get('row_counts') or {})}")
    return "\n".join(lines)


def _tavily_search(query: str) -> Dict[str, Any]:
    if not _internet_enabled():
        return {"ok": False, "error": "Internet is OFF", "results": []}
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 5, "include_answer": False, "include_raw_content": False}
    try:
        r = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
        if r.status_code >= 400:
            return {"ok": False, "error": f"Tavily error {r.status_code}: {r.text[:300]}", "results": []}
        data = r.json() or {}
        return {"ok": True, "results": [{"title": it.get("title"), "url": it.get("url"), "content": (it.get("content") or "")[:300]} for it in data.get("results", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def _post_with_retries(url: str, headers: dict, payload: dict, timeout: int = 60) -> Tuple[bool, str]:
    last_err = ""
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HF error {r.status_code}: {r.text[:600]}"
                time.sleep(1 + attempt)
                continue
            if r.status_code >= 400:
                return False, f"HF error {r.status_code}: {r.text[:600]}"
            return True, r.text
        except Exception as e:
            last_err = str(e)
            time.sleep(1 + attempt)
    return False, last_err or "HF transient error"


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    out = []
    for m in messages:
        out.append(f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}\n")
    out.append("[ASSISTANT]\n")
    return "\n".join(out)


def _hf_router_chat(model: str, token: str, messages: List[Dict[str, str]], timeout: int = 60) -> Tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 650}
    ok, raw = _post_with_retries(HF_ROUTER_CHAT_URL, headers, payload, timeout)
    if not ok:
        return False, raw
    try:
        data = json.loads(raw)
        return True, str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    except Exception:
        return False, f"Bad HF chat response: {raw[:600]}"


def _hf_router_completions(model: str, token: str, prompt: str, timeout: int = 60) -> Tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "temperature": 0.2, "max_tokens": 650}
    ok, raw = _post_with_retries(HF_ROUTER_COMPLETIONS_URL, headers, payload, timeout)
    if not ok:
        return False, raw
    try:
        data = json.loads(raw)
        return True, str(((data.get("choices") or [{}])[0].get("text") or "")).strip()
    except Exception:
        return False, f"Bad HF completions response: {raw[:600]}"


def _younchat_hf_system_prompt() -> str:
    return (
        "You are younchat — the Autonomous Financial Intelligence Engine for the Njangi platform the young shall grow. "
        "Never invent numbers. Never guess balances, totals, dates, counts, or member IDs. "
        "Never output SQL or Python. Start with Hello. Be professional, analytical, and bullet-structured."
    )


def _looks_like_code_output(txt: str) -> bool:
    t = (txt or "").strip().lower()
    return bool(t) and ("```" in t or any(m in t for m in ["import ", "def ", "class ", "select ", "create table", "alter table", "drop table"]))


def _score_model_answer(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 1.0
    low = t.lower()
    if low.startswith("hello"):
        score += 1
    if any(x in low for x in ["analysis", "summary", "risk", "because", "recommend", "therefore"]):
        score += 2
    if "-" in t or "1️⃣" in t:
        score += 1
    words = len(t.split())
    if 40 <= words <= 250:
        score += 1.5
    elif words > 400:
        score -= 1
    if _looks_like_code_output(t):
        score -= 10
    return score


def _hf_single_model_call(model: str, token: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = _messages_to_prompt(messages)
    candidates = []

    ok, txt = _hf_router_chat(model, token, messages)
    if ok and txt:
        candidates.append({"model": model, "mode": "chat", "ok": True, "text": txt, "score": _score_model_answer(txt)})

    ok, txt = _hf_router_completions(model, token, prompt)
    if ok and txt:
        candidates.append({"model": model, "mode": "completions", "ok": True, "text": txt, "score": _score_model_answer(txt)})

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[0] if candidates else {"model": model, "mode": "failed", "ok": False, "text": "", "score": 0.0}


def _model_council(results: List[Dict[str, Any]]) -> Tuple[bool, str, str, str, Dict[str, Any]]:
    good = [r for r in results if r.get("ok") and r.get("text")]
    if not good:
        return False, "All HF models failed.", "parallel", "none", {"models": results}

    safe = [r for r in good if not _looks_like_code_output(r["text"])]
    if not safe:
        return True, "Hello 👋🏽 I can’t output code here. Use database commands like members, loans, finance kpis, tables, show <table>, describe <table>, or type a member_id.", "parallel_blocked_code", "model_council", {"models": results}

    for r in safe:
        words = set(r["text"].lower().split())
        for other in safe:
            if r is not other:
                r["score"] += min(len(words & set(other["text"].lower().split())) / 50, 1.5)

    best = sorted(safe, key=lambda x: x["score"], reverse=True)[0]
    meta = {
        "winner": best["model"],
        "winner_mode": best["mode"],
        "models": [{"model": r["model"], "mode": r["mode"], "ok": r["ok"], "score": round(r["score"], 4), "preview": r["text"][:180]} for r in results],
    }
    return True, best["text"], "parallel_council", best["model"], meta


def _hf_call(token: str, messages: List[Dict[str, str]]) -> Tuple[bool, str, str, str, Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=len(HF_ALLOWED_MODELS)) as executor:
        futures = {executor.submit(_hf_single_model_call, model, token, messages): model for model in HF_ALLOWED_MODELS}
        for future in as_completed(futures):
            model = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"model": model, "mode": "error", "ok": False, "text": str(e), "score": 0.0})
    return _model_council(results)


def _df_payload(title: str, df: pd.DataFrame, limit: int = 200) -> Dict[str, Any]:
    if df is None:
        return {"title": title, "columns": [], "rows": []}
    df = df.head(limit) if len(df) > limit else df
    return {"title": title, "columns": list(df.columns), "rows": df.to_dict(orient="records")}


def _member_report_tables_only(schema: str, member_id: str, members_truth: pd.DataFrame) -> str:
    if not _member_exists(members_truth, member_id):
        return "Hello 👋🏽 I can’t confirm that member_id exists in `members`. Type **members** to verify IDs, then retry."

    name = _member_name_from_truth(members_truth, member_id)
    totals, notes = _compute_member_totals_from_tables(schema, member_id)
    grade = _member_risk_grade(_to_float(totals.get("active_loan_balance")), _to_float(totals.get("active_unpaid_interest")))

    return "\n".join([
        "Hello 👋🏽 Member Financial Intelligence (DB-grounded)\n",
        "1️⃣ Current Situation",
        f"- Member: **{name}** (member_id={member_id})",
        f"- Contributions total: **{_fmt(totals.get('contributions_total'))}**",
        f"- Foundation total: **{_fmt(totals.get('foundation_total'))}**",
        f"- Fines total: **{_fmt(totals.get('fines_total'))}**",
        f"- Active loan balance: **{_fmt(totals.get('active_loan_balance'))}**",
        f"- Active unpaid interest: **{_fmt(totals.get('active_unpaid_interest'))}**",
        f"- Interest ledger total: **{_fmt(totals.get('interest_total'))}**",
        "\n2️⃣ Risk Assessment",
        f"- Member Risk Grade: **{grade}**",
        "\n🧾 DB Proof",
        f"- {_db_proof_line(totals.get('_rows', {}))}",
    ])


def _handle_db_commands(schema: str, q: str, last_member_id: Optional[str]) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    members_truth = _load_members_truth(schema=schema, limit=3000)

    if _wants_internet(q):
        if not _internet_enabled():
            return "Hello 👋🏽 Internet is OFF. Set TAVILY_API_KEY and INTERNET_MODE=on.", "tavily:off", last_member_id, None
        res = _tavily_search(_strip_web_prefix(q))
        if not res.get("ok"):
            return f"Hello 👋🏽 Internet error: {res.get('error')}", "tavily:error", last_member_id, None
        lines = ["Hello 👋🏽 Here are the top web results:\n"]
        for it in (res.get("results") or [])[:5]:
            lines.append(f"- {it.get('title') or 'Source'} — {it.get('url') or ''}")
            if it.get("content"):
                lines.append(f"  - {it.get('content')[:180]}…")
        return "\n".join(lines), "tavily", last_member_id, None

    if _wants_tables_list(q):
        df = pd.DataFrame([{"relation": k, "type": RELATIONS[k].get("type", "?")} for k in sorted(RELATIONS.keys())])
        return "Hello 👋🏽 Here are the tables/views younchat can read:", "relations", last_member_id, _df_payload("Readable relations", df)

    if _wants_describe(q):
        rel = _extract_relation_name(q)
        if not rel:
            return "Hello 👋🏽 Say: **describe loans**.", "describe:help", last_member_id, None
        df = _sb_select(schema, rel, limit=1)
        return f"Hello 👋🏽 Columns for **{rel}**:", f"describe:{rel}", last_member_id, _df_payload(f"Columns: {rel}", pd.DataFrame({"column_name": list(df.columns)}))

    if _wants_show_table(q):
        rel = _extract_relation_name(q)
        if not rel:
            return "Hello 👋🏽 Say: **show contributions**.", "show:help", last_member_id, None
        df = _sb_select(schema, rel, limit=2000)
        return f"Hello 👋🏽 Preview of **{rel}**:", f"show:{rel}", last_member_id, _df_payload(f"Preview: {rel}", df)

    if _wants_list_members(q):
        if members_truth.empty:
            return "Hello 👋🏽 I couldn’t read **members**. Check RLS/permissions.", "members:error", last_member_id, None
        lines = ["Hello 👋🏽 Here are all members:\n"]
        for r in members_truth.itertuples(index=False):
            lines.append(f"- **{r.member_id}** • {r.member_name}")
        return "\n".join(lines), "members", last_member_id, _df_payload("members", members_truth)

    if _wants_kpis(q):
        df = _sb_select(schema, "v_finance_kpis", limit=200)
        if df.empty:
            return "Hello 👋🏽 No KPI rows returned.", "v_finance_kpis", last_member_id, None
        return "Hello 👋🏽 Finance KPIs:", "v_finance_kpis", last_member_id, _df_payload("Finance KPIs", df)

    if _wants_loans(q):
        mid = _extract_member_id(q) or last_member_id
        filters = [("member_id", "eq", mid)] if mid else None
        src = "v_loans_with_member" if "v_loans_with_member" in RELATIONS else "loans"
        df = _sb_select(schema, src, limit=5000, filters=filters)
        title = "Loans" if not mid else f"Loans for {_member_name_from_truth(members_truth, mid)} (member_id={mid})"
        if df.empty:
            return f"Hello 👋🏽 {title}: no rows returned.", src, mid, None
        return f"Hello 👋🏽 {title}:", src, mid, _df_payload(title, df)

    if _wants_financial_review(q):
        metrics = _compute_global_metrics(_collect_global_finance(schema))
        return _build_control_tower_report(metrics), "finance_intel", last_member_id, None

    if _wants_verify_member(q):
        mid = _extract_verify_member_id(q) or last_member_id
        if not mid:
            return "Hello 👋🏽 Say: **verify member 10**", "verify:help", last_member_id, None
        return _member_report_tables_only(schema, str(mid), members_truth), "verify:tables", str(mid), None

    mid2 = _extract_member_id(q)
    if mid2:
        return _member_report_tables_only(schema, str(mid2), members_truth), "member:tables", str(mid2), None

    if _lc(q) in RELATIONS:
        rel = _lc(q)
        df = _sb_select(schema, rel, limit=2000)
        return f"Hello 👋🏽 Preview of **{rel}**:", f"show:{rel}", last_member_id, _df_payload(f"Preview: {rel}", df)

    return "Hello 👋🏽 Try: **members**, **loans**, **finance kpis**, **tables**, **show contributions**, **describe loans**, **How are we doing?**, or type a member_id like **5**.", "db:guide", last_member_id, None


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "time": datetime.now(timezone.utc).isoformat(),
        "supabase_url_set": bool(_clean_env_value(SUPABASE_URL)),
        "supabase_anon_set": bool(_clean_env_value(SUPABASE_ANON_KEY)),
        "supabase_service_set": bool(_clean_env_value(SUPABASE_SERVICE_KEY)),
        "supabase_init_error": _SUPABASE_INIT_ERROR or None,
        "hf_token_set": bool(HF_TOKEN),
        "hf_models_parallel": HF_ALLOWED_MODELS,
        "internet": "ON" if _internet_enabled() else "OFF",
        "schema_default": DEFAULT_SCHEMA,
    }


@app.get("/relations")
def relations():
    return [{"relation": k, "type": RELATIONS[k].get("type")} for k in sorted(RELATIONS.keys())]


@app.get("/describe/{relation}")
def describe(relation: str, schema: str = DEFAULT_SCHEMA):
    _relation_guard(relation)
    df = _sb_select(schema, relation, limit=1)
    return {"relation": relation, "type": RELATIONS[relation]["type"], "columns": list(df.columns)}


@app.get("/preview/{relation}")
def preview(relation: str, schema: str = DEFAULT_SCHEMA, limit: int = 50):
    _relation_guard(relation)
    limit = max(1, min(int(limit), 2000))
    df = _sb_select(schema, relation, limit=limit)
    return _df_payload(f"Preview: {relation}", df, limit=limit)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = _clean(req.message)
    if not q:
        raise HTTPException(status_code=400, detail="message required")

    schema = (req.schema or DEFAULT_SCHEMA).strip() or DEFAULT_SCHEMA
    last_member_id = _clean(req.last_member_id or "") or None

    detected = _extract_member_id(q)
    if detected:
        last_member_id = detected

    brain = _macco_brain(q, last_member_id)

    if brain["use_db"] or brain["use_web"] or _is_db_command(q) or _wants_internet(q):
        reply, used_source, member_focus, df = _handle_db_commands(schema, q, last_member_id)
        return ChatResponse(
            reply=reply if reply == _intro_only() else _force_hello_prefix(reply),
            used_source=used_source,
            member_id_focus=member_focus,
            dataframe=df,
            meta={
                "schema": schema,
                "internet": "ON" if _internet_enabled() else "OFF",
                "hf_token_set": bool(HF_TOKEN),
                "brain": brain,
            },
        )

    if HF_TOKEN:
        messages = [{"role": "system", "content": _younchat_hf_system_prompt()}]
        for m in (req.history or [])[-10:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": q})

        ok, txt, mode, model_used, council_meta = _hf_call(HF_TOKEN, messages)
        used_source = f"hf:{mode}:{model_used}" if ok else f"hf:failed:{model_used}"

        if not ok:
            reply = f"Hello 👋🏽 HF is not reachable: {txt}"
        elif _looks_like_code_output(txt):
            reply = "Hello 👋🏽 I can’t output code. Use: **members**, **loans**, **finance kpis**, **tables**, **show <table>**, **describe <table>**, or type a **member_id**."
            used_source += ":blocked_code"
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
                "brain": brain,
                "model_council": council_meta,
            },
        )

    return ChatResponse(
        reply="Hello 👋🏽",
        used_source="local:fallback",
        member_id_focus=last_member_id,
        dataframe=None,
        meta={
            "schema": schema,
            "internet": "ON" if _internet_enabled() else "OFF",
            "hf_token_set": False,
            "brain": brain,
        },
        )
