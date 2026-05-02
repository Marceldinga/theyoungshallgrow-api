# =============================================================================
# PART 1/5
# File: main.py
# Advanced Transformer-Style Younchat API
# Paste Part 1 first, then paste Part 2 directly under it.
# =============================================================================

from __future__ import annotations

import json
import os
import re
import time
import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from supabase import create_client
except Exception as e:
    raise RuntimeError(
        "Missing dependency: supabase-py. Add `supabase` to requirements.txt"
    ) from e


# =============================================================================
# CONFIG
# =============================================================================

APP_NAME = "theyoungshallgrow-api (younchat advanced transformer)"
APP_VERSION = "3.0.0"

DEFAULT_SCHEMA = (os.getenv("SUPABASE_SCHEMA", "public").strip() or "public")

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_ROUTER_COMPLETIONS_URL = "https://router.huggingface.co/v1/completions"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

HF_ALLOWED_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

MAX_HISTORY_MESSAGES = 16
MAX_RESPONSE_TOKENS = 900
MAX_PREVIEW_ROWS = 2000
MAX_DB_ROWS = 200000

TRANSFORMER_CONFIDENCE_THRESHOLD = 0.62
DB_GROUNDING_REQUIRED_THRESHOLD = 0.70


# =============================================================================
# SUPABASE RELATION ALLOWLIST
# =============================================================================

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


# =============================================================================
# REQUIRED INTRO
# =============================================================================

def _intro_only() -> str:
    return "Hello 👋🏽 I’m younchat — your Njangi assistant."


# =============================================================================
# ENVIRONMENT
# =============================================================================

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _clean_env_value(v: Optional[str]) -> str:
    if not v:
        return ""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1].strip()
    return v


SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = _env("SUPABASE_SERVICE_KEY")

HF_TOKEN = _env("HF_TOKEN")
HF_FORCE_MODE = _env("HF_FORCE_MODE", "auto").lower()

TAVILY_API_KEY = _env("TAVILY_API_KEY")
INTERNET_MODE = _env("INTERNET_MODE", "off").lower()


def _internet_enabled() -> bool:
    return INTERNET_MODE != "off" and bool(TAVILY_API_KEY)


# =============================================================================
# SUPABASE CLIENT INITIALIZATION
# =============================================================================

_SUPABASE_INIT_ERROR = ""


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
            if _SUPABASE_INIT_ERROR:
                _SUPABASE_INIT_ERROR += f" | Service key error: {e}"
            else:
                _SUPABASE_INIT_ERROR = f"Service key error: {e}"
            sb_service = None

    return sb_anon, sb_service


SB_ANON, SB_SERVICE = _supabase_clients()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Advanced transformer-style Njangi financial intelligence API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    schema: Optional[str] = None
    last_member_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = None
    safe_mode: bool = True
    advanced_mode: bool = True


class ChatResponse(BaseModel):
    reply: str
    used_source: str
    member_id_focus: Optional[str] = None
    dataframe: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class IntentType(str, Enum):
    GREETING = "greeting"
    TABLES = "tables"
    DESCRIBE = "describe"
    PREVIEW = "preview"
    MEMBERS = "members"
    VERIFY_MEMBER = "verify_member"
    MEMBER_REPORT = "member_report"
    KPIS = "kpis"
    LOANS = "loans"
    CONTRIBUTIONS = "contributions"
    FOUNDATION = "foundation"
    PAYOUTS = "payouts"
    FINES = "fines"
    ATTENDANCE = "attendance"
    FINANCE_REVIEW = "finance_review"
    INTERNET = "internet"
    GENERAL_AI = "general_ai"
    UNKNOWN = "unknown"


@dataclass
class TransformerIntent:
    intent: IntentType
    confidence: float
    member_id: Optional[str] = None
    relation: Optional[str] = None
    requires_db: bool = False
    requires_web: bool = False
    reason: str = ""
    extracted: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerContext:
    schema: str
    message: str
    normalized: str
    history: List[Dict[str, str]]
    last_member_id: Optional[str]
    safe_mode: bool = True
    advanced_mode: bool = True


# =============================================================================
# BASIC TEXT HELPERS
# =============================================================================

def _clean(text: str) -> str:
    return (text or "").strip()


def _lc(text: str) -> str:
    return _clean(text).lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _force_hello_prefix(text: str) -> str:
    t = _clean(text)
    if not t:
        return "Hello 👋🏽"
    if not t.lower().startswith("hello"):
        return "Hello 👋🏽 " + t
    return t


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


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


def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_sum(df: pd.DataFrame, col: Optional[str]) -> float:
    if df is None or df.empty or not col or col not in df.columns:
        return 0.0
    return float(_to_num_series(df[col]).sum())


def _safe_mean(df: pd.DataFrame, col: Optional[str]) -> float:
    if df is None or df.empty or not col or col not in df.columns:
        return 0.0
    return float(_to_num_series(df[col]).mean())


def _safe_max(df: pd.DataFrame, col: Optional[str]) -> float:
    if df is None or df.empty or not col or col not in df.columns:
        return 0.0
    return float(_to_num_series(df[col]).max())


def _safe_min(df: pd.DataFrame, col: Optional[str]) -> float:
    if df is None or df.empty or not col or col not in df.columns:
        return 0.0
    return float(_to_num_series(df[col]).min())


def _db_proof_line(row_counts: Dict[str, int]) -> str:
    ts = _utc_now()
    if not row_counts:
        return f"DB Proof: no row counts • fetched_at={ts}"
    parts = [f"{k}={int(v)}" for k, v in row_counts.items()]
    return f"DB Proof: {', '.join(parts)} • fetched_at={ts}"


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def _relation_guard(rel: str) -> None:
    if rel not in RELATIONS:
        raise HTTPException(status_code=400, detail=f"Relation not allowed: {rel}")


def _get_supabase_client():
    return SB_SERVICE or SB_ANON


def _sb_select(
    schema: str,
    relation: str,
    cols: str = "*",
    limit: int = 2000,
    filters: Optional[List[Tuple[str, str, Any]]] = None,
    order: Optional[Tuple[str, bool]] = None,
) -> pd.DataFrame:
    _relation_guard(relation)

    sb = _get_supabase_client()
    if sb is None:
        return pd.DataFrame()

    limit = max(1, min(int(limit), MAX_DB_ROWS))

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
    sb = _get_supabase_client()
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


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

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


def _strip_web_prefix(q: str) -> str:
    return re.sub(
        r"^(web:|internet:|tavily:)\s*",
        "",
        (q or "").strip(),
        flags=re.IGNORECASE,
    ).strip()


# =============================================================================
# TRANSFORMER-STYLE SCORING HELPERS
# =============================================================================

def _keyword_score(text: str, keywords: List[str]) -> float:
    t = _lc(text)

    if not keywords:
        return 0.0

    score = 0.0

    for keyword in keywords:
        k = keyword.lower()
        if k in t:
            score += 1.35 if len(k.split()) > 1 else 1.0

    return min(1.0, score / max(1.0, len(keywords) * 0.55))


def _starts_with_any(text: str, prefixes: List[str]) -> bool:
    t = _lc(text)
    return any(t.startswith(p) for p in prefixes)


def _contains_any(text: str, words: List[str]) -> bool:
    t = _lc(text)
    return any(w.lower() in t for w in words)


def _normalize_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []

    cleaned: List[Dict[str, str]] = []

    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": str(content)})

    return cleaned


# =============================================================================
# ADVANCED TRANSFORMER ROUTER
# =============================================================================

class AdvancedTransformerRouter:
    def __init__(self):
        self.keywords: Dict[IntentType, List[str]] = {
            IntentType.GREETING: [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            ],
            IntentType.TABLES: [
                "tables",
                "relations",
                "views",
                "list tables",
                "list views",
            ],
            IntentType.DESCRIBE: [
                "describe",
                "columns",
                "cols",
                "schema",
                "structure",
            ],
            IntentType.PREVIEW: [
                "show",
                "preview",
                "open",
                "display",
            ],
            IntentType.MEMBERS: [
                "members",
                "list members",
                "show members",
                "all members",
                "member ids",
                "who are the members",
            ],
            IntentType.VERIFY_MEMBER: [
                "verify member",
                "verify",
                "check member",
                "member status",
            ],
            IntentType.KPIS: [
                "kpi",
                "kpis",
                "finance kpi",
                "dashboard kpi",
                "metrics",
            ],
            IntentType.LOANS: [
                "loan",
                "loans",
                "borrow",
                "repay",
                "repayment",
                "overdue",
                "dpd",
                "interest due",
                "principal",
                "unpaid interest",
            ],
            IntentType.CONTRIBUTIONS: [
                "contribution",
                "contributions",
                "member contribution",
                "total contribution",
            ],
            IntentType.FOUNDATION: [
                "foundation",
                "foundation contribution",
                "foundation contributions",
                "reserve",
                "reserves",
            ],
            IntentType.PAYOUTS: [
                "payout",
                "payouts",
                "beneficiary",
                "next beneficiary",
            ],
            IntentType.FINES: [
                "fine",
                "fines",
                "penalty",
                "penalties",
            ],
            IntentType.ATTENDANCE: [
                "attendance",
                "present",
                "absent",
                "meeting attendance",
            ],
            IntentType.FINANCE_REVIEW: [
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
            ],
            IntentType.INTERNET: [
                "web:",
                "internet:",
                "tavily:",
                "search online",
                "look up online",
            ],
        }

    def route(self, ctx: TransformerContext) -> TransformerIntent:
        text = ctx.normalized
        member_id = _extract_member_id(text) or ctx.last_member_id
        relation = _extract_relation_name(text)

        if _starts_with_any(text, ["web:", "internet:", "tavily:"]):
            return TransformerIntent(
                intent=IntentType.INTERNET,
                confidence=1.0,
                member_id=member_id,
                relation=relation,
                requires_web=True,
                reason="Explicit web prefix detected.",
            )

        if text in {"tables", "relations", "views", "list tables", "list views"}:
            return TransformerIntent(
                intent=IntentType.TABLES,
                confidence=1.0,
                member_id=member_id,
                relation=relation,
                requires_db=True,
                reason="Direct table list command.",
            )

        if _starts_with_any(text, ["describe ", "columns ", "cols ", "schema "]):
            return TransformerIntent(
                intent=IntentType.DESCRIBE,
                confidence=0.99,
                member_id=member_id,
                relation=relation,
                requires_db=True,
                reason="Describe command detected.",
            )

        if _starts_with_any(text, ["show ", "preview ", "open "]):
            return TransformerIntent(
                intent=IntentType.PREVIEW,
                confidence=0.98,
                member_id=member_id,
                relation=relation,
                requires_db=True,
                reason="Preview command detected.",
            )

        if text in RELATIONS:
            return TransformerIntent(
                intent=IntentType.PREVIEW,
                confidence=0.95,
                member_id=member_id,
                relation=text,
                requires_db=True,
                reason="Direct relation name detected.",
            )

        if text.isdigit():
            return TransformerIntent(
                intent=IntentType.MEMBER_REPORT,
                confidence=0.99,
                member_id=text,
                relation=None,
                requires_db=True,
                reason="User typed only a member ID.",
            )

        candidates: List[TransformerIntent] = []

        for intent_type, words in self.keywords.items():
            score = _keyword_score(text, words)

            if score <= 0:
                continue

            requires_web = intent_type == IntentType.INTERNET
            requires_db = intent_type not in {
                IntentType.GREETING,
                IntentType.GENERAL_AI,
                IntentType.INTERNET,
            }

            candidates.append(
                TransformerIntent(
                    intent=intent_type,
                    confidence=score,
                    member_id=member_id,
                    relation=relation,
                    requires_db=requires_db,
                    requires_web=requires_web,
                    reason=f"Keyword score={score:.2f}",
                )
            )

        finance_score = _keyword_score(
            text,
            [
                "contribution",
                "loan",
                "interest",
                "payout",
                "fine",
                "balance",
                "foundation",
                "liquidity",
                "risk",
                "member",
                "overdue",
                "repayment",
            ],
        )

        if finance_score >= 0.35:
            candidates.append(
                TransformerIntent(
                    intent=IntentType.FINANCE_REVIEW,
                    confidence=max(0.75, finance_score),
                    member_id=member_id,
                    relation=relation,
                    requires_db=True,
                    requires_web=False,
                    reason="Financial language requires DB grounding.",
                )
            )

        if member_id and not text.isdigit():
            candidates.append(
                TransformerIntent(
                    intent=IntentType.MEMBER_REPORT,
                    confidence=0.82,
                    member_id=member_id,
                    relation=relation,
                    requires_db=True,
                    requires_web=False,
                    reason="Member ID detected.",
                )
            )

        if not candidates:
            return TransformerIntent(
                intent=IntentType.GENERAL_AI,
                confidence=0.55,
                member_id=member_id,
                relation=relation,
                requires_db=False,
                requires_web=False,
                reason="No strong DB intent detected.",
            )

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        best = candidates[0]

        if best.confidence < TRANSFORMER_CONFIDENCE_THRESHOLD:
            return TransformerIntent(
                intent=IntentType.GENERAL_AI,
                confidence=best.confidence,
                member_id=member_id,
                relation=relation,
                requires_db=False,
                requires_web=False,
                reason="Low confidence fallback.",
            )

        return best


ROUTER = AdvancedTransformerRouter()


def _is_db_command_by_intent(intent: TransformerIntent) -> bool:
    return intent.requires_db or intent.intent in {
        IntentType.TABLES,
        IntentType.DESCRIBE,
        IntentType.PREVIEW,
        IntentType.MEMBERS,
        IntentType.VERIFY_MEMBER,
        IntentType.MEMBER_REPORT,
        IntentType.KPIS,
        IntentType.LOANS,
        IntentType.CONTRIBUTIONS,
        IntentType.FOUNDATION,
        IntentType.PAYOUTS,
        IntentType.FINES,
        IntentType.ATTENDANCE,
        IntentType.FINANCE_REVIEW,
    }


# =============================================================================
# END OF PART 1/5
# Paste Part 2 directly below this line.
# =============================================================================



# =============================================================================
# PART 2/5
# Member Truth Layer + Contribution/Foundation Intelligence
# Paste this directly under Part 1.
# =============================================================================


# =============================================================================
# MEMBERS TRUTH SOURCE
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

    if display_col and display_col in df.columns:
        display = (
            df[display_col]
            .astype(str)
            .replace(["None", "nan", "NaN", "NULL", "null"], "")
            .fillna("")
            .str.strip()
        )
    else:
        display = pd.Series([""] * len(df))

    if name_col and name_col in df.columns:
        name = (
            df[name_col]
            .astype(str)
            .replace(["None", "nan", "NaN", "NULL", "null"], "")
            .fillna("")
            .str.strip()
        )
    else:
        name = pd.Series([""] * len(df))

    out["member_name"] = display.where(display != "", name).fillna("").replace("", "(no name)")

    try:
        out["_id_num"] = pd.to_numeric(out["member_id"], errors="coerce")
        out = out.sort_values(["_id_num", "member_id"], ascending=True).drop(columns=["_id_num"])
    except Exception:
        pass

    return out


def _member_exists(members_truth: pd.DataFrame, member_id: str) -> bool:
    if members_truth is None or members_truth.empty:
        return False

    hit = members_truth[members_truth["member_id"].astype(str) == str(member_id)]
    return not hit.empty


def _member_name_from_truth(members_truth: pd.DataFrame, member_id: str) -> str:
    if members_truth is None or members_truth.empty:
        return "(unknown)"

    hit = members_truth[members_truth["member_id"].astype(str) == str(member_id)]

    if hit.empty:
        return "(unknown)"

    return str(hit.iloc[0]["member_name"])


def _members_list_reply(members_truth: pd.DataFrame) -> str:
    if members_truth is None or members_truth.empty:
        return "Hello 👋🏽 I couldn’t read members. Check Supabase RLS or keys."

    lines: List[str] = []
    lines.append("Hello 👋🏽 Here are all members from `members`:\n")

    for r in members_truth.itertuples(index=False):
        lines.append(f"- **{r.member_id}** • {r.member_name}")

    return "\n".join(lines)


# =============================================================================
# CONTRIBUTION HELPERS
# =============================================================================

def _contribution_amount_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "amount",
            "contribution_amount",
            "paid_amount",
            "total_amount",
            "value",
        ],
    )


def _contribution_date_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "created_at",
            "paid_at",
            "payment_date",
            "contribution_date",
            "session_date",
            "date",
        ],
    )


def _contribution_status_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "status",
            "payment_status",
            "state",
        ],
    )


def _load_contributions(schema: str, member_id: Optional[str] = None, limit: int = MAX_DB_ROWS) -> pd.DataFrame:
    filters = [("member_id", "eq", member_id)] if member_id else None
    return _sb_select(schema, "contributions", cols="*", limit=limit, filters=filters)


def _load_foundation_contributions(schema: str, member_id: Optional[str] = None, limit: int = MAX_DB_ROWS) -> pd.DataFrame:
    filters = [("member_id", "eq", member_id)] if member_id else None
    return _sb_select(schema, "foundation_contributions", cols="*", limit=limit, filters=filters)


def _load_member_contribution_totals(schema: str, member_id: Optional[str] = None) -> pd.DataFrame:
    if "member_contribution_totals" not in RELATIONS:
        return pd.DataFrame()

    filters = [("member_id", "eq", member_id)] if member_id else None
    return _sb_select(schema, "member_contribution_totals", cols="*", limit=MAX_DB_ROWS, filters=filters)


def _summarize_contribution_df(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    amount_col = _contribution_amount_col(df)
    status_col = _contribution_status_col(df)
    date_col = _contribution_date_col(df)

    total = _safe_sum(df, amount_col)
    avg = _safe_mean(df, amount_col)
    max_val = _safe_max(df, amount_col)
    min_val = _safe_min(df, amount_col)

    paid_rows = 0
    pending_rows = 0
    failed_rows = 0

    if status_col and status_col in df.columns:
        s = df[status_col].astype(str).str.lower().fillna("")
        paid_rows = int(s.isin(["paid", "complete", "completed", "confirmed", "success", "successful"]).sum())
        pending_rows = int(s.isin(["pending", "processing", "waiting"]).sum())
        failed_rows = int(s.isin(["failed", "cancelled", "canceled", "rejected"]).sum())

    latest_date = None
    earliest_date = None

    if date_col and date_col in df.columns and not df.empty:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if not dates.dropna().empty:
            latest_date = str(dates.max())
            earliest_date = str(dates.min())

    return {
        "label": label,
        "rows": int(len(df)),
        "amount_col": amount_col,
        "status_col": status_col,
        "date_col": date_col,
        "total": total,
        "average": avg,
        "max": max_val,
        "min": min_val,
        "paid_rows": paid_rows,
        "pending_rows": pending_rows,
        "failed_rows": failed_rows,
        "latest_date": latest_date,
        "earliest_date": earliest_date,
    }


def _build_contribution_report(
    schema: str,
    members_truth: pd.DataFrame,
    member_id: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:

    contributions = _load_contributions(schema, member_id=member_id)
    foundation = _load_foundation_contributions(schema, member_id=member_id)

    contribution_summary = _summarize_contribution_df(contributions, "contributions")
    foundation_summary = _summarize_contribution_df(foundation, "foundation_contributions")

    title_name = "All Members"
    if member_id:
        title_name = f"{_member_name_from_truth(members_truth, member_id)} (member_id={member_id})"

    lines: List[str] = []
    lines.append("Hello 👋🏽 Contribution Intelligence Report (DB-grounded)\n")
    lines.append("1️⃣ Scope")
    lines.append(f"- Focus: **{title_name}**")
    lines.append(f"- Schema: **{schema}**")

    lines.append("\n2️⃣ Regular Contributions")
    lines.append(f"- Rows: **{contribution_summary['rows']}**")
    lines.append(f"- Total: **{_fmt(contribution_summary['total'])}**")
    lines.append(f"- Average: **{_fmt(contribution_summary['average'])}**")
    lines.append(f"- Minimum: **{_fmt(contribution_summary['min'])}**")
    lines.append(f"- Maximum: **{_fmt(contribution_summary['max'])}**")
    lines.append(f"- Paid rows: **{contribution_summary['paid_rows']}**")
    lines.append(f"- Pending rows: **{contribution_summary['pending_rows']}**")
    lines.append(f"- Failed rows: **{contribution_summary['failed_rows']}**")

    if contribution_summary.get("earliest_date") or contribution_summary.get("latest_date"):
        lines.append(f"- Date range: **{contribution_summary.get('earliest_date') or '—'} → {contribution_summary.get('latest_date') or '—'}**")

    lines.append("\n3️⃣ Foundation Contributions")
    lines.append(f"- Rows: **{foundation_summary['rows']}**")
    lines.append(f"- Total: **{_fmt(foundation_summary['total'])}**")
    lines.append(f"- Average: **{_fmt(foundation_summary['average'])}**")
    lines.append(f"- Minimum: **{_fmt(foundation_summary['min'])}**")
    lines.append(f"- Maximum: **{_fmt(foundation_summary['max'])}**")
    lines.append(f"- Paid rows: **{foundation_summary['paid_rows']}**")
    lines.append(f"- Pending rows: **{foundation_summary['pending_rows']}**")
    lines.append(f"- Failed rows: **{foundation_summary['failed_rows']}**")

    if foundation_summary.get("earliest_date") or foundation_summary.get("latest_date"):
        lines.append(f"- Date range: **{foundation_summary.get('earliest_date') or '—'} → {foundation_summary.get('latest_date') or '—'}**")

    lines.append("\n4️⃣ Combined View")
    combined_total = _to_float(contribution_summary["total"]) + _to_float(foundation_summary["total"])
    combined_rows = int(contribution_summary["rows"]) + int(foundation_summary["rows"])
    lines.append(f"- Combined contribution rows: **{combined_rows}**")
    lines.append(f"- Combined contribution total: **{_fmt(combined_total)}**")

    lines.append("\n🧾 DB Proof")
    lines.append(
        f"- {_db_proof_line({'contributions': int(len(contributions)), 'foundation_contributions': int(len(foundation))})}"
    )

    payload_rows: List[Dict[str, Any]] = [
        contribution_summary,
        foundation_summary,
        {
            "label": "combined",
            "rows": combined_rows,
            "total": combined_total,
        },
    ]

    payload = _df_payload("Contribution Intelligence Summary", pd.DataFrame(payload_rows), limit=50)

    return "\n".join(lines), payload


# =============================================================================
# FOUNDATION RESERVE INTELLIGENCE
# =============================================================================

def _foundation_health_status(total_foundation: float, active_loan_exposure: float) -> Tuple[str, str]:
    if total_foundation <= 0:
        return "Unknown", "Foundation total is zero or unavailable."

    pressure = active_loan_exposure / total_foundation if total_foundation else 0.0

    if pressure >= 1.0:
        return "High Pressure", "Active loan exposure is equal to or greater than foundation reserves."

    if pressure >= 0.75:
        return "Elevated Pressure", "Active loan exposure is above 75% of foundation reserves."

    if pressure >= 0.50:
        return "Moderate Pressure", "Active loan exposure is above 50% of foundation reserves."

    return "Healthy", "Active loan exposure is below 50% of foundation reserves."


def _build_foundation_report(
    schema: str,
    members_truth: pd.DataFrame,
    member_id: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:

    foundation = _load_foundation_contributions(schema, member_id=member_id)
    loans = _sb_select(schema, "loans", cols="*", limit=MAX_DB_ROWS, filters=[("member_id", "eq", member_id)] if member_id else None)

    foundation_summary = _summarize_contribution_df(foundation, "foundation_contributions")

    active_loans = _active_loan_filter(loans) if "status" in loans.columns or not loans.empty else loans
    bal_col = _loan_balance_col(active_loans)
    active_exposure = _safe_sum(active_loans, bal_col)

    health, explanation = _foundation_health_status(_to_float(foundation_summary["total"]), active_exposure)

    focus = "All Members"
    if member_id:
        focus = f"{_member_name_from_truth(members_truth, member_id)} (member_id={member_id})"

    lines: List[str] = []
    lines.append("Hello 👋🏽 Foundation Reserve Intelligence (DB-grounded)\n")
    lines.append("1️⃣ Scope")
    lines.append(f"- Focus: **{focus}**")
    lines.append(f"- Schema: **{schema}**")

    lines.append("\n2️⃣ Foundation Position")
    lines.append(f"- Foundation rows: **{foundation_summary['rows']}**")
    lines.append(f"- Foundation total: **{_fmt(foundation_summary['total'])}**")
    lines.append(f"- Foundation average payment: **{_fmt(foundation_summary['average'])}**")

    lines.append("\n3️⃣ Exposure Check")
    lines.append(f"- Active loan exposure: **{_fmt(active_exposure)}**")
    lines.append(f"- Foundation health: **{health}**")
    lines.append(f"- Explanation: {explanation}")

    lines.append("\n🧾 DB Proof")
    lines.append(
        f"- {_db_proof_line({'foundation_contributions': int(len(foundation)), 'loans': int(len(loans))})}"
    )

    payload = pd.DataFrame(
        [
            {
                "focus": focus,
                "foundation_total": _to_float(foundation_summary["total"]),
                "foundation_rows": foundation_summary["rows"],
                "active_loan_exposure": active_exposure,
                "health": health,
            }
        ]
    )

    return "\n".join(lines), _df_payload("Foundation Intelligence Summary", payload)


# =============================================================================
# MEMBER COMBINED FINANCIAL TOTALS
# =============================================================================

def _compute_member_totals_from_tables(schema: str, member_id: str) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []

    contributions = _sb_select(
        schema,
        "contributions",
        cols="*",
        limit=MAX_DB_ROWS,
        filters=[("member_id", "eq", member_id)],
    )

    foundation = _sb_select(
        schema,
        "foundation_contributions",
        cols="*",
        limit=MAX_DB_ROWS,
        filters=[("member_id", "eq", member_id)],
    )

    fines = _sb_select(
        schema,
        "fines",
        cols="*",
        limit=MAX_DB_ROWS,
        filters=[("member_id", "eq", member_id)],
    )

    loans = _sb_select(
        schema,
        "loans",
        cols="*",
        limit=MAX_DB_ROWS,
        filters=[("member_id", "eq", member_id)],
    )

    interest_ledger = _sb_select(
        schema,
        "interest_ledger",
        cols="*",
        limit=MAX_DB_ROWS,
        filters=[("member_id", "eq", member_id)],
    )

    contrib_col = _pick_col(contributions, ["amount", "contribution_amount", "paid_amount"])
    foundation_col = _pick_col(foundation, ["amount", "foundation_amount", "paid_amount"])
    fines_col = _pick_col(fines, ["amount", "fine_amount", "penalty_amount"])
    interest_col = _pick_col(interest_ledger, ["amount", "interest_amount"])

    active = _active_loan_filter(loans)
    bal_col = _loan_balance_col(active)
    unpaid_col = _unpaid_interest_col(active)

    if contrib_col is None and not contributions.empty:
        notes.append("Missing contributions amount column.")

    if foundation_col is None and not foundation.empty:
        notes.append("Missing foundation amount column.")

    if fines_col is None and not fines.empty:
        notes.append("Missing fines amount column.")

    if bal_col is None and not active.empty:
        notes.append("Missing loans balance column.")

    if unpaid_col is None and not active.empty:
        notes.append("Missing unpaid interest column.")

    out = {
        "contributions_total": _safe_sum(contributions, contrib_col),
        "foundation_total": _safe_sum(foundation, foundation_col),
        "fines_total": _safe_sum(fines, fines_col),
        "active_loan_balance": _safe_sum(active, bal_col),
        "active_unpaid_interest": _safe_sum(active, unpaid_col),
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


def _member_risk_grade(active_bal: float, unpaid: float) -> str:
    if active_bal <= 0 and unpaid <= 0:
        return "A"

    if active_bal > 0 and unpaid <= 0:
        return "B"

    return "C"


def _member_report_tables_only(
    schema: str,
    member_id: str,
    members_truth: pd.DataFrame,
) -> str:
    if not _member_exists(members_truth, member_id):
        return (
            "Hello 👋🏽 I can’t confirm that member_id exists in `members`. "
            "Type **members** to verify IDs, then retry."
        )

    name = _member_name_from_truth(members_truth, member_id)
    totals, notes = _compute_member_totals_from_tables(schema, member_id)

    active_bal = _to_float(totals.get("active_loan_balance"))
    unpaid = _to_float(totals.get("active_unpaid_interest"))
    grade = _member_risk_grade(active_bal, unpaid)

    lines: List[str] = []
    lines.append("Hello 👋🏽 Member Financial Intelligence (DB-grounded)\n")
    lines.append("1️⃣ Current Situation")
    lines.append(f"- Member: **{name}** (member_id={member_id})")
    lines.append(f"- Contributions total: **{_fmt(totals.get('contributions_total'))}**")
    lines.append(f"- Foundation total: **{_fmt(totals.get('foundation_total'))}**")
    lines.append(f"- Fines total: **{_fmt(totals.get('fines_total'))}**")
    lines.append(f"- Active loan balance: **{_fmt(totals.get('active_loan_balance'))}**")
    lines.append(f"- Active unpaid interest: **{_fmt(totals.get('active_unpaid_interest'))}**")
    lines.append(f"- Interest ledger total: **{_fmt(totals.get('interest_total'))}**")

    lines.append("\n2️⃣ Risk Assessment")
    lines.append(f"- Member Risk Grade: **{grade}**")

    if grade == "A":
        lines.append("- Interpretation: Member has no active loan balance or unpaid interest.")
    elif grade == "B":
        lines.append("- Interpretation: Member has active loan exposure but no unpaid interest detected.")
    else:
        lines.append("- Interpretation: Member has active loan exposure and unpaid interest detected.")

    lines.append("\n🧾 DB Proof")
    lines.append(f"- {_db_proof_line(totals.get('_rows', {}))}")

    if notes:
        lines.append("\n🔒 Data Integrity Notes")
        for n in notes:
            lines.append(f"- {n}")

    return "\n".join(lines)


# =============================================================================
# CONTRIBUTION TREND SUPPORT
# =============================================================================

def _monthly_total_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "total", "rows"])

    amount_col = _contribution_amount_col(df)
    date_col = _contribution_date_col(df)

    if not amount_col or not date_col:
        return pd.DataFrame(columns=["month", "total", "rows"])

    temp = df.copy()
    temp["_date"] = pd.to_datetime(temp[date_col], errors="coerce")
    temp["_amount"] = pd.to_numeric(temp[amount_col], errors="coerce").fillna(0)
    temp = temp.dropna(subset=["_date"])

    if temp.empty:
        return pd.DataFrame(columns=["month", "total", "rows"])

    temp["month"] = temp["_date"].dt.to_period("M").astype(str)

    grouped = (
        temp.groupby("month")
        .agg(total=("_amount", "sum"), rows=("_amount", "count"))
        .reset_index()
        .sort_values("month")
    )

    return grouped


def _build_contribution_trend_report(
    schema: str,
    members_truth: pd.DataFrame,
    member_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:

    contributions = _load_contributions(schema, member_id=member_id)
    foundation = _load_foundation_contributions(schema, member_id=member_id)

    regular_monthly = _monthly_total_from_df(contributions)
    foundation_monthly = _monthly_total_from_df(foundation)

    focus = "All Members"
    if member_id:
        focus = f"{_member_name_from_truth(members_truth, member_id)} (member_id={member_id})"

    lines: List[str] = []
    lines.append("Hello 👋🏽 Contribution Trend Report (DB-grounded)\n")
    lines.append("1️⃣ Scope")
    lines.append(f"- Focus: **{focus}**")
    lines.append(f"- Schema: **{schema}**")

    lines.append("\n2️⃣ Trend Summary")
    lines.append(f"- Regular contribution months detected: **{len(regular_monthly)}**")
    lines.append(f"- Foundation contribution months detected: **{len(foundation_monthly)}**")

    if not regular_monthly.empty:
        latest = regular_monthly.iloc[-1]
        lines.append(f"- Latest regular contribution month: **{latest['month']}**")
        lines.append(f"- Latest regular contribution total: **{_fmt(latest['total'])}**")

    if not foundation_monthly.empty:
        latest_f = foundation_monthly.iloc[-1]
        lines.append(f"- Latest foundation month: **{latest_f['month']}**")
        lines.append(f"- Latest foundation total: **{_fmt(latest_f['total'])}**")

    lines.append("\n🧾 DB Proof")
    lines.append(
        f"- {_db_proof_line({'contributions': int(len(contributions)), 'foundation_contributions': int(len(foundation))})}"
    )

    combined_rows: List[Dict[str, Any]] = []

    for r in regular_monthly.to_dict(orient="records"):
        r["source"] = "contributions"
        combined_rows.append(r)

    for r in foundation_monthly.to_dict(orient="records"):
        r["source"] = "foundation_contributions"
        combined_rows.append(r)

    payload = _df_payload(
        "Contribution Monthly Trend",
        pd.DataFrame(combined_rows),
        limit=500,
    )

    return "\n".join(lines), payload


# =============================================================================
# END OF PART 2/5
# Paste Part 3 directly below this line.
# =============================================================================

# =============================================================================
# PART 3/5
# Loan Intelligence + Global Finance Metrics + Risk Engine
# Paste this directly under Part 2.
# =============================================================================


# =============================================================================
# LOAN HELPERS
# =============================================================================

def _loan_status_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(df, ["status", "loan_status", "state"])


def _loan_member_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(df, ["member_id", "user_id", "borrower_id"])


def _loan_amount_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "principal_current",
            "outstanding_principal",
            "principal_remaining",
            "principal",
            "amount",
            "loan_amount",
        ],
    )


def _loan_original_amount_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "principal",
            "original_principal",
            "loan_amount",
            "amount",
            "approved_amount",
            "disbursed_amount",
        ],
    )


def _loan_interest_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "unpaid_interest",
            "interest_unpaid",
            "interest_due",
            "interest_balance",
            "interest_amount",
        ],
    )


def _loan_date_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "created_at",
            "loan_date",
            "approved_at",
            "disbursed_at",
            "start_date",
            "date",
        ],
    )


def _loan_due_date_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "due_date",
            "next_due_date",
            "maturity_date",
            "repayment_due_date",
        ],
    )


def _loan_dpd_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(df, ["dpd", "days_past_due", "overdue_days"])


def _active_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans

    status_col = _loan_status_col(loans)

    if not status_col:
        return loans

    active_status = {
        "active",
        "open",
        "ongoing",
        "overdue",
        "late",
        "running",
        "disbursed",
        "approved",
    }

    s = loans[status_col].astype(str).str.lower().fillna("")
    return loans[s.isin(active_status)]


def _overdue_loan_filter(loans: pd.DataFrame) -> pd.DataFrame:
    if loans is None or loans.empty:
        return loans

    status_col = _loan_status_col(loans)

    if status_col:
        s = loans[status_col].astype(str).str.lower().fillna("")
        overdue_by_status = loans[s.isin({"overdue", "late", "default", "delinquent"})]
        if not overdue_by_status.empty:
            return overdue_by_status

    dpd_col = _loan_dpd_col(loans)

    if dpd_col:
        dpd = _to_num_series(loans[dpd_col])
        return loans[dpd > 0]

    due_col = _loan_due_date_col(loans)

    if due_col:
        temp = loans.copy()
        temp["_due"] = pd.to_datetime(temp[due_col], errors="coerce", utc=True)
        now = pd.Timestamp.utcnow()
        return temp[temp["_due"].notna() & (temp["_due"] < now)]

    return loans.iloc[0:0]


def _loan_balance_col(loans: pd.DataFrame) -> Optional[str]:
    return _loan_amount_col(loans)


def _unpaid_interest_col(loans: pd.DataFrame) -> Optional[str]:
    return _loan_interest_col(loans)


def _load_loans(schema: str, member_id: Optional[str] = None, limit: int = MAX_DB_ROWS) -> pd.DataFrame:
    filters = [("member_id", "eq", member_id)] if member_id else None

    if "v_loans_with_member" in RELATIONS:
        df = _sb_select(schema, "v_loans_with_member", cols="*", limit=limit, filters=filters)
        if not df.empty:
            return df

    return _sb_select(schema, "loans", cols="*", limit=limit, filters=filters)


def _load_loan_payments(schema: str, member_id: Optional[str] = None, limit: int = MAX_DB_ROWS) -> pd.DataFrame:
    filters = [("member_id", "eq", member_id)] if member_id else None

    if "v_loan_payments_with_member" in RELATIONS:
        df = _sb_select(schema, "v_loan_payments_with_member", cols="*", limit=limit, filters=filters)
        if not df.empty:
            return df

    return _sb_select(schema, "loan_payments", cols="*", limit=limit, filters=filters)


def _loan_payment_amount_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "amount",
            "payment_amount",
            "paid_amount",
            "repayment_amount",
            "total_paid",
        ],
    )


def _loan_payment_date_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(
        df,
        [
            "created_at",
            "payment_date",
            "paid_at",
            "repayment_date",
            "date",
        ],
    )


def _summarize_loans_df(loans: pd.DataFrame) -> Dict[str, Any]:
    amount_col = _loan_amount_col(loans)
    original_col = _loan_original_amount_col(loans)
    interest_col = _loan_interest_col(loans)
    status_col = _loan_status_col(loans)
    dpd_col = _loan_dpd_col(loans)
    date_col = _loan_date_col(loans)
    due_col = _loan_due_date_col(loans)

    active = _active_loan_filter(loans)
    overdue = _overdue_loan_filter(active)

    active_amount_col = _loan_amount_col(active)
    active_interest_col = _loan_interest_col(active)

    status_counts: Dict[str, int] = {}

    if status_col and status_col in loans.columns:
        s = loans[status_col].astype(str).fillna("(blank)")
        status_counts = {str(k): int(v) for k, v in s.value_counts().to_dict().items()}

    avg_dpd = 0.0
    max_dpd = 0.0

    if dpd_col and dpd_col in loans.columns:
        avg_dpd = _safe_mean(loans, dpd_col)
        max_dpd = _safe_max(loans, dpd_col)

    earliest_date = None
    latest_date = None

    if date_col and date_col in loans.columns and not loans.empty:
        dates = pd.to_datetime(loans[date_col], errors="coerce")
        dates = dates.dropna()
        if not dates.empty:
            earliest_date = str(dates.min())
            latest_date = str(dates.max())

    earliest_due = None
    latest_due = None

    if due_col and due_col in loans.columns and not loans.empty:
        dues = pd.to_datetime(loans[due_col], errors="coerce")
        dues = dues.dropna()
        if not dues.empty:
            earliest_due = str(dues.min())
            latest_due = str(dues.max())

    return {
        "rows": int(len(loans)),
        "active_rows": int(len(active)) if active is not None else 0,
        "overdue_rows": int(len(overdue)) if overdue is not None else 0,
        "amount_col": amount_col,
        "original_col": original_col,
        "interest_col": interest_col,
        "status_col": status_col,
        "dpd_col": dpd_col,
        "date_col": date_col,
        "due_col": due_col,
        "total_current_balance": _safe_sum(loans, amount_col),
        "total_original_amount": _safe_sum(loans, original_col),
        "total_unpaid_interest": _safe_sum(loans, interest_col),
        "active_loan_exposure": _safe_sum(active, active_amount_col),
        "active_unpaid_interest": _safe_sum(active, active_interest_col),
        "avg_dpd": avg_dpd,
        "max_dpd": max_dpd,
        "status_counts": status_counts,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "earliest_due": earliest_due,
        "latest_due": latest_due,
    }


def _summarize_loan_payments_df(payments: pd.DataFrame) -> Dict[str, Any]:
    amount_col = _loan_payment_amount_col(payments)
    date_col = _loan_payment_date_col(payments)

    earliest_date = None
    latest_date = None

    if date_col and date_col in payments.columns and not payments.empty:
        dates = pd.to_datetime(payments[date_col], errors="coerce")
        dates = dates.dropna()
        if not dates.empty:
            earliest_date = str(dates.min())
            latest_date = str(dates.max())

    return {
        "rows": int(len(payments)),
        "amount_col": amount_col,
        "date_col": date_col,
        "total_paid": _safe_sum(payments, amount_col),
        "average_payment": _safe_mean(payments, amount_col),
        "max_payment": _safe_max(payments, amount_col),
        "min_payment": _safe_min(payments, amount_col),
        "earliest_date": earliest_date,
        "latest_date": latest_date,
    }


def _build_loans_report(
    schema: str,
    members_truth: pd.DataFrame,
    member_id: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:

    loans = _load_loans(schema, member_id=member_id)
    payments = _load_loan_payments(schema, member_id=member_id)

    loan_summary = _summarize_loans_df(loans)
    payment_summary = _summarize_loan_payments_df(payments)

    focus = "All Members"

    if member_id:
        focus = f"{_member_name_from_truth(members_truth, member_id)} (member_id={member_id})"

    overdue_ratio = _ratio(
        float(loan_summary["overdue_rows"]),
        float(loan_summary["active_rows"]),
    )

    repayment_coverage = _ratio(
        _to_float(payment_summary["total_paid"]),
        _to_float(loan_summary["total_original_amount"]),
    )

    lines: List[str] = []
    lines.append("Hello 👋🏽 Loan Intelligence Report (DB-grounded)\n")
    lines.append("1️⃣ Scope")
    lines.append(f"- Focus: **{focus}**")
    lines.append(f"- Schema: **{schema}**")

    lines.append("\n2️⃣ Loan Position")
    lines.append(f"- Loan rows: **{loan_summary['rows']}**")
    lines.append(f"- Active loans: **{loan_summary['active_rows']}**")
    lines.append(f"- Overdue loans: **{loan_summary['overdue_rows']}**")
    lines.append(f"- Overdue ratio: **{_pct(overdue_ratio)}**")
    lines.append(f"- Current loan balance: **{_fmt(loan_summary['total_current_balance'])}**")
    lines.append(f"- Original loan amount: **{_fmt(loan_summary['total_original_amount'])}**")
    lines.append(f"- Active loan exposure: **{_fmt(loan_summary['active_loan_exposure'])}**")
    lines.append(f"- Unpaid interest: **{_fmt(loan_summary['total_unpaid_interest'])}**")
    lines.append(f"- Active unpaid interest: **{_fmt(loan_summary['active_unpaid_interest'])}**")

    if loan_summary.get("dpd_col"):
        lines.append(f"- Average DPD: **{loan_summary['avg_dpd']:.1f}**")
        lines.append(f"- Maximum DPD: **{loan_summary['max_dpd']:.1f}**")

    if loan_summary.get("earliest_date") or loan_summary.get("latest_date"):
        lines.append(
            f"- Loan date range: **{loan_summary.get('earliest_date') or '—'} → {loan_summary.get('latest_date') or '—'}**"
        )

    if loan_summary.get("earliest_due") or loan_summary.get("latest_due"):
        lines.append(
            f"- Due date range: **{loan_summary.get('earliest_due') or '—'} → {loan_summary.get('latest_due') or '—'}**"
        )

    lines.append("\n3️⃣ Repayment Position")
    lines.append(f"- Payment rows: **{payment_summary['rows']}**")
    lines.append(f"- Total paid: **{_fmt(payment_summary['total_paid'])}**")
    lines.append(f"- Average payment: **{_fmt(payment_summary['average_payment'])}**")
    lines.append(f"- Repayment coverage: **{_pct(repayment_coverage)}**")

    if payment_summary.get("earliest_date") or payment_summary.get("latest_date"):
        lines.append(
            f"- Payment date range: **{payment_summary.get('earliest_date') or '—'} → {payment_summary.get('latest_date') or '—'}**"
        )

    risk_label, signals = _loan_risk_classification(loan_summary, payment_summary)

    lines.append("\n4️⃣ Loan Risk Assessment")
    lines.append(f"- Loan risk classification: **{risk_label}**")

    if signals:
        lines.append("- Signals:")
        for s in signals:
            lines.append(f"  - {s}")
    else:
        lines.append("- Signals: **None detected**")

    lines.append("\n🧾 DB Proof")
    lines.append(
        f"- {_db_proof_line({'loans': int(len(loans)), 'loan_payments': int(len(payments))})}"
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "loan_rows", "value": loan_summary["rows"]},
            {"metric": "active_loans", "value": loan_summary["active_rows"]},
            {"metric": "overdue_loans", "value": loan_summary["overdue_rows"]},
            {"metric": "current_loan_balance", "value": loan_summary["total_current_balance"]},
            {"metric": "original_loan_amount", "value": loan_summary["total_original_amount"]},
            {"metric": "active_loan_exposure", "value": loan_summary["active_loan_exposure"]},
            {"metric": "unpaid_interest", "value": loan_summary["total_unpaid_interest"]},
            {"metric": "payment_rows", "value": payment_summary["rows"]},
            {"metric": "total_paid", "value": payment_summary["total_paid"]},
        ]
    )

    return "\n".join(lines), _df_payload("Loan Intelligence Summary", summary_df)


def _loan_risk_classification(
    loan_summary: Dict[str, Any],
    payment_summary: Dict[str, Any],
) -> Tuple[str, List[str]]:

    signals: List[str] = []
    score = 0

    active_rows = int(loan_summary.get("active_rows", 0) or 0)
    overdue_rows = int(loan_summary.get("overdue_rows", 0) or 0)

    overdue_ratio = _ratio(float(overdue_rows), float(active_rows)) if active_rows else 0.0

    unpaid_interest = _to_float(loan_summary.get("active_unpaid_interest"))
    exposure = _to_float(loan_summary.get("active_loan_exposure"))
    paid = _to_float(payment_summary.get("total_paid"))
    original = _to_float(loan_summary.get("total_original_amount"))
    coverage = _ratio(paid, original)

    if overdue_ratio is not None and overdue_ratio >= 0.30:
        score += 3
        signals.append("Overdue ratio is 30% or higher.")
    elif overdue_ratio is not None and overdue_ratio >= 0.10:
        score += 2
        signals.append("Overdue ratio is 10% or higher.")

    if unpaid_interest > 0:
        score += 1
        signals.append("Unpaid interest exists on active loans.")

    if exposure > 0 and paid <= 0:
        score += 1
        signals.append("Active exposure exists with no detected repayment coverage.")

    if coverage is not None and coverage < 0.25 and original > 0:
        score += 1
        signals.append("Repayment coverage is below 25% of original loan amount.")

    if score >= 5:
        return "High", signals
    if score >= 3:
        return "Elevated", signals
    if score >= 1:
        return "Moderate", signals

    return "Low", signals


# =============================================================================
# GLOBAL FINANCE INTELLIGENCE
# =============================================================================

def _snapshot_to_metrics(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None

    if (
        isinstance(snapshot.get("totals"), dict)
        or isinstance(snapshot.get("counts"), dict)
        or isinstance(snapshot.get("ratios"), dict)
    ):
        totals = snapshot.get("totals") or {}
        counts = snapshot.get("counts") or {}
        ratios = snapshot.get("ratios") or {}

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

    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}

    return {
        "notes": [],
        "row_counts": {k: int(v) for k, v in counts.items() if v is not None},
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


def _collect_global_finance(schema: str) -> Dict[str, Any]:
    snapshot = _rpc_finance_snapshot(schema)

    if snapshot:
        return {
            "ok": True,
            "notes": [],
            "snapshot": snapshot,
            "df": {},
        }

    ctx: Dict[str, Any] = {
        "ok": True,
        "notes": ["Snapshot unavailable → fallback compute."],
        "snapshot": {},
        "df": {},
    }

    ctx["df"]["contributions"] = _sb_select(schema, "contributions", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["foundation_contributions"] = _sb_select(schema, "foundation_contributions", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["loans"] = _sb_select(schema, "loans", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["loan_payments"] = _sb_select(schema, "loan_payments", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["interest_ledger"] = _sb_select(schema, "interest_ledger", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["fines"] = _sb_select(schema, "fines", cols="*", limit=MAX_DB_ROWS)
    ctx["df"]["payouts"] = _sb_select(schema, "payouts", cols="*", limit=MAX_DB_ROWS)

    return ctx


def _compute_global_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = ctx.get("snapshot") or {}
    snap_metrics = _snapshot_to_metrics(snapshot) if isinstance(snapshot, dict) else None

    if snap_metrics is not None:
        return snap_metrics

    dfc = (ctx.get("df") or {}).get("contributions", pd.DataFrame())
    dff = (ctx.get("df") or {}).get("foundation_contributions", pd.DataFrame())
    dfl = (ctx.get("df") or {}).get("loans", pd.DataFrame())
    dfp = (ctx.get("df") or {}).get("loan_payments", pd.DataFrame())
    dfi = (ctx.get("df") or {}).get("interest_ledger", pd.DataFrame())
    dffines = (ctx.get("df") or {}).get("fines", pd.DataFrame())
    dfpayouts = (ctx.get("df") or {}).get("payouts", pd.DataFrame())

    contrib_col = _contribution_amount_col(dfc)
    foundation_col = _contribution_amount_col(dff)
    fines_col = _pick_col(dffines, ["amount", "fine_amount", "penalty_amount"])
    payout_col = _pick_col(dfpayouts, ["amount", "payout_amount", "paid_amount"])

    active_loans = _active_loan_filter(dfl)
    overdue_loans = _overdue_loan_filter(active_loans)

    bal_col = _loan_amount_col(active_loans)
    unpaid_col = _loan_interest_col(active_loans)
    interest_col = _pick_col(dfi, ["amount", "interest_amount"])
    payment_col = _loan_payment_amount_col(dfp)

    total_contributions = _safe_sum(dfc, contrib_col)
    foundation_total = _safe_sum(dff, foundation_col)
    total_fines = _safe_sum(dffines, fines_col)
    total_payouts = _safe_sum(dfpayouts, payout_col)

    active_loan_exposure = _safe_sum(active_loans, bal_col)
    unpaid_interest = _safe_sum(active_loans, unpaid_col)
    interest_total = _safe_sum(dfi, interest_col)
    loan_payment_total = _safe_sum(dfp, payment_col)

    active_count = int(len(active_loans)) if active_loans is not None else 0
    overdue_count = int(len(overdue_loans)) if overdue_loans is not None else 0

    overdue_ratio = overdue_count / active_count if active_count > 0 else 0.0
    liquidity_pressure = _ratio(active_loan_exposure, total_contributions)
    foundation_pressure = _ratio(active_loan_exposure, foundation_total)
    payout_to_contribution_ratio = _ratio(total_payouts, total_contributions)

    available_like_funds = total_contributions + foundation_total + total_fines + interest_total + loan_payment_total - total_payouts
    net_position_after_loans = available_like_funds - active_loan_exposure

    return {
        "notes": ctx.get("notes", []),
        "row_counts": {
            "contributions": int(len(dfc)),
            "foundation_contributions": int(len(dff)),
            "loans": int(len(dfl)),
            "loan_payments": int(len(dfp)),
            "interest_ledger": int(len(dfi)),
            "fines": int(len(dffines)),
            "payouts": int(len(dfpayouts)),
        },
        "total_contributions": total_contributions,
        "foundation_total": foundation_total,
        "total_fines": total_fines,
        "total_payouts": total_payouts,
        "loan_payment_total": loan_payment_total,
        "active_loan_exposure": active_loan_exposure,
        "active_loan_count": active_count,
        "overdue_loan_count": overdue_count,
        "overdue_ratio": overdue_ratio,
        "unpaid_interest": unpaid_interest,
        "interest_total": interest_total,
        "liquidity_pressure_ratio": liquidity_pressure,
        "foundation_pressure_ratio": foundation_pressure,
        "payout_to_contribution_ratio": payout_to_contribution_ratio,
        "available_like_funds": available_like_funds,
        "net_position_after_loans": net_position_after_loans,
    }


def _risk_classification(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    signals: List[str] = []
    score = 0

    lpr = metrics.get("liquidity_pressure_ratio")
    fpr = metrics.get("foundation_pressure_ratio")
    overdue_ratio = metrics.get("overdue_ratio")
    unpaid_interest = metrics.get("unpaid_interest")
    net_position = metrics.get("net_position_after_loans")

    if lpr is not None and lpr > 0.75:
        score += 2
        signals.append("Liquidity pressure is above 75% of total contributions.")
    elif lpr is not None and lpr > 0.50:
        score += 1
        signals.append("Liquidity pressure is above 50% of total contributions.")

    if fpr is not None and fpr > 1.0:
        score += 2
        signals.append("Active loan exposure is greater than foundation reserves.")
    elif fpr is not None and fpr > 0.75:
        score += 1
        signals.append("Active loan exposure is above 75% of foundation reserves.")

    if overdue_ratio is not None and overdue_ratio > 0.30:
        score += 2
        signals.append("Overdue ratio is above 30% of active loans.")
    elif overdue_ratio is not None and overdue_ratio > 0.10:
        score += 1
        signals.append("Overdue ratio is above 10% of active loans.")

    if unpaid_interest is not None and unpaid_interest > 0:
        score += 1
        signals.append("Unpaid interest exists on active loans.")

    if net_position is not None and net_position < 0:
        score += 2
        signals.append("Net position after active loan exposure is negative.")

    if score >= 7:
        return "High", signals

    if score >= 4:
        return "Elevated", signals

    if score >= 1:
        return "Moderate", signals

    return "Low", signals


def _health_score(metrics: Dict[str, Any]) -> int:
    score = 100

    lpr = metrics.get("liquidity_pressure_ratio")
    fpr = metrics.get("foundation_pressure_ratio")
    overdue_ratio = metrics.get("overdue_ratio")
    unpaid_interest = _to_float(metrics.get("unpaid_interest"))
    net_position = _to_float(metrics.get("net_position_after_loans"))

    if lpr is not None:
        if lpr > 0.75:
            score -= 25
        elif lpr > 0.50:
            score -= 12

    if fpr is not None:
        if fpr > 1.0:
            score -= 20
        elif fpr > 0.75:
            score -= 10

    if overdue_ratio is not None:
        if overdue_ratio > 0.30:
            score -= 25
        elif overdue_ratio > 0.10:
            score -= 12

    if unpaid_interest > 0:
        score -= 8

    if net_position < 0:
        score -= 15

    return max(0, min(100, score))


def _health_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _build_control_tower_report(metrics: Dict[str, Any]) -> str:
    risk_label, signals = _risk_classification(metrics)
    score = _health_score(metrics)
    grade = _health_grade(score)

    lines: List[str] = []
    lines.append("Hello 👋🏽 Njangi Financial Intelligence Review (DB-grounded)\n")

    lines.append("1️⃣ Financial Position")
    lines.append(f"- Health score: **{score}/100**")
    lines.append(f"- Health grade: **{grade}**")
    lines.append(f"- Risk classification: **{risk_label}**")
    lines.append(f"- Total contributions: **{_fmt(metrics.get('total_contributions'))}**")
    lines.append(f"- Foundation reserves: **{_fmt(metrics.get('foundation_total'))}**")
    lines.append(f"- Total fines: **{_fmt(metrics.get('total_fines'))}**")
    lines.append(f"- Interest ledger total: **{_fmt(metrics.get('interest_total'))}**")
    lines.append(f"- Loan payment total: **{_fmt(metrics.get('loan_payment_total'))}**")
    lines.append(f"- Total payouts: **{_fmt(metrics.get('total_payouts'))}**")

    lines.append("\n2️⃣ Loan and Liquidity Position")
    lines.append(f"- Active loan exposure: **{_fmt(metrics.get('active_loan_exposure'))}**")
    lines.append(f"- Active loans: **{int(metrics.get('active_loan_count', 0) or 0)}**")
    lines.append(f"- Overdue loans: **{int(metrics.get('overdue_loan_count', 0) or 0)}**")
    lines.append(f"- Overdue ratio: **{_pct(metrics.get('overdue_ratio'))}**")
    lines.append(f"- Unpaid interest: **{_fmt(metrics.get('unpaid_interest'))}**")
    lines.append(f"- Liquidity pressure ratio: **{_pct(metrics.get('liquidity_pressure_ratio'))}**")
    lines.append(f"- Foundation pressure ratio: **{_pct(metrics.get('foundation_pressure_ratio'))}**")

    lines.append("\n3️⃣ Net Position")
    lines.append(f"- Available-like funds: **{_fmt(metrics.get('available_like_funds'))}**")
    lines.append(f"- Net position after active loan exposure: **{_fmt(metrics.get('net_position_after_loans'))}**")
    lines.append(f"- Payout-to-contribution ratio: **{_pct(metrics.get('payout_to_contribution_ratio'))}**")

    lines.append("\n4️⃣ Early Warning Signals")

    if signals:
        for s in signals:
            lines.append(f"- {s}")
    else:
        lines.append("- None detected.")

    lines.append("\n5️⃣ Recommendation")
    if risk_label in {"High", "Elevated"}:
        lines.append("- Tighten loan approval, monitor overdue accounts, and increase repayment follow-up.")
        lines.append("- Review foundation reserve strength before approving additional large loans.")
    elif risk_label == "Moderate":
        lines.append("- Continue monitoring loan exposure and unpaid interest before the next payout cycle.")
    else:
        lines.append("- Current indicators look stable based on available database records.")

    lines.append("\n🧾 DB Proof")
    lines.append(f"- {_db_proof_line(metrics.get('row_counts') or {})}")

    notes = metrics.get("notes") or []

    if notes:
        lines.append("\n🔒 Data Integrity Notes")
        for n in notes:
            lines.append(f"- {n}")

    return "\n".join(lines)


def _finance_metrics_dataframe(metrics: Dict[str, Any]) -> Dict[str, Any]:
    rows = []

    for key, value in metrics.items():
        if key in {"notes", "row_counts"}:
            continue
        rows.append({"metric": key, "value": value})

    return _df_payload("Finance Metrics", pd.DataFrame(rows), limit=200)


# =============================================================================
# END OF PART 3/5
# Paste Part 4 directly below this line.
# =============================================================================






