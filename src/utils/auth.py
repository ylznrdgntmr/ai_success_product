"""
Local Authentication Module
Provides email + password sign-up / sign-in for the Streamlit app.

Design notes (important for a "serious" / production-leaning deployment):
- Passwords are NEVER stored in plain text. Each password is hashed with
  PBKDF2-HMAC-SHA256 (200,000 iterations) and a unique random 16-byte salt
  per user, using only Python's standard library (`hashlib`) — no extra
  dependency required.
- Users are persisted in a local JSON file (`.local_users.json` by default,
  already covered by a .gitignore rule so it is never committed to git).
  This is appropriate for a single-server / internal-tool deployment. If
  this app is ever deployed multi-instance behind a load balancer, swap
  `JSONUserStore` for a real database-backed store — the interface below
  (`get_user`, `save_user`) is intentionally small so that's a drop-in change.
- This module does not implement password-reset e-mails (that requires an
  outbound e-mail/SMTP service to be configured) — a "forgot password"
  flow can be layered on top of `JSONUserStore` later.
"""

from typing import Optional, Dict, Any
import os
import re
import json
import hashlib
import secrets
import time
import streamlit as st

from src.utils.i18n import t

USER_STORE_PATH = os.getenv("APP_USER_STORE_PATH", ".local_users.json")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

PBKDF2_ITERATIONS = 200_000


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class JSONUserStore:
    """Minimal local JSON-backed user store: {email: {name, salt, hash, created_at}}."""

    def __init__(self, path: str = USER_STORE_PATH):
        self.path = path

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        return self._load().get(email.strip().lower())

    def save_user(self, email: str, record: Dict[str, Any]) -> None:
        data = self._load()
        data[email.strip().lower()] = record
        self._save(data)


_store = JSONUserStore()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return {"salt": salt.hex(), "hash": digest.hex()}


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return secrets.compare_digest(digest.hex(), hash_hex)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_user(name: str, email: str, password: str) -> Optional[str]:
    """
    Attempts to register a new user.
    Returns None on success, or an i18n-translated error message on failure.
    """
    email = email.strip().lower()
    if not name or not email or not password:
        return t("auth_required_fields")
    if not _EMAIL_RE.match(email):
        return t("auth_register_invalid_email")
    if len(password) < 6:
        return t("auth_register_weak")
    if _store.get_user(email):
        return t("auth_register_exists")

    hashed = _hash_password(password)
    _store.save_user(email, {
        "name": name.strip(),
        "salt": hashed["salt"],
        "hash": hashed["hash"],
        "created_at": time.time(),
    })
    return None


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Returns the user record (without hash/salt) on success, None on failure.
    Includes any saved UI preferences (theme/accent/language) so the caller
    can restore them right after login — see get_user_preferences below."""
    email = email.strip().lower()
    record = _store.get_user(email)
    if not record:
        return None
    if _verify_password(password, record["salt"], record["hash"]):
        return {
            "email": email,
            "name": record.get("name", email),
            "preferences": record.get("preferences", {}),
        }
    return None


# ---------------------------------------------------------------------------
# Per-user UI preferences (theme, accent color palette, language) — small,
# non-sensitive settings saved alongside the account so a personalized look
# follows the user back the next time they log in, on any device.
# ---------------------------------------------------------------------------
def get_user_preferences(email: str) -> Dict[str, Any]:
    record = _store.get_user(email)
    if not record:
        return {}
    return record.get("preferences", {})


def save_user_preferences(email: str, preferences: Dict[str, Any]) -> None:
    """Merges `preferences` into the user's saved preferences (doesn't wipe
    keys not present in this call) and persists them."""
    email = email.strip().lower()
    record = _store.get_user(email)
    if not record:
        return
    current = record.get("preferences", {})
    current.update(preferences)
    record["preferences"] = current
    _store.save_user(email, record)


def is_authenticated() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("auth_user")


def logout() -> None:
    st.session_state.pop("auth_user", None)


def render_auth_gate() -> bool:
    """
    Renders the login / register screen when nobody is signed in.
    Returns True if the caller may proceed to render the main app
    (i.e. a user is authenticated), False if it just rendered the gate
    and the caller should stop.
    """
    if is_authenticated():
        return True

    st.markdown(f"## {t('app_title')}")
    st.caption(t("auth_app_tagline"))

    login_tab, register_tab = st.tabs([t("auth_login_tab"), t("auth_register_tab")])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(t("auth_email_label"), key="login_email")
            password = st.text_input(t("auth_password_label"), type="password", key="login_password")
            submitted = st.form_submit_button(t("auth_login_btn"), type="primary", width="stretch")
        if submitted:
            if not email or not password:
                st.error(t("auth_required_fields"))
            else:
                user = authenticate_user(email, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.success(t("auth_login_success"))
                    st.rerun()
                else:
                    st.error(t("auth_login_fail"))
        st.caption(t("auth_demo_hint", tab=t("auth_register_tab")))

    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            name = st.text_input(t("auth_name_label"), key="register_name")
            email_r = st.text_input(t("auth_email_label"), key="register_email")
            pw1 = st.text_input(t("auth_password_label"), type="password", key="register_password")
            pw2 = st.text_input(t("auth_password_confirm_label"), type="password", key="register_password_confirm")
            submitted_r = st.form_submit_button(t("auth_register_btn"), type="primary", width="stretch")
        if submitted_r:
            if not name or not email_r or not pw1 or not pw2:
                st.error(t("auth_required_fields"))
            elif pw1 != pw2:
                st.error(t("auth_register_mismatch"))
            else:
                error = register_user(name, email_r, pw1)
                if error:
                    st.error(error)
                else:
                    st.success(t("auth_register_success"))

    return False
