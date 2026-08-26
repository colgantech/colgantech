"""Stateless client-portal starter: GitHub login for humans, bearer keys for
machines, scope-guarded RPC endpoints, one Mithril screen.

All configuration is environment variables; there is no database and no disk
state. Sessions are signed cookies, so restarts and multiple replicas need
nothing shared except SESSION_SECRET.

    SESSION_SECRET        secret for signing session cookies (auto-generated
                          if unset, which logs everyone out on restart)
    BASE_URL              public URL of this app, default http://localhost:8000
    GITHUB_CLIENT_ID      from the GitHub OAuth app whose callback URL is
    GITHUB_CLIENT_SECRET  {BASE_URL}/auth/callback
    USERS                 who may log in and what they may call:
                          "dcolgan = * ; frank = contact.read crm.write"
    API_KEYS              machine principals (schedulers, agents, connectors):
                          "sk-3kJx9q = nightly-sync : crm.write ; sk-8mPw2v = claude : contact.read"

A caller is resolved to a principal one of two ways: a session cookie set by
the GitHub callback, or an "Authorization: Bearer <key>" header matched
against API_KEYS. Every RPC endpoint declares the scopes it requires; "*"
grants everything.
"""

import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count

import click
import httpx
import uvicorn
from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import SecurityScopes
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")


def parse_users(raw: str) -> dict[str, set[str]]:
    """USERS entries are "login = scope scope"; logins are matched
    case-insensitively because GitHub logins are."""
    users: dict[str, set[str]] = {}
    for entry in raw.split(";"):
        if "=" not in entry:
            continue
        login, _, scopes = entry.partition("=")
        users[login.strip().lower()] = set(scopes.split())
    return users


def parse_api_keys(raw: str) -> dict[str, tuple[str, set[str]]]:
    """API_KEYS entries are "key = label : scope scope"."""
    keys: dict[str, tuple[str, set[str]]] = {}
    for entry in raw.split(";"):
        if "=" not in entry:
            continue
        key, _, rest = entry.partition("=")
        label, _, scopes = rest.partition(":")
        keys[key.strip()] = (label.strip(), set(scopes.split()))
    return keys


USERS = parse_users(os.environ.get("USERS", ""))
API_KEYS = parse_api_keys(os.environ.get("API_KEYS", ""))

# --------------------------------------------------------------------------
# Principals and scope enforcement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    label: str
    kind: str  # "human" | "machine"
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


class RpcError(Exception):
    """Failure that should reach the screen as the standard envelope."""

    def __init__(self, message: str, status: int = 400, invalid: dict | None = None):
        self.message = message
        self.status = status
        self.invalid = invalid


def resolve_principal(request: Request) -> Principal | None:
    login = request.session.get("login", "")
    if login and (scopes := USERS.get(login)) is not None:
        return Principal(login, "human", frozenset(scopes))
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and (
        found := API_KEYS.get(auth.removeprefix("Bearer ").strip())
    ):
        label, scopes = found
        return Principal(label, "machine", frozenset(scopes))
    return None


def current(security_scopes: SecurityScopes, request: Request) -> Principal:
    principal = resolve_principal(request)
    if principal is None:
        raise RpcError("Sign in required.", status=401)
    missing = [s for s in security_scopes.scopes if not principal.allows(s)]
    if missing:
        raise RpcError(f"Missing permission: {', '.join(missing)}.", status=403)
    return principal


# --------------------------------------------------------------------------
# App and the RPC envelope
# --------------------------------------------------------------------------

app = FastAPI(title="Client Portal Starter")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET or secrets.token_urlsafe(32),
    same_site="lax",
    https_only=BASE_URL.startswith("https"),
)


@app.exception_handler(RpcError)
async def rpc_error_handler(request: Request, exc: RpcError) -> JSONResponse:
    body: dict = {"ok": False, "message": exc.message}
    if exc.invalid:
        body["invalid"] = exc.invalid
    return JSONResponse(body, status_code=exc.status)


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Translate pydantic's validation report into the screen contract:
    {"invalid": {"field": ["message"], "non_field_errors": [...]}}."""
    invalid: dict[str, list[str]] = {}
    for error in exc.errors():
        loc = error["loc"]
        field = str(loc[1]) if len(loc) > 1 and loc[0] == "body" else "non_field_errors"
        invalid.setdefault(field, []).append(error["msg"])
    return JSONResponse(
        {"ok": False, "message": "Please fix the errors below.", "invalid": invalid},
        status_code=400,
    )


def ok(data: object = None) -> dict:
    return {"ok": True, "data": data}


# --------------------------------------------------------------------------
# RPC: auth
# --------------------------------------------------------------------------


@app.post("/rpc/auth.me")
async def auth_me(request: Request) -> dict:
    principal = resolve_principal(request)
    if principal is None:
        return ok(None)
    return ok(
        {
            "label": principal.label,
            "kind": principal.kind,
            "scopes": sorted(principal.scopes),
        }
    )


@app.post("/rpc/auth.logout")
async def auth_logout(request: Request) -> dict:
    request.session.clear()
    return ok()


# --------------------------------------------------------------------------
# RPC: demo resource
#
# A stand-in for the real work. In a client app these handlers would call the
# client's systems of record (Less Annoying CRM, Trello, Eventbrite, ...) via
# httpx; nothing about the auth or envelope changes when they do. The
# in-memory list exists only so the screen has something to show.
# --------------------------------------------------------------------------

_ids = count(4)
CONTACTS: list[dict] = [
    {
        "id": 1,
        "name": "Frank",
        "email": "frank@example.com",
        "created": "2026-08-01T09:00:00Z",
    },
    {
        "id": 2,
        "name": "Dana Innovation",
        "email": "dana@example.com",
        "created": "2026-08-10T14:30:00Z",
    },
    {
        "id": 3,
        "name": "Muncie Makers",
        "email": "hello@example.com",
        "created": "2026-08-20T11:15:00Z",
    },
]


class ContactList(BaseModel):
    q: str = ""


class ContactSave(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.post("/rpc/contact.list")
async def contact_list(
    body: ContactList, principal: Principal = Security(current, scopes=["contact.read"])
) -> dict:
    q = body.q.strip().lower()
    rows = [c for c in CONTACTS if q in c["name"].lower() or q in c["email"].lower()]
    return ok(rows)


@app.post("/rpc/contact.save")
async def contact_save(
    body: ContactSave, principal: Principal = Security(current, scopes=["crm.write"])
) -> dict:
    contact = {
        "id": next(_ids),
        "name": body.name,
        "email": body.email,
        "created": datetime.now(UTC).isoformat(),
    }
    CONTACTS.append(contact)
    return ok(contact)


# --------------------------------------------------------------------------
# GitHub login (humans only; machines never touch these routes)
# --------------------------------------------------------------------------


@app.get("/login")
async def login(request: Request) -> Response:
    if not GITHUB_CLIENT_ID:
        return HTMLResponse(
            "<p>GitHub login is not configured: set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.</p>",
            status_code=503,
        )
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    params = httpx.QueryParams(
        client_id=GITHUB_CLIENT_ID,
        redirect_uri=f"{BASE_URL}/auth/callback",
        state=state,
    )
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = "", state: str = "") -> Response:
    if not state or state != request.session.pop("oauth_state", None):
        return HTMLResponse(
            "<p>Login expired or was tampered with. <a href='/login'>Try again.</a></p>",
            status_code=400,
        )
    async with httpx.AsyncClient() as http:
        token_response = await http.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": f"{BASE_URL}/auth/callback",
            },
        )
        access_token = token_response.json().get("access_token", "")
        if not access_token:
            return HTMLResponse(
                "<p>GitHub did not accept the login. <a href='/login'>Try again.</a></p>",
                status_code=400,
            )
        user_response = await http.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    login_name = user_response.json().get("login", "").lower()
    if login_name not in USERS:
        return HTMLResponse(
            f"<p>The GitHub account <strong>{login_name}</strong> is not on the allowlist for this app.</p>",
            status_code=403,
        )
    request.session["login"] = login_name
    return RedirectResponse("/")


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Client Portal Starter</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.8/css/bootstrap.min.css">
</head>
<body>
<main class="container py-4">
  <div id="flash"></div>
  <!-- screen:content -->
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h1 class="h3 mb-0">Client Portal Starter</h1>
  </div>
  <div id="app"></div>
  <!-- /screen:content -->
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mithril/2.3.6/mithril.min.js"></script>
<script>
/* ===== support library - paste verbatim into every page, never edit ===== */
(function () {
  "use strict";
  function panic(text) {
    var el = document.getElementById("panic");
    if (!el) {
      el = document.createElement("pre");
      el.id = "panic";
      el.style.cssText =
        "position:fixed;left:0;right:0;bottom:0;z-index:9999;margin:0;" +
        "max-height:45vh;overflow:auto;padding:12px 16px;background:#7f1d1d;" +
        "color:#fff;font:12px/1.5 monospace;white-space:pre-wrap;";
      document.body.appendChild(el);
    }
    el.textContent += (el.textContent ? "  |  " : "") + text;
  }
  window.addEventListener("error", (e) =>
    panic((e.message || "Script error") + " (" + (e.filename || "") + ":" + (e.lineno || "?") + ")")
  );
  window.addEventListener("unhandledrejection", (e) =>
    panic("Unhandled rejection: " + String((e.reason && e.reason.stack) || e.reason))
  );
  var flashes = [];
  var flashId = 0;
  function flash(text, level) {
    flashes.push({ id: ++flashId, text: text, level: level || "danger" });
    m.redraw();
  }
  var Flash = {
    view: () =>
      flashes.map((f) =>
        m(".alert.alert-dismissible", { key: f.id, class: "alert-" + f.level, role: "alert" }, [
          f.text,
          m("button.btn-close", {
            type: "button",
            "aria-label": "Close",
            onclick: () => { flashes = flashes.filter((x) => x.id !== f.id); },
          }),
        ])
      ),
  };
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }
  ready(() => {
    var el = document.getElementById("flash");
    if (el) m.mount(el, Flash);
  });
  var PARAMS = {};
  new URLSearchParams(location.search).forEach((v, k) => { PARAMS[k] = v; });
  function syncParams() {
    var qs = new URLSearchParams(PARAMS).toString();
    history.replaceState(null, "", qs ? "?" + qs : location.pathname);
  }
  async function call(name, payload) {
    try {
      var res = await fetch("/rpc/" + name, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      return await res.json();
    } catch (err) {
      flash("Request failed: " + String(err));
      return { ok: false, message: String(err) };
    } finally {
      m.redraw();
    }
  }
  window.rpc = {
    query: (name, params) => call(name, params),
    command: (name, body) => call(name, body),
  };
  function bind(state, key) {
    return {
      value: state[key] == null ? "" : state[key],
      oninput: (e) => { state[key] = e.target.value; },
    };
  }
  bind.check = (state, key) => ({
    checked: !!state[key],
    onchange: (e) => { state[key] = e.target.checked; },
  });
  window.ui = {
    mount: (id, Component) => {
      var el = document.getElementById(id);
      if (!el) { panic("ui.mount('" + id + "'): no element with that id."); return; }
      m.mount(el, Component);
    },
    submit: (fn) => (e) => { e.preventDefault(); return fn(e); },
    bind: bind,
    fieldClass: (inv, name) => (inv && inv[name] ? "is-invalid" : ""),
    fieldError: (inv, name) =>
      inv && inv[name] ? m(".invalid-feedback.d-block", [].concat(inv[name]).join(" ")) : null,
    formError: (inv) => {
      var msgs = inv && (inv.non_field_errors || inv.detail);
      return msgs ? m(".alert.alert-danger", [].concat(msgs).join(" ")) : null;
    },
    flash: flash,
    fmt: {
      date: (v) =>
        v ? new Date(v).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }) : "",
      datetime: (v) => (v ? new Date(v).toLocaleString() : ""),
      money: (v) => (v == null ? "" : "$" + Number(v).toFixed(2)),
    },
    param: (name) => (name in PARAMS ? PARAMS[name] : null),
    setParams: (patch) => {
      Object.entries(patch || {}).forEach(([k, v]) => {
        if (v === null || v === undefined || v === "") delete PARAMS[k];
        else PARAMS[k] = v;
      });
      syncParams();
    },
    go: (path) => location.assign(path),
    copy: (text, message) =>
      navigator.clipboard.writeText(text).then(() => flash(message || "Copied to clipboard.", "success")),
    pageData: () => null,
    debounce: (fn, ms) => {
      var t;
      return function () {
        var args = arguments;
        clearTimeout(t);
        t = setTimeout(() => { fn.apply(null, args); m.redraw(); }, ms || 300);
      };
    },
  };
})();
</script>
<script>
/* screen:scripts */
function App() {
  let user; // undefined while auth.me is in flight, null when signed out
  let contacts = null;
  const filters = { q: ui.param("q") || "" };
  const form = { name: "", email: "" };
  let inv = null;
  let saving = false;

  function can(scope) {
    return user && (user.scopes.includes("*") || user.scopes.includes(scope));
  }
  async function boot() {
    const r = await rpc.query("auth.me");
    if (r.ok) {
      user = r.data;
      if (can("contact.read")) load();
    }
  }
  async function load() {
    ui.setParams(filters);
    const r = await rpc.query("contact.list", filters);
    if (r.ok) contacts = r.data;
  }
  const search = ui.debounce(load, 300);
  async function save() {
    saving = true;
    inv = null;
    const r = await rpc.command("contact.save", form);
    saving = false;
    if (r.ok) {
      ui.flash("Saved " + r.data.name + ".", "success");
      form.name = "";
      form.email = "";
      load();
      return;
    }
    if (r.invalid) inv = r.invalid;
    else if (r.message) ui.flash(r.message);
  }
  async function logout() {
    const r = await rpc.command("auth.logout");
    if (r.ok) { user = null; ui.flash("Signed out.", "success"); }
  }

  const loginScreen = () =>
    m(".row.justify-content-center.mt-5",
      m(".col-md-5",
        m(".card.text-center",
          m(".card-body.py-5", [
            m("h1.h4.mb-2", "Sign in"),
            m("a.btn.btn-dark.btn-lg", { href: "/login" }, "Log in with GitHub"),
          ]))));

  const contactRow = (c) =>
    m("tr", { key: c.id }, [
      m("td", c.name),
      m("td", c.email),
      m("td", ui.fmt.date(c.created)),
    ]);

  const contactTable = () =>
    contacts === null
      ? m("p.text-muted", "Loading…")
      : contacts.length === 0
        ? m("p.text-muted", "No contacts match.")
        : m("table.table.align-middle", [
            m("thead", m("tr", [m("th", "Name"), m("th", "Email"), m("th", "Added")])),
            m("tbody", contacts.map(contactRow)),
          ]);

  const addForm = () =>
    m("form.card.card-body.mb-4", { onsubmit: ui.submit(save) }, [
      m("h2.h6", "Add a contact"),
      ui.formError(inv),
      m(".row.g-2", [
        m(".col-md-5", [
          m("input.form-control", { type: "text", placeholder: "Name",
            class: ui.fieldClass(inv, "name"), ...ui.bind(form, "name") }),
          ui.fieldError(inv, "name"),
        ]),
        m(".col-md-5", [
          m("input.form-control", { type: "email", placeholder: "Email",
            class: ui.fieldClass(inv, "email"), ...ui.bind(form, "email") }),
          ui.fieldError(inv, "email"),
        ]),
        m(".col-md-2.d-grid",
          m("button.btn.btn-primary", { type: "submit", disabled: saving },
            saving ? "Saving…" : "Save")),
      ]),
    ]);

  boot();
  return {
    view: () =>
      user === undefined ? m("p.text-muted", "Loading…")
      : user === null ? loginScreen()
      : m("div", [
          m(".d-flex.justify-content-between.align-items-center.mb-3", [
            m("div", [
              m("span.me-2", user.label),
              user.scopes.map((s) => m("span.badge.text-bg-secondary.me-1", { key: s }, s)),
            ]),
            m("button.btn.btn-outline-secondary", { onclick: logout }, "Log out"),
          ]),
          can("crm.write") ? addForm() : null,
          can("contact.read")
            ? m("div", [
                m("input.form-control.mb-3", {
                  type: "search",
                  placeholder: "Search contacts…",
                  value: filters.q,
                  oninput: (e) => { filters.q = e.target.value; search(); },
                }),
                contactTable(),
              ])
            : m("p.text-muted", "Your account has no contact permissions."),
        ]),
  };
}
ui.mount("app", App);
/* /screen:scripts */
</script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@click.command()
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=8000, type=int, help="Bind port")
def main(host: str, port: int) -> None:
    """Run the portal."""
    if not SESSION_SECRET:
        click.echo(
            "SESSION_SECRET is unset: using a throwaway secret, sessions reset on restart."
        )
    if not GITHUB_CLIENT_ID:
        click.echo(
            "GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET unset: GitHub login disabled until configured."
        )
    click.echo(f"{len(USERS)} user(s), {len(API_KEYS)} API key(s), base URL {BASE_URL}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
