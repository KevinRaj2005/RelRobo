# LIBRARIES
import os
import glob
import json
import re
import requests
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from rapidfuzz import fuzz, process
import faiss
import mariadb
from dotenv import load_dotenv

load_dotenv()

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

# CONFIG
INFO_DIR = "./Info"
KEYWORD_FILE = "./keyword_rules.json"
EMPLOYEE_API_URL = "https://68a8bc77b115e67576e9abab.mockapi.io/Employee"

# Open Router LLM
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Local semantic search gate
SIMILARITY_THRESHOLD = 0.72

# Scraping
WEBSITES = [
    "https://www.rinfra.com",
    "https://www.reliancepower.co.in",
    
   
]
COMMON_PATHS = [
    "/", "/projects", "/our-business", "/business", "/generation", "/renewables",
    "/renewable-energy", "/solar", "/sustainability", "/media", "/news","/our websites"
]

MAX_PAGES_PER_SITE = 6
MAX_SNIPPETS = 3
REQUEST_TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SupportBot/1.0; +https://example.local)"}

# Scrape relevance controls
GENERIC_NOISE = [
    "about us", "investor relations", "forms and procedures", "board of directors",
    "founder", "listing particulars", "postal ballot"
]
MIN_SNIPPET_SIM = 0.28  # cosine threshold for snippet acceptance

# FAST API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_state: Dict[str, Dict[str, Any]] = {}

# LOAD DATA
def load_json_files():
    data = {}
    for filepath in glob.glob(os.path.join(INFO_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            data[name] = json.load(f)
    return data

def load_keyword_rules():
    if os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

info_data = load_json_files()


def check_keyword_rules(message):
    message_lower = message.lower()
    best_match = None
    best_score = 0

    for category, rule in keyword_rules.items():
        for kw in rule["keywords"]:
            score = fuzz.partial_ratio(message_lower, kw.lower())
            if score > best_score:
                best_score = score
                best_match = rule

#  Only return if similarity is strong enough
    if best_match and best_score >= 70:
        return best_match["response"]

# Visitor queries
    
    if any(word in msg.lower() for word in ["visitor", "visitor pass", "visitor entry", "visitor gate pass"]):
       return (
        "For Visitor Pass related queries, please contact the Admin Department. "
        "They will help you with visitor entries and approvals."
    )

    # RCB Telebook queries
    if any(word in msg.lower() for word in ["rcb", "rcb telebook", "telebook", "telephone book", "rcb telephone book"]):
      return (
        "The RCB Telebook is Reliance Infrastructure’s internal directory containing employee contact details.\n\n"
        "You can access it here: https://intranet.rinfra.com/rcstelebook/"
    )
    return None

# Load admin contacts JSON
def load_admin_contacts():
    filepath = "./admin_contacts.json"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

admin_contacts = load_admin_contacts()

# Hardcoded keyword rules (instead of external JSON file)
keyword_rules = {
    
    "lost_id": {
        "keywords": ["id", "access card", "identity", "id card", "id lost", "lost id","Access card"],
        "response": {
            "Dept": "Security",
            "Ext No": "4003",
            "Contact Person": "Security Desk",
            "Contact": "022-43034003",
            "Email": "Security.reliancecentre@reliancegroupindia.com"
        }
    },
    
    "lost_belongings": {
        "keywords": ["lost", "missing", "belongings", "wallet", "bag", "keys", "item"],
        "response": {
            "Dept": "Security",
            "Ext No": "4003",
            "Contact Person": "Security Desk",
            "Contact": "022-43034003",
            "Email": "Security.reliancecentre@reliancegroupindia.com"
        }
    },
    "Car_parking": {
        "keywords": ["parking","car parking","bike","scooty","car parking", "vehicle parking", "park my car", "parking near office",],
        "response": {
            "Dept": "Security",
            "Ext No": "4003",
            "Contact Person": "Security Desk",
            "Contact": "022-43034003",
            "Email": "Security.reliancecentre@reliancegroupindia.com"
        }
    },
    "access_issue": {
        "keywords": ["Access card","access", "entry", "gate pass", "building access", "swipe", "access denied","id card","id card renewal","entry card","access card renewal","access card renew","renew access card","entry card"],
        "response": {
            "Dept": "Security",
            "Ext No": "4003",
            "Contact Person": "Security Desk",
            "Contact": "022-43034003",
            "Email": "Security.reliancecentre@reliancegroupindia.com"
        }
    },
    "pc_issue": {
        "keywords": ["pc", "computer", "laptop","laptop charger","IT helpesk","It helpdesk" "system", "machine", "desktop", "attached", "virus", "hacked", "pc","printer","printer is not accessible","server"],
        "response": {
            "Dept": "IT Support",
            "Ext No": "4444",
            "Contact Person": "IT Helpdesk",
            "Contact": "022-43034444",
            "Email": "Reliance.ITSupport@reliancegroupindia.com"
        }
    },
    "cyber_attack": {
        "keywords": [ "phishing", "malware", "ransomware", "data breach", "security incident", "cybersecurity","Internet","attacked"],
        "response": {
            "Dept": "IT Support Team",
            "Ext No": "4444",
            "Contact Person": "IT Support Team",
            "Contact": "022-43034444",
            "Email": "Reliance.ITSupport@reliancegroupindia.com"
        }
    },
    "meeting_room": {
        "keywords": ["meeting", "book room", "conference", "schedule meeting", "reserve room", "meeting hall", "discussion room","need to book room","meeting room","meeting room booking"],
        "response": {
            "Dept": "Meeting Room Booking",
            "Ext No": "4001",
            "Contact Person": "Reception (Board)",
            "Contact": "022-43034011",
            "Email": "Ugfloorsecurity.Rcb@reliancegroupindia.com"
        }
    },
    "cafeteria": {
        "keywords": ["cafeteria", "canteen", "food", "what is in lunch", "what is in snacks", "mess", "tea", "coffee","todays menu","paper cup"],
        "response": {
            "Dept": "Admin",
            "Ext No": "4005",
            "Contact Person": "Cafeteria Support",
            "Contact": "022-43034005",
            "Email": "um.rcballardestate@afoozo.com"
        }
    },
    "electricity": {
        "keywords": ["electricity", "power supply", "lights", "electric", "plug", "socket", "power cut", "electric","light ","bulb fuse","power supply","cable wires","wires"],
        "response": {
            "Dept": "Electrician",
            "Ext No": "4033",
            "Contact Person": "Electrician",
            "Contact": "022-43034008",
            "Email": "Bms.Rcb@reliancegroupindia.com"
        }
    },
    "water_supply": {
        "keywords": ["water", "drinking water", "tap", "water supply", "no water", "water","water supply","Coffee Machine","water supplier"],
        "response": {
            "Dept": "CCD - Coffee Café Day/Water Supply",
            "Ext No": "4022",
            "Contact Person": "CCD - Coffee Café Day/Water Supply",
            "Contact": "022-43034024",
            "Email": "NA"
        }
    },
    "ac_issue": {
        "keywords": ["ac", "air conditioning", "cooling", "air conditioner", "hot", "temperature", "hvac","ac repairing"],
        "response": {
            "Dept": "Electrician",
            "Ext No": "4008",
            "Contact Person": "Facilities Desk",
            "Contact": "022-43034035",
            "Email": "Bms.Rcb@reliancegroupindia.com"
        }
    },
     "Landline": {
        "keywords": ["LAN","LAN cable","LAN wires","Lan connection","landline","telebook","phone","Landline working"],
        "response": {
            "Dept": "IT team",
            "Ext No": "4444",
            "Contact Person": "IT Team / IT Network Team",
            "Contact": "022-43034652",
            "Email": "Reliance.ITSupport@reliancegroupindia.com"
        }
    },
    "desktop_allocation": {
        "keywords": ["new pc","new laptop","new keyborad","cells","cell ", "extension board","extension board"],
        "response": {
            "Dept": "IT team",
            "Ext No": "4444",
            "Contact Person": "IT Team / IT Network Team",
            "Contact": "022-43034652",
            "Email": "Reliance.ITSupport@reliancegroupindia.com"
        }
    },
    "desk_allocation": {
        "keywords": ["chair", "almari chabi","table is broken", "cabin keys",  "chairs","chair is broken","table","carpet","new table","new desk","newdesk"],
        "response": {
            "Dept": "Admin",
            "Ext No": "4024",
            "Contact Person": "Maintenance & Repairs",
            "Contact": "022-43034024",
            "Email": "NA"
        }
    },
    "Housekeeping & Cleanliness": {
        "keywords": ["toilet","toilet not clean","housekeeping","flush","office toilet","toilet flush","stinking","smelling","Room freshner","dustbin","trashbin","waste", "disposal", "kachra dabba", "e-waste", "trash", "bin","i  dustbin near my desk","dustbin needed","toilet paper","tissue paper","room freshner","freshner"],
        "response": {
            "Dept": "Housekeeping & Cleanliness / Waste Disposal / E-waste",
            "Ext No": "4006",
            "Contact Person": "Housekeeping",
            "Contact": "022-43034035",
            "Email": "um.rcballardestate@afoozo.com"
        }
    },
     "Admin_access": {
        "keywords": ["keyboard","my keyboard","Admin access","cells","battery for mouse","My mouse","batteries"],
        "response": {
            "Dept": "IT Support Team",
            "Ext No": "4444",
            "Contact Person": "IT Support",
            "Contact": "4444",
            "Email": "Reliance.ITSupport@reliancegroupindia.com"
        }
    },
    "RCB_Telebook": {
    "keywords": ["rcb telebook", "telebook", "rcb directory"],
    "response": {
        "Dept": "Internal Directory",
        "Link": "http://10.8.61.42/rcb-directory/",
        "Note": "You can access the RCB Telebook from the above link using your employee credentials."
    }
},

    "Visitors": {
    "keywords": ["visitor", "guest visitor"],
    "response": {
        "Dept": "Admin",
        "Ext No": "4011",
        "Contact Person": "Milind Bagkar",
        "Contact": "NA",
        "Email": "Milind.Bagkar@reliancegroupindia.com",
        "Note": "Try contacting the concerned department for visitor-related queries."
    }
}
}

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(data_dict):
    texts, keys = [], []
    for filename, records in data_dict.items():
        if isinstance(records, dict):
            for key, value in records.items():
                texts.append(str(value))
                keys.append((filename, key))
        elif isinstance(records, list):
            for idx, item in enumerate(records):
                texts.append(str(item))
                keys.append((filename, idx))
    if not texts:
        return None, [], None
    embeddings = model.encode(texts, convert_to_numpy=True)  # not normalized
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, keys, embeddings

faiss_index, faiss_keys, faiss_embeddings = build_faiss_index(info_data)

# HELPERS: Database connection
def get_db_connection():
    return mariadb.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )

# HELPERS: tokenization & intents
STOPWORDS = {
    "contact","details","for","from","department","dept","support","helpdesk","the","a","to","of",
    "about","want","need","find","get","call","reach","phone","mobile","email","ext","extension",
    "number","hr","it","admin","finance","security","facilities","maintenance","query","queries",
    "please","i","me","info","information","show","tell","give","person","detail"
}

def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) >= 2]

def extract_name_phrase(message_raw: str) -> Optional[str]:
    """
    Find 'First Last' style names from query.
    Find 'John Smith' style names from the original-cased query.
    """
    m = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", message_raw)
    if m:   
        return m[0] 
    tokens = message_raw.strip().split()
    if len(tokens) >= 2:
        return " ".join(tokens[:2])

    return None

def looks_like_employee_query(message_raw: str) -> bool:
    msg = message_raw.lower()
    has_digits = bool(re.search(r"\b\d{4,}\b", msg))
    if has_digits:
        return True
    if any(w in msg for w in ["contact", "details", "extension", "mobile", "email", "phone", "reach"]):
        return True
    if extract_name_phrase(message_raw):
        return True
    return False

# HELPERS: Employee API
def query_employee_db_best(query: str) -> Optional[Dict[str, Any]]:
    """
    Query employees table and score by relevance.
    Prioritizes designation and strong name matches.
    Adds fuzzy name matching for better recall.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        employees = [dict(zip(columns, row)) for row in rows]
        cursor.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    q = query.lower()
    name_phrase = extract_name_phrase(query)
    tokens = [t for t in tokenize(query) if t not in STOPWORDS]

    best_emp = None
    best_score = 0

    for emp in employees:
        full_name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip().lower()
        designation = str(emp.get("designation", "")).lower()
        extension = str(emp.get("extension_number", ""))
        contact = str(emp.get("contact", ""))
        email = str(emp.get("email", ""))

        fields_str = " ".join([full_name, designation, email, contact, extension]).lower()
        score = 0

        # 🔹 Strong designation match (give high weight)
        if any(t in designation for t in tokens):
            score += 8  

        # 🔹 Strong full-name match
        if name_phrase:
            if name_phrase.lower() in full_name:
                score += 6
            else:
                # Fuzzy match boost (handles case + minor typos)
                similarity = fuzz.ratio(name_phrase.lower(), full_name)
                if similarity > 80:  # tweak threshold if needed
                    score += 10  

        else:
            # Fallback: token overlap for partial names
            name_tokens = set(tokenize(full_name))
            score += 2 * len(name_tokens.intersection(tokens))

        # 🔹 General token overlap with all fields
        for t in tokens:
            if t and t in fields_str:
                score += 1

        # 🔹 Digit overlap for extension/contact
        for d in re.findall(r"\d{3,}", q):
            if d in extension or d in contact:
                score += 3

        if score > best_score:
            best_score = score
            best_emp = emp

    return best_emp if best_score > 0 else None


def format_employee(emp: Dict[str, Any]) -> str:
    """Format employee details into a plain text response."""
    lines = []
    full_name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip()
    lines.append(f"Contact details for {full_name}")

    if emp.get("designation"):
        lines.append(f"Designation: {emp['designation']}")
    if emp.get("contact"):
        lines.append(f"Phone: {emp['contact']}")
    if emp.get("extension_number"):
        lines.append(f"Extension: {emp['extension_number']}")
    if emp.get("email"):
        lines.append(f"Email: {emp['email']}")

    return "\n".join(lines)

# HELPERS: Local Info (FAISS)
def search_info(query: str, top_k=3) -> List[Any]:
    """Search local JSON with FAISS and a similarity + token overlap gate."""
    if faiss_index is None:
        return []

    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, top_k)
    results: List[Any] = []

    query_tokens = set(tokenize(query))

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        # crude similarity from L2 distance (vectors not normalized)
        similarity = 1 - min(max(score, 0.0), 2.0) / 2.0

        filename, key = faiss_keys[idx]
        val = info_data[filename]
        item = val[key] if isinstance(val, (dict, list)) else val

        # --- NEW: accept if either similarity OR token overlap ---
        title_tokens = set(tokenize(item.get("title", ""))) if isinstance(item, dict) else set()
        if similarity >= SIMILARITY_THRESHOLD or query_tokens.intersection(title_tokens):
            results.append(item)

    return results


def format_contact_response(contact: Dict[str, Any]) -> str:
    """Format contact details with Markdown links."""
    lines = []
    if "Dept" in contact:
        lines.append(f"*Department:* {contact['Dept']}")
    if "Ext No" in contact:
        lines.append(f"*Extension:* {contact['Ext No']}")
    if "Contact Person" in contact:
        lines.append(f"*Contact Person:* {contact['Contact Person']}")
    if "Contact" in contact:
        lines.append(f"*Phone:* {contact['Contact']}")
    if "Email" in contact:
        lines.append(f"*Email:* {contact['Email']}")
    return "\n".join(lines)

def format_info_result(item: Any) -> str:
    """Format arbitrary local info objects nicely."""
    if isinstance(item, dict):
        if "title" in item and "steps" in item and isinstance(item["steps"], list):
            steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(item["steps"])])
            return f"{item['title']}\n\n{steps}"
        if any(k in item for k in ["Ext No", "Dept", "Contact", "Contact Person", "Email"]):
            return format_contact_response(item)
        # Generic dict → bullets
        return "\n".join([f"- *{k}:* {v}" for k, v in item.items()])
    if isinstance(item, list):
        return "\n".join(map(str, item))
    return str(item)

# HELPERS: Special Rules (Admin, Finance, etc.)
def check_special_rules(message: str) -> Optional[str]:
    msg = message.lower()
     
    # 1) RCB Telebook
    if "rcb telebook" in msg or "telebook" in msg or "rcb directory" in msg:
        return "For RCB telebook, please refer the below link:\n\nhttp://10.8.61.42/rcb-directory/"

    # 2) Meeting Room booking
    meeting_keywords = ["meeting room", "reserve room", "conference hall", "book room", "room booking", "schedule room"]
    if any(k in msg for k in meeting_keywords):
        return (
            "For Meeting Room booking, please contact Reception (Board):\n\n"
            + format_contact_response({
                "Dept": "Reception (Board)",
                "Ext No": "4003",
                "Contact Person": "Reception (Board)",
                "Contact": "022-43034001",
                "Email": "Ugfloorsecurity Rcb/REL/RelianceADAA"
            })
        )

    # 3) Admin Access → IT Support
    if "admin access" in msg or "access for admin" in msg:
        return (
            "For Admin Access issues, please contact IT Support:\n\n"
            + format_contact_response({
                "Dept": "IT Support",
                "Ext No": "4444",
                "Contact Person": "IT Helpdesk",
                "Contact": "022-43034444",
                "Email": "Reliance.ITSupport@reliancegroupindia.com"
            })
        )
    
    # 4) Access card
    if "identity card" in msg or "entry card" in msg or "gate pass" in msg:
        return (
            "For Entry pass issues, please contact Security Department:\n\n"
            + format_contact_response({
                "Dept": "Security",
                "Ext No": "4003",
                "Contact Person": "Security",
                "Contact": "022-43034444",
                "Email": "Security Reliancecentre/Services/RCL/RelianceADA"
            })
        )
    
    # 5) Finance queries
    if "finance" in msg or "finance team" in msg or "finance people" in msg:
        return "Sorry 🙏 I don’t have access to Finance-related data. Please contact the Finance department directly."
    
    # 6) Legal queries
    if "legal" in msg or "legal team" in msg:
        return "Sorry 🙏 I don’t have access to Legal-related data. Please contact the Legal department directly."

    return None

# HELPERS: Keyword Rules
def check_keyword_rules(message: str) -> Optional[Dict[str, Any]]:
    message_lower = message.lower()
    best_match = None
    best_score = 0

    for category, rule in keyword_rules.items():
        for kw in rule["keywords"]:
            # Use token_sort_ratio (better for sentence-like inputs)
            score = fuzz.token_sort_ratio(message_lower, kw.lower())
            if score > best_score:
                best_score = score
                best_match = rule

    # Lower threshold (60) so it works even with loose queries
    if best_match and best_score >= 60:
        return best_match["response"]

    return None

# HELPERS: Admin Contacts JSON
def search_admin_contacts(query: str) -> Optional[Dict[str, Any]]:
    """Search the admin_contacts.json for a department match with fuzzy logic."""
    try:
        with open("./admin_contacts.json", "r", encoding="utf-8") as f:
            admin_data = json.load(f)
    except Exception as e:
        print("Error loading admin_contacts.json:", e)
        return None

    q = query.lower().split()

    for record in admin_data:
        dept = record.get("Dept", "").lower().split()

        # fuzzy match: check overlap between query words & dept words
        if any(word in dept for word in q) or any(word in q for word in dept):
            # Found a match
            response = {
                "contact": record,
                "extra": ""
            }

            # Add booking form link for meeting room requests
            if "meeting" in q or "room" in q or "book" in q:
                response["extra"] = (
                    "\n\n📅 You can also submit a booking request here: "
                    "[Fill Meeting Room Booking Form](https://your-company-booking-form-url)"
                )

            return response

    return None


# HELPERS: Web scraping 

def fetch_url(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        if r.status_code == 200 and ("text/html" in r.headers.get("Content-Type", "").lower()):
            return r.text
    except Exception as e:
        print("fetch error:", url, e)
    return None

def extract_text_blocks(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    blocks = []
    for el in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        txt = el.get_text(separator=" ", strip=True)
        if txt and len(txt.split()) >= 5:
            blocks.append(txt)
    return blocks

def rank_snippets_by_semantic(query: str, snippets: List[Tuple[str, str]]) -> List[Tuple[float, str, str]]:
    if not snippets:
        return []
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    texts = [s[0] for s in snippets]
    t_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = (q_emb @ t_emb.T).flatten()
    ranked = sorted([(float(sims[i]), texts[i], snippets[i][1]) for i in range(len(texts))],
                    key=lambda x: x[0], reverse=True)
    return ranked

def scrape_sites(query: str) -> Optional[str]:
    if not BS4_AVAILABLE:
        return None

    q_lower = query.lower()
    # If solar is mentioned, prioritize solar/renewable paths
    prioritized_paths = []
    if any(k in q_lower for k in ["solar", "renewable", "pv", "photovoltaic"]):
        prioritized_paths = [p for p in COMMON_PATHS if "solar" in p or "renew" in p or "generation" in p]
        if not prioritized_paths:
            prioritized_paths = COMMON_PATHS
    else:
        prioritized_paths = COMMON_PATHS

    collected: List[Tuple[str, str]] = []  # (snippet, url)
    for base in WEBSITES:
        pages = [base.rstrip("/") + path for path in prioritized_paths[:MAX_PAGES_PER_SITE]]
        for u in pages:
            html = fetch_url(u)
            if not html:
                continue
            blocks = extract_text_blocks(html)
            for b in blocks:
                text = b.strip()
                # Filter obvious generic investor/about noise before ranking
                if any(noise in text.lower() for noise in GENERIC_NOISE):
                    continue
                snippet = (text[:200] + "…") if len(text) > 220 else text
                collected.append((snippet, u))

    ranked = rank_snippets_by_semantic(query, collected)
    if not ranked:
        return None

    
# HELPERS: OpenRouter LLM
def query_openrouter(prompt: str) -> str:
    """Fallback to OpenRouter if no local/keyword match is found."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-4o-mini",   # or another available model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"(OpenRouter error {resp.status_code})"
    except Exception as e:
        return f"(Error contacting OpenRouter: {e})"

# API
class ChatRequest(BaseModel):
    message: str

CLOSINGS = ["thanks", "thank you", "thx", "ok", "okay", "got it", "understood"]
GOODBYES = ["bye", "goodbye", "see you", "cya", "quit", "exit"]

@app.post("/ask")
async def ask_endpoint(body: ChatRequest, request: Request):
    user_ip = request.client.host
    message_raw = body.message.strip()
    message = message_raw.lower()
    state = session_state.get(user_ip, {"category": None})

# Special rules first (Admin, Finance, etc.)
    special_reply = check_special_rules(message_raw)
    if special_reply:
        return {"reply": special_reply, "source": "special_rule"}
    
# Goodbye intent
    if any(msg in message for msg in GOODBYES):
        if user_ip in session_state:
            del session_state[user_ip] 
        return {
            "reply": "Goodbye 👋 Have a great day! I’ve reset our session. You can start fresh anytime by saying Good morning!",
            "source": "goodbye"
        }
    # Polite / closing intent
    if any(msg in message for msg in CLOSINGS):
        return {
            "reply": "You're welcome! 😊 Glad I could help. Anything else I can assist you with?",
            "source": "closing"
        }
    # CATEGORY SELECTION
    if message in ["it support", "admin support", "hr support", "human support"]:
        categories = {
            "it support": "Please describe your IT issue.",
            "admin support": "Please describe your Admin-related issue.",
            "hr support": "Please describe your HR-related query.",
        }
        state["category"] = message.split()[0]
        session_state[user_ip] = state
        return {"reply": categories[message], "source": "category"}
    

    # 1) Keyword Rules (Security, PC issues, etc.)
    kw_response = check_keyword_rules(message_raw)
    if kw_response:
        reply_text = (
        "You may try contacting the concerned team for assistance:\n\n"
        + format_contact_response(kw_response)
    )
        return {"reply": reply_text, "source": "keyword_rule"}

   # 2) Employee DB 
    if looks_like_employee_query(message_raw):
        emp = query_employee_db_best(message_raw)
        if emp:
            return {"reply": format_employee(emp), "source": "employee_db"}


    # 3) Local Info JSON (FAISS)
    local_hits = search_info(message_raw, top_k=3)
    if local_hits:
        return {"reply": format_info_result(local_hits[0]), "source": "info_json"}

    # 4) Website scraping (relevance-limited & semantic-ranked)
    web_ans = scrape_sites(message_raw)
    if web_ans:
        return {"reply": web_ans, "source": "website"}

    # 5) Final fallback 

    llm_reply = query_openrouter(message_raw)
    return {
        "reply": llm_reply,
        "source": "openrouter"
    }