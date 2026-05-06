import requests
import time
import copy

GATEWAY_URL = "http://104.198.29.19/api/v1/chat/completions"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWUsImlhdCI6MTUxNjIzOTAyMiwibW9kZWxzIjoiUXdlbi9Rd2VuMi41LTEuNUItSW5zdHJ1Y3QifQ.DZsAsceq6bzOW50jvVNNQ5kgTjmSMLueDQursuoxsHU"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def send_request(payload, label, index):
    print(f"\n[{index}] Sending {label} request...")
    try:
        start = time.perf_counter()
        resp = requests.post(GATEWAY_URL, headers=HEADERS, json=payload, timeout=120)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"[{index}] Status: {resp.status_code} | Client-side total: {elapsed:.1f}ms")

        if resp.status_code == 200:
            try:
                data = resp.json()
                stats = data.get("stats", {})
                print(f"[{index}] Gateway completion_time: {stats.get('completion_time', 'N/A')}ms")
                print(f"[{index}] Response preview: {json.dumps(data, ensure_ascii=False)[:300]}")
            except Exception:
                print(f"[{index}] Success body: {resp.text[:300]}")
        else:
            print(f"[{index}] Error body: {resp.text[:500]}")
    except Exception as e:
        print(f"[{index}] ERROR: {e}")

    time.sleep(1)


# ============================================================
# TOOL REQUEST BUILDERS - exact style of your working payloads
# ============================================================

def build_score_lead_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "score_lead",
                    "description": "Score lead quality based on revenue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "company": {"type": "string"},
                            "revenue_estimate": {"type": "number"}
                        },
                        "required": ["name", "company", "revenue_estimate"],
                        "additionalProperties": False
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lead_score_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result_type": {"type": "string"},
                        "summary": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "lead_name": {"type": "string"},
                                "company": {"type": "string"},
                                "score": {"type": "number"},
                                "recommendation": {"type": "string"}
                            },
                            "required": ["lead_name", "company", "score", "recommendation"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["result_type", "summary", "data"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


def build_square_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "square",
                    "description": "Compute the square of a number and return it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer"}
                        },
                        "required": ["n"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "square_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "square": {"type": "integer"}
                    },
                    "required": ["n", "square"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


def build_get_customer_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_customer",
                    "description": "Fetch customer profile by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"}
                        },
                        "required": ["customer_id"],
                        "additionalProperties": False
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "customer_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "segment": {"type": "string"},
                        "lifetime_value": {"type": "number"}
                    },
                    "required": ["name", "email", "phone", "segment", "lifetime_value"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


def build_add_3_numbers_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add_3_numbers",
                    "description": "Add 3 numbers and return the sum.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                            "c": {"type": "integer"}
                        },
                        "required": ["a", "b", "c"],
                        "additionalProperties": False
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "add_3_numbers_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "c": {"type": "integer"},
                        "sum": {"type": "integer"}
                    },
                    "required": ["a", "b", "c", "sum"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


def build_sort_array_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sort_array",
                    "description": "Sort an array in ascending order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arr": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        },
                        "required": ["arr"],
                        "additionalProperties": False
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sort_array_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sorted": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    },
                    "required": ["sorted"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


def build_create_ticket_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": user_text
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "issue_type": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["customer_id", "issue_type", "description", "priority"],
                        "additionalProperties": False
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "create_ticket_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ticket_id": {"type": "string"}
                    },
                    "required": ["ticket_id"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        "temperature": 0.0
    }


# ============================================================
# 25 WITH-TOOL QUERIES
# ============================================================

tool_payloads = [
    build_score_lead_payload("New lead: Priya Sharma from TradeCorp, ₹2.5Cr annual revenue. Score her."),
    build_score_lead_payload("New lead: Arjun Mehta from DataNova Systems, 850000 annual revenue. Score him."),
    build_score_lead_payload("New lead: Kiran Joseph from Nexa Manufacturing, 1250000 annual revenue. Score this lead."),
    build_score_lead_payload("New lead: Dev Patel from ByteForge Analytics, 640000 annual revenue. Score the lead."),
    build_score_lead_payload("New lead: Anita Rao from GreenAxis, 920000 annual revenue. Score her."),
    
    build_square_payload("square of 23"),
    build_square_payload("square of 17"),
    build_square_payload("square of 248"),
    build_square_payload("square of 999"),
    build_square_payload("square of 1234"),

    build_get_customer_payload("Get customer profile for CUST001."),
    build_get_customer_payload("Fetch customer profile for CUST001."),
    build_get_customer_payload("Look up customer CUST001."),
    build_get_customer_payload("Retrieve details for customer CUST001."),
    build_get_customer_payload("Show the CRM profile for CUST001."),

    build_add_3_numbers_payload("Add these 3 numbers: 145, 278, 399."),
    build_add_3_numbers_payload("Add 9999, 8888, and 7777."),
    build_add_3_numbers_payload("Sum these numbers: 12345, 54321, 11111."),
    build_add_3_numbers_payload("Please add 456, 789, and 123."),
    build_add_3_numbers_payload("Find the sum of 50, 75, and 125."),

    build_sort_array_payload("Sort this array: [45, 12, 98, 3, 67, 3, 21, 88, 1, 54]."),
    build_sort_array_payload("Sort this array: [100, 54, 2, 78, 65, 33, 11, 90, 4, 4, 18]."),
    build_sort_array_payload("Sort this array: [500, 120, 450, 999, 1, 75, 63, 63, 240]."),
    build_create_ticket_payload("Create a support ticket for customer CUST001 with issue type payment_failure, description 'Payment failed three times during checkout', priority high."),
    build_create_ticket_payload("Create a support ticket for customer CUST001 with issue type login_problem, description 'User cannot log in after password reset', priority medium.")
]


# ============================================================
# 25 WITHOUT-TOOL PAYLOADS - exact style you gave
# ============================================================

def build_plain_payload(user_text):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "Follow instructions strictly."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "max_completion_tokens": 1024,
        "stream": False,
        "stream_options": {
            "include_usage": False
        },
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "store": False
    }


plain_payloads = [
    build_plain_payload("Solve for x: 2x + 5 = 17. Show the steps."),
    build_plain_payload("Solve for x: 3x - 9 = 12. Show the steps."),
    build_plain_payload("Find x if 5x + 10 = 60. Explain step by step."),
    build_plain_payload("Simplify and solve: 2(3x + 4) = 20."),
    build_plain_payload("Solve for x: 7x - 14 = 35."),
    build_plain_payload("A train travels 60 km in 1.5 hours. What is its speed? Show the calculation."),
    build_plain_payload("If 12 workers finish a task in 10 days, how long will 6 workers take?"),
    build_plain_payload("What is the capital of France and why is it historically important?"),
    build_plain_payload("Explain what an API is in simple English."),
    build_plain_payload("Explain machine learning in simple English with one example."),
    build_plain_payload("What is Python memory management?"),
    build_plain_payload("Explain JWT authentication in simple words."),
    build_plain_payload("What is FastAPI used for?"),
    build_plain_payload("Explain round robin load balancing with an example."),
    build_plain_payload("What is the difference between synchronous and asynchronous programming?"),
    build_plain_payload("Why do production systems use retries and timeouts?"),
    build_plain_payload("What does token usage mean in language models?"),
    build_plain_payload("What is structured output in LLM APIs?"),
    build_plain_payload("Why is temperature 0 used in testing?"),
    build_plain_payload("How does an HTTP request travel from client to server and back?"),
    build_plain_payload("What is the difference between chat completions and a response-style API?"),
    build_plain_payload("Why is JSON schema useful in tool-calling workflows?"),
    build_plain_payload("How do you measure latency in an API system?"),
    build_plain_payload("What are common reasons for a 500 internal server error?"),
    build_plain_payload("Explain streaming vs non-streaming responses in chat APIs.")
]


# ============================================================
# RUN
# ============================================================

print("=" * 60)
print("SENDING 25 REQUESTS WITH TOOLS")
print("=" * 60)

for i, payload in enumerate(tool_payloads, start=1):
    send_request(copy.deepcopy(payload), "WITH_TOOL", i)

print("\n" + "=" * 60)
print("SENDING 25 REQUESTS WITHOUT TOOLS")
print("=" * 60)

for i, payload in enumerate(plain_payloads, start=1):
    send_request(copy.deepcopy(payload), "NO_TOOL", i + 25)