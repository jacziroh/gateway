import uuid
from datetime import datetime

TICKETS = {}

def create_ticket(args):
    ticket_id = str(uuid.uuid4())[:8].upper()
    TICKETS[ticket_id] = {
        "customer_id": args.get("customer_id"),
        "issue_type": args.get("issue_type"),
        "description": args.get("description"),
        "priority": args.get("priority", "medium"),
        "status": "open",
        "created_at": datetime.now().isoformat()
    }
    return {"ticket_id": ticket_id}
