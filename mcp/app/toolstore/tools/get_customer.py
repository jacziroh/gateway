CRM_DB = {
    "CUST001": {
        "name": "Rajesh Kumar",
        "email": "rajesh@fintech.co.in",
        "phone": "+91-9845012345",
        "segment": "VIP",
        "lifetime_value": 250000
    }
}

def get_customer(args):
    cid = (args.get("customer_id") or "").strip().upper()
    if not cid:
        return {"error": "customer_id is required"}
    return CRM_DB.get(cid, {"error": "Customer not found"})
