def score_lead(args):
    revenue = float(args.get("revenue_estimate", 0))
    score = min(100, revenue / 10000 + 30)
    return {
        "lead_name": args.get("name"),
        "company": args.get("company"),
        "score": round(score, 1),
        "recommendation": "qualified" if score > 70 else "nurture"
    }
