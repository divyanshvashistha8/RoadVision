def estimate_severity(detections, image_width, image_height):
    if not detections:
        return {"score": 0.0, "level": "Low", "damage_area_ratio": 0.0, "count": 0}

    total_area = 0.0
    weighted = 0.0
    image_area = max(image_width * image_height, 1)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        area = max(0, x2-x1) * max(0, y2-y1)
        total_area += area
        weighted += area * float(d["confidence"])

    ratio = min(total_area / image_area, 1.0)
    count_factor = min(len(detections) / 10.0, 1.0)
    conf_area = min(weighted / image_area, 1.0)

    score = 100.0 * (0.55 * ratio + 0.25 * count_factor + 0.20 * conf_area)

    if score < 15:
        level = "Low"
    elif score < 40:
        level = "Medium"
    else:
        level = "High"

    return {
        "score": round(score, 2),
        "level": level,
        "damage_area_ratio": round(ratio, 4),
        "count": len(detections),
    }
