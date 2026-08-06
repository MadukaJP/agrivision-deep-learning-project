"""
app/treatment_data.py — AgriVision AI

Static lookup table: class_name -> treatment recommendation text.
This is intentionally NOT a generative model — a simple, reliable lookup
is the right scope for this part of the assignment. Expand/edit the text
below with more specific local guidance if you have access to Nigerian
agricultural extension resources.
"""

TREATMENTS = {
    "cassava_bacterial_blight": "Remove and burn infected plant debris. Avoid overhead irrigation. Use certified disease-free cuttings for next planting. Copper-based bactericides can help in early stages.",
    "cassava_brown_streak_disease": "No chemical cure — remove and destroy infected plants immediately to prevent whitefly spread. Plant only certified virus-free cuttings from disease-free gardens.",
    "cassava_green_mottle": "Uproot and destroy infected plants. Control whitefly vectors with appropriate insecticide. Source planting material from verified disease-free stock.",
    "cassava_mosaic_disease": "Remove infected plants early. Control whitefly populations (main vector). Use resistant cassava varieties where available for replanting.",
    "cassava_healthy": "No action needed. Continue routine monitoring and good field sanitation.",

    "maize_gray_leaf_spot": "Rotate crops away from maize/corn for at least one season. Use resistant hybrids. Fungicide application (e.g. strobilurin-based) if detected early in the season.",
    "maize_common_rust": "Apply fungicide if severe and detected early. Plant rust-resistant maize varieties. Avoid dense planting to improve airflow.",
    "maize_northern_leaf_blight": "Use resistant hybrids. Rotate with non-host crops. Fungicide treatment can reduce spread if caught early.",
    "maize_healthy": "No action needed. Continue routine monitoring.",

    "tomato_bacterial_spot": "Use copper-based bactericide sprays. Avoid working in fields when plants are wet. Remove and destroy severely infected plants.",
    "tomato_early_blight": "Apply fungicide (chlorothalonil or copper-based). Remove lower infected leaves. Ensure adequate plant spacing for airflow.",
    "tomato_late_blight": "Apply fungicide immediately — late blight spreads fast and can destroy a crop within days. Remove and destroy infected plants. Avoid overhead watering.",
    "tomato_leaf_mold": "Improve ventilation (common in humid/greenhouse conditions). Apply fungicide if severe. Avoid leaf wetness.",
    "tomato_septoria_leaf_spot": "Remove infected lower leaves. Apply fungicide. Avoid overhead irrigation and rotate crops.",
    "tomato_spider_mites": "Apply miticide or insecticidal soap. Increase humidity around plants (mites thrive in dry conditions). Introduce natural predators (e.g. ladybugs) where feasible.",
    "tomato_target_spot": "Apply fungicide. Remove infected plant debris. Rotate crops and avoid dense planting.",
    "tomato_yellow_leaf_curl_virus": "No cure — remove and destroy infected plants immediately. Control whitefly vector (main transmission route). Use resistant varieties for replanting.",
    "tomato_mosaic_virus": "No cure — remove infected plants. Disinfect tools between plants. Avoid tobacco use near plants (can carry related virus strains).",
    "tomato_healthy": "No action needed. Continue routine monitoring.",

    "pepper_bacterial_spot": "Apply copper-based bactericide. Avoid overhead irrigation. Remove and destroy infected plant debris. Use disease-free seed.",
    "pepper_healthy": "No action needed. Continue routine monitoring.",
}

UNCERTAIN_MESSAGE = (
    "Confidence too low for a reliable diagnosis, or the image may not be a "
    "recognized crop leaf. Please retake the photo with good lighting, a "
    "single leaf filling most of the frame, and consult a local agricultural "
    "extension officer if symptoms persist."
)


def get_treatment(class_name):
    return TREATMENTS.get(class_name, UNCERTAIN_MESSAGE)
