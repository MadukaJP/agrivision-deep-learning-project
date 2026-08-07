# BUSINESS_PLAN.md — AgriVision AI

## Problem

Smallholder and commercial farmers across Nigeria (and West Africa
broadly) lose significant crop yield to diseases that go undiagnosed or
misdiagnosed until visible damage is severe. Access to agricultural
extension officers is limited and slow — many farmers rely on guesswork,
word-of-mouth, or delayed in-person visits to identify what's affecting
their crop and what to do about it. By the time a diagnosis happens, the
disease may have already spread across a field.

## Solution

AgriVision AI lets a farmer photograph a crop leaf with their phone and
get an instant, AI-generated diagnosis (cassava, maize, tomato, pepper —
four crops with major footprint in Nigerian agriculture), a visual
explanation of what the model detected (Grad-CAM heatmap over the
affected leaf region), and a concrete treatment recommendation they can
act on immediately, packaged as a downloadable PDF report they can share
with an agro-dealer or extension officer.

## Unique Value Proposition (UVP)

- **Instant, not delayed**: diagnosis in seconds from a phone photo, no
  waiting for an extension visit.
- **Explainable, not a black box**: the Grad-CAM overlay shows *why* the
  model reached its conclusion, building trust with users who are
  understandably skeptical of an unexplained AI verdict on something as
  consequential as their livelihood.
- **Actionable, not just informational**: every diagnosis comes with a
  specific treatment recommendation and a shareable PDF report, not just
  a disease name.
- **Locally relevant crop coverage**: trained specifically on crops with
  major Nigerian agricultural relevance (cassava, maize, tomato, pepper),
  not a generic global crop set.

## Target Audience

- Smallholder farmers
- Commercial farm managers
- Agricultural extension agents (as a triage/productivity tool for their
  own fieldwork)
- Agro-chemical distributors (as a lead-generation and customer-service
  channel)

## Revenue Streams

1. **Freemium Tier**: 3 free leaf diagnostics per week. Drives adoption
   and trust before asking for payment — critical in a market where the
   product category (AI crop diagnostics) is unfamiliar to most users.
2. **Pro Farmer Plan — ₦2,500/month**: unlimited scans, downloadable PDF
   reports, offline-capable PWA (progressive web app) for low-connectivity
   rural areas, automated weekly crop health alerts.
3. **B2B Agro-Dealer Marketplace — ₦15,000/month**: local agro-chemical
   and input dealers pay to be listed as verified treatment suppliers
   surfaced to farmers when a specific disease is diagnosed in their
   geographic area — turning every diagnosis into a potential referral.

## Go-To-Market Strategy

- **Phase 1 (pilot)**: partner with 1-2 agricultural extension programs
  or cooperatives to onboard an initial user base of farmers already
  engaged with formal agricultural support structures — lowers the
  trust barrier for a new AI product.
- **Phase 2 (organic growth)**: leverage the freemium tier and
  word-of-mouth within farming communities/cooperatives, where
  recommendations from a trusted peer carry more weight than direct
  advertising.
- **Phase 3 (B2B expansion)**: once a meaningful diagnosis volume exists
  in a given region, approach local agro-dealers with real usage data to
  sell the marketplace listing tier.

## Illustrative Revenue Projection (Year 1)

These are planning assumptions for a pilot-to-early-growth trajectory,
not market research — useful for the defense presentation to show
you've thought through the model, not as a guaranteed forecast.

| Milestone | Free users | Pro Farmer subscribers | Agro-dealer listings | Monthly revenue |
|---|---|---|---|---|
| Month 3 (pilot) | 500 | 20 | 0 | ₦50,000 |
| Month 6 | 2,000 | 100 | 5 | ₦325,000 |
| Month 12 | 8,000 | 400 | 25 | ₦1,375,000 |

Assumptions: ~5% of active free users convert to Pro Farmer by month 12
(a conservative freemium conversion rate); agro-dealer listings grow
only after enough diagnosis volume exists in a given region to make the
marketplace tier valuable to a dealer (hence 0 at month 3).


- Current model accuracy (58% test set) and known weaknesses (cassava
  disease discrimination, OOD rejection reliability — both detailed in
  `DEFENSE_ANSWERS.md`) mean this is presented as a pilot-stage product,
  not a finished commercial-grade diagnostic tool. A production launch
  would need further data collection (especially more cassava field
  images) and OOD handling improvements before scaling paid tiers.
- Yam, despite being a significant Nigerian staple crop, isn't covered
  due to lack of public labeled training data — a real gap for the
  target market that would need addressing via original data collection
  in a future iteration.
