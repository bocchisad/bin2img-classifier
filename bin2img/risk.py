"""Unified risk score (0–100) and product verdict."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bin2img.model import PredictionResult
from bin2img.rules import RuleHit

# Family prior risk contribution when model is confident.
_FAMILY_RISK: dict[str, int] = {
    "Benign_Code": 5,
    "Downloader_TextHeavy": 55,
    "Packed_PE": 65,
    "Ransomware_Encrypted": 90,
    "Unknown": 35,
}


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    score: int  # 0..100
    verdict: str  # Benign | Suspicious | Malicious | Unknown
    factors: list[str]
    rule_hits: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_risk(
    *,
    classification: PredictionResult | None,
    rule_hits: list[RuleHit],
    risk_tags: list[str],
) -> RiskAssessment:
    """Combine model + rules into a SOC-friendly score/verdict."""
    score = 0
    factors: list[str] = []

    # Rules
    rule_score = sum(h.weight for h in rule_hits)
    score += min(55, rule_score)
    if rule_score:
        factors.append(f"rules_weight={rule_score}")

    # Model family prior × confidence
    if classification is not None:
        family = classification.raw_label or classification.label
        if classification.is_unknown:
            family = "Unknown"
            factors.append("model_unknown")
        prior = _FAMILY_RISK.get(family, 40)
        contrib = int(round(prior * max(classification.confidence, 0.0)))
        # Unknown softens contribution
        if classification.is_unknown:
            contrib = min(contrib, 40)
        score += min(45, contrib)
        factors.append(f"family={family}:{contrib}")

    if "HIGH_ENTROPY_PACKED" in risk_tags and score < 30:
        score += 10
        factors.append("entropy_boost")

    score = int(max(0, min(100, score)))

    if classification is not None and classification.is_unknown and score < 50:
        verdict = "Unknown"
    elif score >= 70:
        verdict = "Malicious"
    elif score >= 35:
        verdict = "Suspicious"
    else:
        verdict = "Benign"

    return RiskAssessment(
        score=score,
        verdict=verdict,
        factors=factors,
        rule_hits=[h.to_dict() for h in rule_hits],
    )
