const MAX_REVIEW_CHARS = 20_000;

type JudgeResult = {
  decision: "Reject" | "Accept" | "Strong Accept";
  review: string;
};

function isJudgeDecision(value: unknown): value is JudgeResult["decision"] {
  return value === "Reject" || value === "Accept" || value === "Strong Accept";
}

export function parseJudgeResult(value: unknown): JudgeResult | null {
  if (
    !value
    || typeof value !== "object"
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
  ) return null;
  const result = value as Record<string, unknown>;
  if (
    Object.keys(result).length !== 2
    || !("decision" in result)
    || !("review" in result)
    || !isJudgeDecision(result.decision)
    || typeof result.review !== "string"
    || !result.review.trim()
    || result.review.length > MAX_REVIEW_CHARS
  ) return null;
  return { decision: result.decision, review: result.review };
}
