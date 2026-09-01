import assert from "node:assert/strict";
import test from "node:test";

import worker from "./index.ts";
import { parseJudgeResult } from "./judge-result.ts";

test("parseJudgeResult accepts only the exact bounded three-decision shape", () => {
  assert.deepEqual(parseJudgeResult({ decision: "Accept", review: "Clear review." }), {
    decision: "Accept",
    review: "Clear review.",
  });
  assert.equal(parseJudgeResult({ decision: "Strong Reject", review: "Old." }), null);
  assert.equal(parseJudgeResult({ decision: "require human review", review: "Old." }), null);
  assert.equal(parseJudgeResult({ decision: "Accept", review: "Clear.", extra: true }), null);
  assert.equal(parseJudgeResult(["Accept", "Clear."]), null);
  assert.equal(parseJudgeResult({ decision: "Accept", review: "x".repeat(100_000) }), null);
});

test("worker rejects incomplete OpenAI responses even when they contain review text", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    status: "incomplete",
    output_text: JSON.stringify({ decision: "Accept", review: "Partial review." }),
  }));

  try {
    const response = await worker.fetch(
      new Request("https://judge.example/v1/reviews/task-proposal", {
        method: "POST",
        headers: {
          "Authorization": "Bearer judge-key",
          "X-OpenAI-API-Key": "Bearer openai-key",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ proposal: "A proposal." }),
      }),
      {
        JUDGE_API_KEY: "judge-key",
        RUBRICS: { get: async () => "Private rubric." },
      },
    );

    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), { error: "OpenAI returned an incomplete response" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
