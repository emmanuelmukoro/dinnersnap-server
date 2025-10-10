// api/analyze.js — Vercel serverless function
import fetch from "node-fetch";

// Calls Google Cloud Vision (Label Detection) with a base64-encoded image
async function detectLabelsBase64(base64Content) {
  const resp = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GCV_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requests: [{
          image: { content: base64Content },
          features: [{ type: "LABEL_DETECTION", maxResults: 20 }]
        }]
      })
    }
  );

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(`Vision API error ${resp.status}: ${JSON.stringify(data)}`);
  }

  const labels = data?.responses?.[0]?.labelAnnotations ?? [];
  // Normalize to a simple pantry array: lower-case, de-dupe, confidence threshold
  const pantry = Array.from(
    new Set(
      labels
        .filter(l => (l.score ?? 0) >= 0.6)  // tune threshold if needed
        .map(l => l.description.toLowerCase())
    )
  );
  return pantry;
}

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Use POST" });
    }

    // Support either req.json() (Vercel runtime) or traditional body
    const { imageBase64, prefs } = (await (async () => {
      try { return await req.json(); } catch { return req.body; }
    })()) || {};

    if (!imageBase64) {
      return res.status(400).json({ error: "imageBase64 required" });
    }

    // Strip data URL header if present
    const base64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const pantry = await detectLabelsBase64(base64);

    // Return pantry (detected ingredients). You can add recipe DB/LLM later here.
    return res.status(200).json({
      pantry,
      // (Optional) Return a simple example recipe so your UI can render immediately
      recipes: [
        {
          id: "auto1",
          title: "Quick Tomato & Garlic Pasta",
          time: 18, cost: 2.2, energy: "hob",
          ingredients: ["pasta","tomato","garlic","onion","olive oil"]
            .map(n => ({ name: n, have: pantry.includes(n) })),
          swaps: [{ from: "olive oil", to: "butter", note: "1 tbsp" }],
          steps: [
            { id: "s1", text: "Boil salted water.", secs: 60 * 8 },
            { id: "s2", text: "Sauté garlic & onion.", secs: 60 * 5 },
            { id: "s3", text: "Add tomatoes; simmer.", secs: 60 * 4 },
            { id: "s4", text: "Cook pasta; toss with sauce.", secs: 60 * 2 },
          ]
        }
      ]
    });

  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Server error", details: String(e) });
  }
}
