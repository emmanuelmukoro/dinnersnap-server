// api/analyze.js — improved food detection
import fetch from "node-fetch";

// --- Food lexicon (expand anytime) ---
const FOOD_WHITELIST = new Set([
  "banana","broccoli","butternut squash","squash","onion","garlic","tomato","tomatoes",
  "pasta","rice","egg","eggs","chicken","chicken breast","courgette","zucchini","feta",
  "cucumber","carrot","potato","sweet potato","pepper","bell pepper","mushroom","spinach",
  "lettuce","parsley","basil","coriander","lemon","lime","olive oil","butter","milk","yogurt",
  "beans","kidney beans","chickpeas","tuna","salmon","beef","mince","sausage","bread",
  "wrap","tortilla","passata","tomato sauce","canned tomatoes","sauce"
]);

const FOOD_BLACKLIST_SUBSTRINGS = [
  "plastic","container","utensil","cutlery","countertop","table","cabinet","sticker",
  "label","brand","kitchen","furniture","appliance","bottle","jar","packaging","tin"
];

function looksBlacklisted(s) {
  return FOOD_BLACKLIST_SUBSTRINGS.some(bad => s.includes(bad));
}

function canonicalize(name) {
  let s = name.toLowerCase();
  // quick plurals
  if (s.endsWith("es")) s = s.slice(0, -2);
  else if (s.endsWith("s") && !s.endsWith("ss")) s = s.slice(0, -1);
  // synonyms / normalize
  const syn = {
    zucchini: "courgette",
    courgettes: "courgette",
    bell: "pepper",
    peppers: "pepper",
    scallion: "spring onion",
    scallions: "spring onion",
    coriander: "cilantro" // or the other way around, your call
  };
  if (syn[s]) s = syn[s];
  return s;
}

function isFoodWord(s) {
  if (looksBlacklisted(s)) return false;
  if (FOOD_WHITELIST.has(s)) return true;
  // loose allow for obvious produce words not in list
  const loose = ["apple","banana","broccoli","squash","pumpkin","tomato","onion","garlic","ginger",
                 "cucumber","carrot","potato","pepper","mushroom","spinach","lettuce","orange","lemon",
                 "lime","grape","strawberry","blueberry","courgette","zucchini"];
  return loose.includes(s);
}

async function detectFoodBase64(base64Content) {
  const resp = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GCV_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requests: [{
          image: { content: base64Content },
          imageContext: { languageHints: ["en"] },
          features: [
            { type: "OBJECT_LOCALIZATION", maxResults: 50 },
            { type: "LABEL_DETECTION",     maxResults: 50 },
            { type: "WEB_DETECTION",       maxResults: 50 }
          ]
        }]
      })
    }
  );

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(`Vision API ${resp.status}: ${JSON.stringify(data)}`);
  }

  const r = data?.responses?.[0] || {};
  const labels  = r.labelAnnotations || [];
  const objs    = r.localizedObjectAnnotations || [];
  const webEnts = r.webDetection?.webEntities || [];

  const map = new Map(); // name -> {score, src}

  const add = (raw, score = 0.5, src = "label") => {
    if (!raw) return;
    const name = canonicalize(String(raw));
    if (!isFoodWord(name)) return;
    const prev = map.get(name);
    const sc = Math.max(score || 0, prev?.score || 0);
    map.set(name, { score: sc, src });
  };

  labels.forEach(l => add(l.description, l.score, "label"));
  objs.forEach(o => add(o.name, o.score ?? o.confidence ?? 0.55, "object"));
  webEnts.forEach(w => add(w.description, w.score ?? 0.55, "web"));

  // Sort by score, return unique names
  const pantry = Array.from(map.entries())
    .sort((a, b) => b[1].score - a[1].score)
    .map(([name]) => name);

  return pantry;
}

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Use POST" });
    }

    const body = (await (async () => { try { return await req.json(); } catch { return req.body; } })()) || {};
    const { imageBase64, prefs } = body;
    if (!imageBase64) return res.status(400).json({ error: "imageBase64 required" });

    const base64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const pantry = await detectFoodBase64(base64);

    // Optionally: log to Vercel function logs to see raw output while tuning
    console.log("Pantry:", pantry);

    return res.status(200).json({
      pantry,
      // Keep dummy recipes for now (your app still uses local generator)
      recipes: []
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Server error", details: String(e) });
  }
}
